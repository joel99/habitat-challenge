#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from torch.jit import Final
import torch.nn.functional as F

from .belief_policy import (
    MultipleBeliefPolicy,
    AttentiveBeliefCore,
)

r"""
A few different prototypes of attention
- TransformerBelief: direct transformer on belief output. If we have diverse beliefs, their outputs should be transformable into potent representations
- TransformerIMBelief: transformer on belief output feeds back into beliefs on interval.
- HiddenTransformerBelief: transformer on belief hidden states. Communicates on interval

"""
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    n_head: Final[int]
    d_head: Final[int]

    def __init__(self, hidden_size, n_head, dropout_p, prenorm):
        super().__init__()

        self.prenorm = prenorm
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.attn_ln = nn.LayerNorm(hidden_size)

        self.pos_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(2 * hidden_size, hidden_size),
        )
        self.pos_ff_ln = nn.LayerNorm(hidden_size)

        self.n_head = n_head
        self.d_head = hidden_size // n_head
        self.drop_layer = nn.Dropout(p=dropout_p)


        self.register_buffer("_scale", torch.tensor(1.0 / (hidden_size ** self.d_head)))

    def forward(self, x):
        t, n, _ = x.size()
        residual = x
        if self.prenorm:
            x = self.attn_ln(x)
        q, k, v = torch.chunk(self.qkv(x), 3, dim=2)

        q = q.view(t, n, self.n_head, self.d_head)
        k = k.view(t, n, self.n_head, self.d_head)
        v = v.view(t, n, self.n_head, self.d_head)

        logits = torch.einsum("ibhd, jbhd -> ijbh", q, k)
        probs = F.softmax(logits * self._scale, dim=1)
        probs = self.drop_layer(probs)
        attn_out = torch.einsum("ijbh, jbhd -> ibhd", probs, v)
        attn_out = attn_out.reshape(t, n, -1)
        attn_out = self.output_layer(attn_out)

        x = residual + attn_out
        if not self.prenorm:
            x = self.attn_ln(x)
        residual = x
        if self.prenorm:
            x = self.pos_ff_ln(x)
        x = residual + self.pos_ff(x)
        if not self.prenorm:
            x = self.pos_ff_ln(x)
        return x

class TransformerEncoder(nn.Module):
    # ! Unused. Just use PermutedEncoder below
    def __init__(self, hidden_size, n_head, n_layers, dropout_p, prenorm):
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, n_head, dropout_p, prenorm) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class PermutedEncoder(nn.Module):
    r"""
        We typically operate with beliefs as batch x k modules x h.
        Pytorch transformer encoders transform along the first dimension, so we permute twice to accommodate.
    """
    def __init__(self, config, hidden_size, num_tasks, *args, **kwargs):
        # transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=config.num_heads, dropout=config.dropout_p)
        # transformer_norm = nn.LayerNorm(hidden_size)
        # super().__init__(transformer_layer, num_layers=config.num_layers, norm=transformer_norm, *args, **kwargs)
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, config.num_heads, config.dropout_p, config.prenorm) for _ in range(config.num_layers)]
        )
        # num_layers=config.num_layers, norm=transformer_norm, *args, **kwargs)
        self.module_embedder = None
        if config.embed_module:
            self.module_embedder = nn.Embedding(num_tasks + 1, hidden_size)
            self.register_buffer("_module_ids", torch.arange(num_tasks, dtype=torch.long))
        self.message_flag = None
        if config.embed_message_flag:
            self.message_flag = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, modules):
        r"""
            Modules: b x k x h
        """
        modules = modules.permute(1, 0, 2) # k x b x h
        if self.module_embedder is not None:
            module_embeddings = self.module_embedder(self._module_ids)
            modules = modules + module_embeddings.unsqueeze(1)

        for layer in self.layers:
            modules = layer(modules)
        result = modules.permute(1, 0, 2)
        # result = super().forward(modules).permute(1, 0, 2)
        # ^ doesn't seem to work in torchscript

        if self.message_flag is not None:
            result = result + self.message_flag.view(1, 1, -1)
        return result

class LinearMessage(nn.Module):
    r"""
        When we send messages, we need to integrate them into the belief input.
        We'd like to flag them as "messages" (a message embedding) when this happens so the modules can treat the input differently.
        Note: this is different than from belief identity embedding.
        This should be shared across belief modules.
    """
    FLAG_SIZE = 4

    def __init__(self, device, hidden_size):
        super().__init__()
        # ! TODO bring this param back
        self.message_flag = nn.Parameter(torch.zeros(self.FLAG_SIZE, device=device))
        self.fc = nn.Linear(hidden_size + self.FLAG_SIZE, hidden_size)

    def forward(self, message):
        expanded_size = (*message.size()[:-1], self.FLAG_SIZE)
        return self.fc(torch.cat([message, self.message_flag.expand(expanded_size)], dim=-1))

# -----------------------------------------------------------------------------
# Cores/Nets
# -----------------------------------------------------------------------------

class TransformerBelief(AttentiveBeliefCore):
    r"""
        Apply a transformer to the belief outputs. Allows for minimal interaction before fusion.
    """
    def __init__(
        self,
        config=None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.transformer = PermutedEncoder(config.TRANSFORMER, self._hidden_size, self.num_tasks)

    def _fuse_beliefs(self, beliefs, *args):
        # beliefs -> b x k x h
        self_attended_beliefs = self.transformer(beliefs)
        return super()._fuse_beliefs(self_attended_beliefs, *args)

class HiddenTransformerBelief(AttentiveBeliefCore):
    r"""
        Add a transformer step to the belief update directly (over hidden states).
        Communicates at intervals instead of every timestep.
        If transformed, the hidden states returned are the transformed states.
        The hidden states returned are the transformed versions (if transformed).
    """
    def __init__(
        self,
        config=None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.transformer = nn.Sequential(
            PermutedEncoder(config.TRANSFORMER, self._hidden_size, self.num_tasks),
            LinearMessage(self.device, self._hidden_size)
        )
        self.COMM_INTERVAL = config.IM.comm_interval
        self.COMM_RATIO = 1.0 / config.IM.comm_interval

    def forward(self, obs, rnn_hidden_states, masks):
        r"""
            Args:
                obs
                rnn_hidden_states: l x n x k x h
                masks
        """
        n = rnn_hidden_states.size(1)
        t = int(obs.size(0) / n)
        if t == 1: # Acting
            if torch.rand((1,)) < self.COMM_RATIO:
                beliefs, hidden_states = self._step_rnn(obs, rnn_hidden_states, masks)
                top_hidden = self.transformer(hidden_states[-1]).unsqueeze(0)
                rnn_hidden_states = torch.cat((hidden_states[:-1], top_hidden), dim=0)
            else:
                beliefs, rnn_hidden_states = self._step_rnn(obs, rnn_hidden_states, masks)
        else:
            masks = masks.view(t, n, -1)
            time_obs = obs.view(t, n, -1)
            beliefs = []
            # In update, interval is deterministic - we comm every interval
            # TODO unify the logic for this
            for start in range(0, t, self.COMM_INTERVAL):
                end = min(t, start + self.COMM_INTERVAL)
                embeddings, hidden_states = self._step_rnn(
                    time_obs[start:end].flatten(end_dim=1),
                    rnn_hidden_states,
                    masks[start:end].flatten(end_dim=1)
                )
                top_hidden = self.transformer(hidden_states[-1]).unsqueeze(0) # transform the top layer
                rnn_hidden_states = torch.cat((hidden_states[:-1], top_hidden), dim=0)
                beliefs.append(embeddings) # appending (step*n) x k x h
            beliefs = torch.cat(beliefs, 0) # We can concat along time since each step is flattened t * n (time first)
            # So the flattened matrix looks like [n0 n0 n0 n1 n1 n1], and we cat [n2 n2 n2 n3 n3 n3]
        contextual_embedding, weights = self._fuse_beliefs(beliefs, obs)
        return contextual_embedding, rnn_hidden_states, beliefs, weights


# -----------------------------------------------------------------------------
# Policies
# -----------------------------------------------------------------------------

class TransformerBeliefPolicy(MultipleBeliefPolicy):
    def __init__(self, *args, net=TransformerBelief, **kwargs):
        super().__init__(*args, net=net, **kwargs)

class HiddenTransformerBeliefPolicy(MultipleBeliefPolicy):
    def __init__(self, *args, net=HiddenTransformerBelief, **kwargs):
        super().__init__(*args, net=net, **kwargs)

