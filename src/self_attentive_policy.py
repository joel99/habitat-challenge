#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn

from .models import RNNStateEncoder
from .belief_policy import MultipleBeliefPolicy, AttentiveBelief

r"""
A few different prototypes of attention
- TransformerBelief: direct transformer on belief output. If we have diverse beliefs, their outputs should be transformable into potent representations
- TransformerIMBelief: transformer on belief output feeds back into beliefs on interval.
- HiddenTransformerBelief: transformer on belief hidden states. Communicates on interval
"""

class PermutedEncoder(nn.TransformerEncoder):
    def __init__(self, config, hidden_size, *args, **kwargs):
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=config.num_heads)
        transformer_norm = nn.LayerNorm(hidden_size)
        super().__init__(transformer_layer, num_layers=config.num_layers, norm=transformer_norm, *args, **kwargs)

    def forward(self, modules, *args, **kwargs):
        return super().forward(modules.permute(1, 0, 2), *args, **kwargs).permute(1, 0, 2)

class TransformerBelief(AttentiveBelief):
    r"""
        Naive self-attention. Note: we can either self-attend over outputs or hidden states.
        This one is outputs. (just a little easier) See AttentiveCommunicative for the other. We do not feed transformed into next state
    """
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        config,
        **kwargs,
    ):
        super().__init__(observation_space, hidden_size, num_tasks, config, **kwargs)
        self.transformer = PermutedEncoder(config.TRANSFORMER, self._hidden_size)

    def _fuse_beliefs(self, beliefs, *args):
        # beliefs -> b x k x h
        self_attended_beliefs = self.transformer(beliefs)
        return super()._fuse_beliefs(self_attended_beliefs, *args)

class IntervalMessageBelief(AttentiveBelief):
    r"""
        We want to communicate, not disrupt dynamics. Defaults to a transformer
        Transformer output is a message that feeds into your own next step
        - makes it easier to learn to reject
        - message must be in the shape of a regular observation since we only send it at interval
        - defaults to interval messaging
        - messages are generated from hidden state
        Observations in processing loop are custom per module
    """
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        config,
        **kwargs
    ):
        super().__init__(observation_space, hidden_size, num_tasks, config=config, **kwargs)
        self._init_message_path(config)
        self.COMM_INTERVAL = config.IM.comm_interval
        self.COMM_RATIO = 1.0 / self.COMM_INTERVAL

    def _initialize_state_encoder(self):
        self.state_encoders = nn.ModuleList([
            RNNStateEncoder(self._embedding_size, self._hidden_size) for _ in range(self.num_tasks)
        ])

    def _step_rnn(self, obs, rnn_hidden_states, masks):
        r""" Forward obs through time
            obs: custom messages per encoder n/nxt x k x h
            rnn_hidden_states:
        """
        if self.cuda_streams is None:
            outputs = [encoder(obs[:, i], rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
        else:
            outputs = [None] * self.num_tasks
            torch.cuda.synchronize()
            for i, encoder in enumerate(self.state_encoders):
                with torch.cuda.stream(self.cuda_streams[i]):
                    outputs[i] = encoder(obs[:, i], rnn_hidden_states[:, :, i], masks)
            torch.cuda.synchronize()
        embeddings, hidden_states = zip(*outputs) # list of embeddings, hidden states, (txn)xh, (layers)xnxh
        return torch.stack(embeddings, dim=-2), torch.stack(hidden_states, dim=-2)

    def _init_message_path(self, config):
        r""" Make message creation apparati """
        raise NotImplementedError

    def _get_message(self, obs, beliefs): # Ideally we can split and restrict here
        r""" Will merge in a belief message into the observation. If multiple observations, only merge into first.
        args:
            obs: n/n*t x k x h
            beliefs: n x k x h
        Returns:
            obs: same as input
        """
        raise NotImplementedError

    # rnn_hidden_states.size(): num_layers, num_envs, num_tasks, hidden, (only first timestep)
    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        x = self._get_observation_embedding(visual_embedding, observations)
        obs = x.unsqueeze(1).expand(-1, self.num_tasks, -1)
        n = rnn_hidden_states.size(1)
        t = int(visual_embedding.size(0) / n)
        if t == 1:
            if self.COMM_RATIO >= 1 or torch.rand((1,)) < self.COMM_RATIO:
                obs = self._get_message(obs, rnn_hidden_states[-1])
            beliefs, rnn_hidden_states = self._step_rnn(obs, rnn_hidden_states, masks)
        else:
            beliefs = []
            for start in range(0, t, self.COMM_INTERVAL):
                end = min(t, start + self.COMM_INTERVAL)
                message = self._get_message(
                    obs[start * n:end * n],
                    rnn_hidden_states[-1]
                )
                embeddings, rnn_hidden_states = self._step_rnn(
                    message,
                    rnn_hidden_states,
                    masks[start * n:end * n]
                )
                beliefs.append(embeddings) # appending n x k x h
            beliefs = torch.cat(beliefs, 0)
        contextual_embedding, weights = self._fuse_beliefs(beliefs, x)
        return contextual_embedding, rnn_hidden_states, beliefs, weights

class TransformerIMBelief(IntervalMessageBelief):
    r"""
        - each module outputs a message to the world which is going to be encoded
    """
    def _init_message_path(self, config):
        self.transformer = PermutedEncoder(config.TRANSFORMER, self._hidden_size)
        self.input_encoder = nn.Sequential(
            nn.Linear(self._hidden_size + self._embedding_size, self._hidden_size), # take message and observation, create a single input
            nn.ReLU(True),
            nn.Linear(self._hidden_size, self._embedding_size),
            nn.ReLU(True)
        )

    def _get_message(self, obs, beliefs): # Ideally we can split and restrict here
        r"""
        args:
            obs: (t/t*n) x k x h
            beliefs: n x k x h
        returns:
            obs: (t/t*n) x k x h (first message is augmented)
        """
        # This mixes hidden state messages into observations
        messages = self.transformer(beliefs) # n x k x h
        message_obs = self.input_encoder(torch.cat([obs[:beliefs.size(0)], messages], dim=-1)) # n x k x h
        if obs.size(0) > beliefs.size(0):
            message_obs = torch.cat([message_obs, obs[beliefs.size(0):]], dim=0) # cat the sequence back together
        return message_obs

class LinearMessage(nn.Module):
    FLAG_SIZE = 4
    def __init__(self, device, hidden_size):
        super().__init__()
        self.message_flag = torch.zeros(self.FLAG_SIZE, device=device)
        self.fc = nn.Linear(hidden_size + self.FLAG_SIZE, hidden_size)

    def forward(self, message):
        expanded_size = (*message.size()[:-1], self.FLAG_SIZE)
        return self.fc(torch.cat([message, self.message_flag.expand(expanded_size)], dim=-1))

class HiddenTransformerBelief(AttentiveBelief):
    r"""
        Self-attention over hidden states instead of outputs. This means modules are communicating.
        Still finally visually conditioned fusion
        The hidden states we return are the transformed versions
    """
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        config,
        **kwargs
    ):
        super().__init__(observation_space, hidden_size, num_tasks, config, **kwargs)
        self.transformer = nn.Sequential(
            PermutedEncoder(config.TRANSFORMER, self._hidden_size),
            LinearMessage(self.device, self._hidden_size)
        )
        self.COMM_INTERVAL = config.IM.comm_interval
        self.COMM_RATIO = 1.0 / config.IM.comm_interval


    def _step_rnn(self, obs, rnn_hidden_states, masks):
        if self.cuda_streams is None:
            outputs = [encoder(obs, rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
        else:
            outputs = [None] * self.num_tasks
            torch.cuda.synchronize()
            for i, encoder in enumerate(self.state_encoders):
                with torch.cuda.stream(self.cuda_streams[i]):
                    outputs[i] = encoder(obs, rnn_hidden_states[:, :, i], masks)
            torch.cuda.synchronize()
        embeddings, hidden_states = zip(*outputs) # list of embeddings, hidden states, (txn)xh, (layers)xnxh
        hidden_states = torch.stack(hidden_states, dim=-2)
        final_hidden = hidden_states[-1]
        other_hidden = hidden_states[:-1]
        return torch.stack(embeddings, dim=-2), hidden_states, final_hidden, other_hidden

    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        x = self._get_observation_embedding(visual_embedding, observations)
        # rnn_hidden_states.size(): num_layers, num_envs, num_tasks, hidden, (only first timestep)
        n = rnn_hidden_states.size(1)
        t = int(x.size(0) / n)
        if t == 1:
            # if torch.randint(self.COMM_INTERVAL, (1,)) < 1:
            if torch.rand((1,)) < self.COMM_RATIO:
                beliefs, _, final_hidden, other_hidden = self._step_rnn(x, rnn_hidden_states, masks)
                final_hidden = self.transformer(final_hidden).unsqueeze(0)
                rnn_hidden_states = torch.cat((other_hidden, final_hidden), dim=0)
            else:
                beliefs, rnn_hidden_states, _, _ = self._step_rnn(x, rnn_hidden_states, masks)
        else:
            masks = masks.view(t, n, -1)
            obs = x.view(t, n, -1)
            beliefs = []
            # In update, interval is deterministic
            for update in range(math.ceil(t * self.COMM_RATIO)):
                start = update * self.COMM_INTERVAL
                end = min(t, start + self.COMM_INTERVAL)
                embeddings, _, final_hidden, other_hidden = self._step_rnn(
                    obs[start:end].view(-1, self._embedding_size),
                    rnn_hidden_states,
                    masks[start:end].flatten(end_dim=1)
                )
                final_hidden = self.transformer(final_hidden).unsqueeze(0)
                rnn_hidden_states = torch.cat((other_hidden, final_hidden), dim=0)
                beliefs.append(embeddings) # appending n x k x h
            beliefs = torch.cat(beliefs, 0) # hopefully this is reshaped properly
        contextual_embedding, weights = self._fuse_beliefs(beliefs, x)
        return contextual_embedding, rnn_hidden_states, beliefs, weights

class TransformerBeliefPolicy(MultipleBeliefPolicy):
    r"""
        Naive implementation of self-attention
        Separate modules self-attend for k layers, and fuse linearly
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=TransformerBelief,
            **kwargs
        )

class HiddenTransformerBeliefPolicy(MultipleBeliefPolicy):
    r"""
        Naive implementation of self-attention
        Separate modules self-attend for k layers, and fuse linearly
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=HiddenTransformerBelief,
            **kwargs
        )

class TransformerIMBeliefPolicy(MultipleBeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=TransformerIMBelief,
            **kwargs
        )