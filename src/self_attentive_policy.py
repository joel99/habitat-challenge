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

class TransformerBelief(AttentiveBelief):
    r"""
        Naive self-attention. Note: we can either self-attend over outputs or hidden states.
        This one is outputs. (just a little easier) See AttentiveCommunicative for the other. We do not feed transformed into next state
        # TODO support num_layers
    """
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        num_layers=1, # Self-attn layers
        num_heads=6,
        **kwargs,
    ):
        super().__init__(observation_space, hidden_size, num_tasks, **kwargs)
        self.num_layers = num_layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=self._hidden_size, nhead=num_heads)
        transformer_norm = nn.LayerNorm(self._hidden_size)
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(transformer_layer, num_layers=1, norm=transformer_norm),
            LinearMessage(self.device, self._hidden_size)
        )

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
    """
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        config,
        **kwargs
    ):
        super().__init__(observation_space, hidden_size, num_tasks, **kwargs)
        self._init_message_path()
        self.COMM_INTERVAL = config.IM.comm_interval

    def _initialize_state_encoder(self):
        self.state_encoders = nn.ModuleList([
            RNNStateEncoder(self._embedding_size, self._hidden_size) for _ in range(self.num_tasks)
        ])

    def _step_rnn(self, obs, rnn_hidden_states, masks):
        if len(obs.size()) == 3: # is customized message
            outputs = [encoder(obs[:, i], rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
        else:
            outputs = [encoder(obs, rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
        embeddings, hidden_states = zip(*outputs) # list of embeddings, hidden states, (txn)xh, (layers)xnxh
        hidden_states = torch.stack(hidden_states, dim=-2)
        return torch.stack(embeddings, dim=-2), hidden_states

    # Message creation apparati
    def _init_message_path(self):
        raise NotImplementedError

    def _get_message(self, obs, beliefs): # Ideally we can split and restrict here
        raise NotImplementedError

    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        x = self._get_observation_embedding(visual_embedding, observations)
        # ! get the message from the previous hidden state - we can do this because we are using GRUs (i.e. hidden state = belief)
        # ! we actually want to transform beliefs but we can skip that for now
        # rnn_hidden_states.size(): num_layers, num_envs, num_tasks, hidden, (only first timestep)
        n = rnn_hidden_states.size(1)
        t = int(visual_embedding.size(0) / n)
        if t == 1:
            if torch.randint(self.COMM_INTERVAL, (1,)) < 1:
                message = self._get_message(x, rnn_hidden_states[-1]) # customized per module

                beliefs, rnn_hidden_states = self._step_rnn(message, rnn_hidden_states, masks)
            else:
                beliefs, rnn_hidden_states = self._step_rnn(x, rnn_hidden_states, masks)
        else:
            masks = masks.view(t, n, -1)
            obs = x.view(t, n, -1)
            beliefs = []
            for update in range(math.ceil(t/self.COMM_INTERVAL)):
                start = update * self.COMM_INTERVAL
                end = min(t, start + self.COMM_INTERVAL)
                obs_input = obs[start:end]
                message = self._get_message(obs_input, rnn_hidden_states[-1])
                embeddings, rnn_hidden_states = self._step_rnn(
                    message.flatten(end_dim=1),
                    rnn_hidden_states,
                    masks[start:end].flatten(end_dim=1)
                )
                beliefs.append(embeddings) # appending n x k x h
            beliefs = torch.cat(beliefs, 0) # hopefully this is reshaped properly
        contextual_embedding, weights = self._fuse_beliefs(beliefs, x)
        return contextual_embedding, rnn_hidden_states, beliefs, weights

class TransformerIMBelief(IntervalMessageBelief):
    r"""
        - each module outputs a message to the world which is going to be encoded
    """
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        num_heads=6,
        **kwargs
    ):
        self.num_heads = num_heads
        super().__init__(observation_space, hidden_size, num_tasks, **kwargs)

    def _init_message_path(self):
        transformer_layer = nn.TransformerEncoderLayer(d_model=self._hidden_size, nhead=self.num_heads)
        transformer_norm = nn.LayerNorm(self._hidden_size)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1, norm=transformer_norm)
        self.input_encoder = nn.Sequential(
            nn.Linear(self._hidden_size + self._embedding_size, self._hidden_size), # take message and observation, create a single input
            nn.ReLU(True),
            nn.Linear(self._hidden_size, self._embedding_size),
            nn.ReLU(True) # ! Changed
        )

    def _get_message(self, obs, beliefs): # Ideally we can split and restrict here
        messages = self.transformer(beliefs) # n x k x h
        # obs either has t x n x h (update) or n x h
        # t x n x h
        expanded_size = (*obs.size()[:-1], messages.size(1), obs.size(-1))
        expanded_obs = obs.unsqueeze(-2).expand(expanded_size)
        if len(obs.size()) == 3: # t x n x h
            other_obs = expanded_obs[1:]
            message_obs = self.input_encoder(torch.cat([expanded_obs[0], messages], dim=-1)).unsqueeze(0) # n x k x h
            message_obs = torch.cat([message_obs, other_obs], dim=0)
        else:
            message_obs = self.input_encoder(torch.cat([expanded_obs, messages], dim=-1)) # n x k x h

        return message_obs

class TransformerIMBeliefNoReLU(TransformerIMBelief):
    def _init_message_path(self):
        transformer_layer = nn.TransformerEncoderLayer(d_model=self._hidden_size, nhead=self.num_heads)
        transformer_norm = nn.LayerNorm(self._hidden_size)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1, norm=transformer_norm)
        self.input_encoder = nn.Sequential(
            nn.Linear(self._hidden_size + self._embedding_size, self._hidden_size), # take message and observation, create a single input
            nn.ReLU(True),
            nn.Linear(self._hidden_size, self._embedding_size)
        )

class GatedMessagePassingBelief(IntervalMessageBelief):
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        **kwargs
    ):
        super().__init__(observation_space, hidden_size, num_tasks, **kwargs)
        self._message_gate = nn.Sequential(
            nn.Linear(self._embedding_size, 1),
            nn.ReLU(True)
        )
        torch.autograd.set_detect_anomaly(True)

    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        x = self._get_observation_embedding(visual_embedding, observations)
        n = rnn_hidden_states.size(1)
        t = int(visual_embedding.size(0) / n)
        gate = self._message_gate(x) # nonzero means, communicate
        if t == 1:
            beliefs = []
            for env in range(n):
                if gate[env] != 0:
                    message = self._get_message(x[env:env+1], rnn_hidden_states[-1, env:env+1]) # customized per module
                else:
                    message = x[env:env+1]
                belief, rnn_hidden_states[:,env:env+1] = self._step_rnn(
                    message, rnn_hidden_states[:,env:env+1], masks[env:env+1]
                )
                beliefs.append(belief)
            beliefs = torch.cat(beliefs, 0)
        else:
            masks = masks.view(t, n, -1)
            obs = x.view(t, n, -1)
            gate = gate.view(t, n)
            beliefs = []
            for env in range(n):
                env_belief = []
                comm_indices = torch.nonzero(gate[:, env]).flatten().tolist() # silly me, not differentiable
                if 0 not in comm_indices:
                    comm_indices = [0] + comm_indices
                comm_indices.append(t)
                for step in range(len(comm_indices) - 1):
                    start = comm_indices[step]
                    end = comm_indices[step + 1]
                    obs_input = obs[start:end, env:env+1] # just make sure we're not changing shape
                    message = self._get_message(obs_input, rnn_hidden_states[-1, env:env+1])
                    embeddings, rnn_hidden_states[:, env:env+1] = self._step_rnn(
                        message.flatten(end_dim=1),
                        rnn_hidden_states[:, env:env+1],
                        masks[start:end, env:env+1].flatten(end_dim=1)
                    )
                    env_belief.append(embeddings)
                embeddings = torch.cat(env_belief, dim=0)
                beliefs.append(embeddings) # appending n x k x h
            beliefs = torch.cat(beliefs, 0) # hopefully this is reshaped properly
        contextual_embedding, weights = self._fuse_beliefs(beliefs, x)
        return contextual_embedding, rnn_hidden_states, beliefs, weights

class TransformerGatedBelief(TransformerIMBelief, GatedMessagePassingBelief):
    pass

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
        # TODO - how do we control communication bandwidth?
        # TODO support num_layers
    """
    COMM_INTERVAL = 16
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        num_layers=1, # Self-attn layers
        num_heads=6,
        **kwargs
    ):
        super().__init__(observation_space, hidden_size, num_tasks, **kwargs)
        self.num_layers = num_layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=self._hidden_size, nhead=num_heads)
        transformer_norm = nn.LayerNorm(self._hidden_size)
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(transformer_layer, num_layers=1, norm=transformer_norm),
            LinearMessage(self.device, self._hidden_size)
        )

    def _step_rnn(self, obs, rnn_hidden_states, masks):
        outputs = [encoder(obs, rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
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
            if torch.randint(self.COMM_INTERVAL, (1,)) < 1:
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
            for update in range(math.ceil(t/self.COMM_INTERVAL)):
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

class TransformerIMBeliefPolicyNoReLU(MultipleBeliefPolicy):
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
            net=TransformerIMBeliefNoReLU,
            **kwargs
        )

class TransformerGatedBeliefPolicy(MultipleBeliefPolicy):
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
            net=TransformerGatedBelief,
            **kwargs
        )

