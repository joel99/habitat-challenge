#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from .belief_policy import (
    MultipleBeliefPolicy,
    AttentiveBeliefCore,
)

from .comm_baselines import (
    PermutedEncoder
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class MessageEncoder(nn.Module):
    r"""
        Create message from beliefs and merge into observations.
    """
    def __init__(
        self,
        config,
        obs_size,
        hidden_size,
        num_tasks,
    ):
        super().__init__()
        self.transformer = PermutedEncoder(config.TRANSFORMER, hidden_size, num_tasks)
        self.msg_encoder = nn.Sequential(
            nn.Linear(obs_size + hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, obs_size),
            nn.ReLU(True)
        )

    def forward(self, obs, beliefs):
        r"""
            args:
                obs: n x k x h
                beliefs: n x k x h
            returns:
                messages: n x k x h
        """
        messages = self.transformer(beliefs)
        return self.msg_encoder(torch.cat([obs, messages], dim=-1))

class IMBeliefCore(AttentiveBeliefCore):
    r"""
        IMBelief core. A messaging architecture that messages on an interval.
        Messaging adds new information to the observations each belief takes in.
        This architecture extracts message information from belief outputs using a transformer.
        Extract information is concat-ed with the original observation, and reduced to original observation size.
    """
    def __init__(self, config=None, **kwargs):
        super().__init__(
            config=config,
            **kwargs
        )
        self.message_encoder = MessageEncoder(
            config,
            self._embedding_size,
            self._hidden_size,
            self.num_tasks
        )
        self.COMM_INTERVAL = config.MESSAGING.interval
        self.COMM_RATIO = 1.0 / self.COMM_INTERVAL

    def forward(self, obs, rnn_hidden_states, masks):
        r"""
        Note, this logic is very close to the logic in `HiddenTransformerBelief`.
        Don't think we'll use it elsewhere, so no refactor
            obs: bxkxh (see CustomObservationsPolicy)
            rnn_hidden_states: lxnxkxh
            masks: bxh=1
        """

        n = rnn_hidden_states.size(1)
        t = masks.size(0) // n
        fusion_obs = obs[:, 0]

        if t == 1:
            if self.COMM_RATIO >= 1 or torch.rand((1,)) < self.COMM_RATIO:
                obs = self.message_encoder(obs, rnn_hidden_states[-1])
            beliefs, rnn_hidden_states = self._step_rnn(obs, rnn_hidden_states, masks)
        else:
            masks = masks.view(t, n, 1)
            time_obs = obs.view(t, n, obs.size(1), obs.size(2))
            beliefs = []
            for start in range(0, t, self.COMM_INTERVAL):
                end = min(t, start + self.COMM_INTERVAL)
                # Since we use a GRU, we can use the hidden state as beliefs
                masked_hidden_states = rnn_hidden_states[-1] * masks[start].view(n, 1, 1)
                message = self.message_encoder(time_obs[start], masked_hidden_states)
                updated_obs = torch.cat([message.unsqueeze(0), time_obs[start + 1:end]], dim=0)
                embeddings, rnn_hidden_states = self._step_rnn(
                    updated_obs.flatten(end_dim=1),
                    rnn_hidden_states,
                    masks[start:end].flatten(end_dim=1)
                )
                beliefs.append(embeddings) # s*n x k x h
            beliefs = torch.cat(beliefs, 0) # cats on time
        contextual_embedding, weights = self._fuse_beliefs(beliefs, fusion_obs)
        return contextual_embedding, rnn_hidden_states, beliefs, weights

# -----------------------------------------------------------------------------
# Policies
# -----------------------------------------------------------------------------

class IMPolicy(MultipleBeliefPolicy):
    r"""
        Policy which expands observations for the downstream core to give each belief separate observations.
        Mainly to keep clean code. We let policy decide encoder observation to feed since some beliefs may use diff encoders (theoretically)
        Defaults to IMBeliefCore
    """
    def __init__(self, net=IMBeliefCore, **kwargs):
        super().__init__(
            net=net,
            **kwargs,
        )
