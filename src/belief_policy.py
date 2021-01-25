#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from habitat_baselines.common.utils import Flatten
from .models import resnet, RNNStateEncoder
from .policy import Policy, GOAL_EMBEDDING_SIZE

class SingleBelief(nn.Module):
    r"""
        Stripped down single recurrent belief.
        Compared to the baseline, the visual encoder has been removed.
    """

    def __init__(
        self,
        observation_space,
        hidden_size,
        goal_sensor_uuid=None,
        additional_sensors=[], # low dim sensors to merge in input
        embed_goal=False,
        device=None,
        **kwargs,
    ):
        super().__init__()

        self.goal_sensor_uuid = goal_sensor_uuid
        self.additional_sensors = additional_sensors
        self.embed_goal = embed_goal
        self.device = device
        self._n_input_goal = 0
        if goal_sensor_uuid is not None and goal_sensor_uuid != "no_sensor":
            self.goal_sensor_uuid = goal_sensor_uuid
            self._initialize_goal_encoder(observation_space)

        self._hidden_size = hidden_size
        embedding_size = (0 if self.is_blind else self._hidden_size) + self._n_input_goal
        for sensor in self.additional_sensors:
            embedding_size += observation_space.spaces[sensor].shape[0]
        self._embedding_size = embedding_size

        self._initialize_state_encoder()

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _initialize_goal_encoder(self, observation_space):

        if not self.embed_goal:
            self._n_input_goal = observation_space.spaces[
                self.goal_sensor_uuid
            ].shape[0]
            return

        self._n_input_goal = GOAL_EMBEDDING_SIZE
        goal_space = observation_space.spaces[
            self.goal_sensor_uuid
        ]
        self.goal_embedder = nn.Embedding(goal_space.high[0] - goal_space.low[0] + 1, self._n_input_goal)

    def _initialize_state_encoder(self):
        self.state_encoder = RNNStateEncoder(self._embedding_size, self._hidden_size)

    def get_target_encoding(self, observations):
        goal = observations[self.goal_sensor_uuid]
        if self.embed_goal:
            return self.goal_embedder(goal.long()).squeeze(-2)
        return goal

    def _get_observation_embedding(self, visual_embedding, observations):
        embedding = [visual_embedding]
        if self.goal_sensor_uuid is not None:
            embedding.append(self.get_target_encoding(observations))
        for sensor in self.additional_sensors:
            embedding.append(observations[sensor])
        return torch.cat(embedding, dim=-1)

    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        x = self._get_observation_embedding(visual_embedding, observations)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states

class BeliefPolicy(Policy):
    r"""
        Base class for policy that will interact with auxiliary tasks.
        Provides a visual encoder, requires a recurrent net.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        net=SingleBelief,
        aux_tasks=[], # bruh are we even forwarding these things...
        config=None,
        **kwargs, # Note, we forward kwargs to the net
    ):
        assert issubclass(net, SingleBelief), "Belief policy must use belief net"
        super().__init__(net(
            observation_space=observation_space,
            hidden_size=hidden_size,
            config=config, # Forward
            **kwargs,
        ), action_space.n)
        self.aux_tasks = aux_tasks

        resnet_baseplanes = 32
        backbone="resnet18"

        visual_resnet = resnet.ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=config.use_mean_and_var
        )

        self.visual_encoder = nn.Sequential(
            visual_resnet,
            Flatten(),
            nn.Linear(
                np.prod(visual_resnet.output_shape), hidden_size
            ),
            nn.ReLU(True),
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs
    ):
        visual_embedding = self.visual_encoder(observations)
        features, rnn_hidden_states = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        visual_embedding = self.visual_encoder(observations)
        features, *_ = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        visual_embedding = self.visual_encoder(observations)
        features, rnn_hidden_states = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()
        # Nones: individual_features, aux entropy, aux weights
        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features, None, None, None

    def shape_aux_inputs(self, sample, final_rnn_state):
        observations = sample[0]
        n = final_rnn_state.size(1)
        masks = sample[6].view(-1, n)
        env_zeros = [] # Episode crossings per env, lots of tasks use this
        for env in range(n):
            env_zeros.append(
                (masks[:, env] == 0.0).nonzero().squeeze(-1).cpu().unbind(0)
            )
        t = masks.size(0)
        actions = sample[2].view(t, n)
        vision_embedding = self.visual_encoder(observations).view(t, n, -1)

        return observations, actions, vision_embedding, n, t, env_zeros

    def evaluate_aux_losses(self, sample, final_rnn_state, rnn_features, *args):
        if len(self.aux_tasks) == 0:
            pass
        observations, actions, vision_embedding, n, t, env_zeros = self.shape_aux_inputs(sample, final_rnn_state)
        belief_features = rnn_features.view(t, n, -1)
        final_belief_state = final_rnn_state[-1] # only use final layer
        # TODO - we want to put in the same semantic vision to avoid redundant calc
        # TODO cuda stream here
        return [task.get_loss(observations, actions, vision_embedding, final_belief_state, belief_features, n, t, env_zeros) for task in self.aux_tasks]

class MultipleBeliefNet(SingleBelief):
    r"""
        Uses multiple belief RNNs. Requires num_tasks, and fusion workings.
    """
    def __init__(
        self,
        observation_space,
        hidden_size,
        num_tasks,
        **kwargs,
    ):
        self.num_tasks = num_tasks # We declare this first so state encoders can be initialized

        super().__init__(observation_space, hidden_size, **kwargs)
        self._initialize_fusion_net()
        self.cuda_streams = None

    @property
    def num_recurrent_layers(self):
        return self.state_encoders[0].num_recurrent_layers

    def _initialize_state_encoder(self):
        self.state_encoders = nn.ModuleList([
            RNNStateEncoder(self._embedding_size, self._hidden_size) for _ in range(self.num_tasks)
        ])

    def set_streams(self, streams):
        self.cuda_streams = streams

    def _initialize_fusion_net(self):
        pass # Do nothing as a default

    @abc.abstractmethod
    def _fuse_beliefs(self, beliefs, x, *args):
        pass

    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        x = self._get_observation_embedding(visual_embedding, observations)
        # rnn_hidden_states.size(): num_layers, num_envs, num_tasks, hidden, (only first timestep)
        if self.cuda_streams is None:
            outputs = [encoder(x, rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
        else:
            outputs = [None] * self.num_tasks
            torch.cuda.synchronize()
            for i, encoder in enumerate(self.state_encoders):
                with torch.cuda.stream(self.cuda_streams[i]):
                    outputs[i] = encoder(x, rnn_hidden_states[:, :, i], masks)
            torch.cuda.synchronize()
        embeddings, rnn_hidden_states = zip(*outputs) # (txn)xh, (layers)xnxh
        rnn_hidden_states = torch.stack(rnn_hidden_states, dim=-2) # (layers) x n x k x h
        beliefs = torch.stack(embeddings, dim=-2) # (t x n) x k x h

        contextual_embedding, weights = self._fuse_beliefs(beliefs, x)
        return contextual_embedding, rnn_hidden_states, beliefs, weights

class MultipleBeliefPolicy(BeliefPolicy):
    r""" Base policy for multiple beliefs adding basic checks and weight diagnostics
    """
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        net=None,
        aux_tasks=[],
        num_tasks=0,
        **kwargs,
    ):
        # 0 tasks allowed for eval
        assert len(aux_tasks) != 1, "Multiple beliefs requires more than one auxiliary task"
        assert issubclass(net, MultipleBeliefNet), "Multiple belief policy requires compatible multiple belief net"

        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net,
            num_tasks=num_tasks,
            aux_tasks=aux_tasks,
            **kwargs,
        )
        # I think these slow things down atm
        # self.cuda_streams = [torch.cuda.Stream() for i in range(num_tasks)]
        # self.net.set_streams(self.cuda_streams)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        weights_output=None,
    ):
        visual_embedding = self.visual_encoder(observations)
        features, rnn_hidden_states, _, weights = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        if weights_output is not None:
            weights_output.copy_(weights)
        return value, action, action_log_probs, rnn_hidden_states


    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        visual_embedding = self.visual_encoder(observations)
        # sequence forwarding
        features, rnn_hidden_states, individual_features, weights = self.net(
            visual_embedding, observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        aux_dist_entropy = Categorical(weights).entropy().mean()
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        weights = weights.mean(dim=0)

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, features, individual_features, aux_dist_entropy, weights

    def evaluate_aux_losses(self, sample, final_rnn_state, rnn_features, individual_rnn_features):
        observations, actions, vision_embedding, n, t, env_zeros = self.shape_aux_inputs(sample, final_rnn_state)
        losses = [None] * len(self.aux_tasks)
        torch.cuda.synchronize()
        for i, task in enumerate(self.aux_tasks):
            losses[i] = \
                task.get_loss(
                    observations,
                    actions,
                    vision_embedding,
                    final_rnn_state[-1, :, i].contiguous(),
                    individual_rnn_features[:, i].contiguous().view(t,n,-1),
                    n, t, env_zeros
                )
        return losses

class AttentiveBelief(MultipleBeliefNet):
    def _initialize_fusion_net(self):
        # self.visual_key_net = nn.Linear(
        self.key_net = nn.Linear(
            self._embedding_size, self._hidden_size
        )
        self.scale = math.sqrt(self._hidden_size)

    def _fuse_beliefs(self, beliefs, x, *args):
        key = self.key_net(x.unsqueeze(-2))
        # key = self.visual_key_net(x.unsqueeze(-2)) # (t x n) x 1 x h
        scores = torch.bmm(beliefs, key.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=1).squeeze(-1) # n x k (logits) x 1 -> (txn) x k

        # n x 1 x k x n x k x h
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

class RecurrentAttentiveBelief(AttentiveBelief):
    def _initialize_fusion_net(self):
       # A proper policy RNN - outputs a key
        self.attention_rnn = RNNStateEncoder(self._embedding_size, self._hidden_size)
        # Goal - output txn x 1 x h at each timestep
    # belief: (txn) x k x h - I can't separate out the batch erk
    def _fuse_beliefs(self, beliefs, x, attention_hidden_states, masks, *args):
        key, attention_hidden_state = self.attention_rnn(x, attention_hidden_states, masks)
        key = key.unsqueeze(-2)
        scores = torch.bmm(beliefs, key.transpose(1, 2)) / math.sqrt(self.num_tasks) # scaled dot product
        weights = F.softmax(scores, dim=1).squeeze(-1) # tn x k (logits) x 1 -> (txn) x k
        # tn x 1 x k x tn x k x h
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights, attention_hidden_states

    def forward(self, visual_embedding, observations, rnn_hidden_states, prev_actions, masks):
        # We forward the individual RNNs and then forward (retains sequential forwarding ability)
        x = self._get_observation_embedding(visual_embedding, observations)
        attention_hidden_states = rnn_hidden_states[:, :, -1]
        outputs = [encoder(x, rnn_hidden_states[:, :, i], masks) for i, encoder in enumerate(self.state_encoders)]
        embeddings, rnn_hidden_states = zip(*outputs) # (txn)xh, (layers)xnxh
        beliefs = torch.stack(embeddings, dim=-2) # (t x n) x k x h
        contextual_embedding, weights, attention_hidden_states = self._fuse_beliefs(beliefs, x, attention_hidden_states, masks)
        rnn_hidden_states = (*rnn_hidden_states, attention_hidden_states)
        rnn_hidden_states = torch.stack(rnn_hidden_states, dim=-2) # (layers) x n x k x h
        return contextual_embedding, rnn_hidden_states, beliefs, weights

class FixedAttentionBelief(MultipleBeliefNet):
    r""" Fixed Attn Baseline for comparison w/ naturally arising peaky distribution
    """
    def _fuse_beliefs(self, beliefs, x, *args):
        txn = x.size()[0]
        weights = torch.zeros(txn, self.num_tasks, dtype=torch.float32, device=beliefs.device)
        weights[:, 0] = 1.0 # all attn on first task
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

class SoftmaxBelief(MultipleBeliefNet):
    r""" Softmax Gating Baseline for comparison w/ regular attention
    """
    def _initialize_fusion_net(self):
        self.softmax_net = nn.Linear(
            self._embedding_size, self.num_tasks
        )

    def _fuse_beliefs(self, beliefs, x, *args):
        scores = self.softmax_net(x) # (t x n) x k
        weights = F.softmax(scores, dim=-1).squeeze(-1) # (txn) x k
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

class AverageBelief(MultipleBeliefNet):
    def _fuse_beliefs(self, beliefs, x, *args):
        txn = x.size(0)
        weights = torch.ones(txn, self.num_tasks, dtype=torch.float32, device=beliefs.device)
        weights /= self.num_tasks
        contextual_embedding = torch.bmm(weights.unsqueeze(1), beliefs).squeeze(1) # txn x h
        return contextual_embedding, weights

# A bunch of wrapper classes attaching a belief net to the multiple belief policy
class AttentiveBeliefPolicy(MultipleBeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        net=AttentiveBelief,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=net,
            **kwargs
        )

class FixedAttentionBeliefPolicy(MultipleBeliefPolicy):
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
            net=FixedAttentionBelief,
            **kwargs
        )

class SoftmaxBeliefPolicy(MultipleBeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=SoftmaxBelief,
            **kwargs
        )

class AverageBeliefPolicy(MultipleBeliefPolicy):
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
            net=AverageBelief,
            **kwargs
        )

class RecurrentAttentiveBeliefPolicy(AttentiveBeliefPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_size,
            net=RecurrentAttentiveBelief,
            **kwargs
        )