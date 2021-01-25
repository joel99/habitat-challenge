#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict
from .policy import Policy
from .belief_policy import (
    BeliefPolicy, AttentiveBeliefPolicy,
    FixedAttentionBeliefPolicy, AverageBeliefPolicy, SoftmaxBeliefPolicy,
    RecurrentAttentiveBeliefPolicy,
)
from .self_attentive_policy import (
    TransformerIMBeliefPolicy
    # TransformerBeliefPolicy, TransformerIMBeliefPolicy, TransformerGatedBeliefPolicy,
    # HiddenTransformerBeliefPolicy, TransformerIMBeliefPolicyNoReLU
)

# from habitat_baselines.rl.ppo.hierarchical_policy import (
#     HierarchicalPolicy,
# )

# from habitat_baselines.rl.ppo.ppo import PPO

SINGLE_BELIEF_CLASSES: Dict[str, Policy] = {
    # "BASELINE": BaselinePolicy,
    "SINGLE_BELIEF": BeliefPolicy,
}

MULTIPLE_BELIEF_CLASSES = {
    "ATTENTIVE_BELIEF": AttentiveBeliefPolicy,
    "FIXED_ATTENTION_BELIEF": FixedAttentionBeliefPolicy,
    "AVERAGE_BELIEF": AverageBeliefPolicy,
    "SOFTMAX_BELIEF": SoftmaxBeliefPolicy,
    "RECURRENT_ATTENTIVE_BELIEF": RecurrentAttentiveBeliefPolicy,
    # "TRANSFORMER_BELIEF": TransformerBeliefPolicy,
    # "HIDDEN_TRANSFORMER_BELIEF": HiddenTransformerBeliefPolicy,
    "TRANSFORMER_IM_BELIEF": TransformerIMBeliefPolicy,
    # "TRANSFORMER_IM_BELIEF_NO_RELU": TransformerIMBeliefPolicyNoReLU,
    # "TRANSFORMER_GATED_BELIEF": TransformerGatedBeliefPolicy,
    # "HIERARCHICAL": HierarchicalPolicy
}

def is_recurrent_attention_policy(policy):
    return "RECURRENT" in policy

POLICY_CLASSES = dict(SINGLE_BELIEF_CLASSES, **MULTIPLE_BELIEF_CLASSES)

__all__ = [
    "Policy", "POLICY_CLASSES", "SINGLE_BELIEF_CLASSES", "MULTIPLE_BELIEF_CLASSES", # "PPO"
]
