#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict
from .policy import (
    Net, Policy,
)
from .belief_policy import (
    BeliefPolicy, AttentiveBeliefPolicy,
    RecurrentAttentiveBeliefPolicy,
)
from .im_policy import (
    IMPolicy
)

POLICY_CLASSES = {
    'BeliefPolicy': BeliefPolicy,
    'AttentiveBeliefPolicy': AttentiveBeliefPolicy,
    'RecurrentAttentiveBeliefPolicy': RecurrentAttentiveBeliefPolicy,
    'IMPolicy': IMPolicy,
}

__all__ = [
    "Policy", "Net",
]
