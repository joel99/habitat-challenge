#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random
import os

import numpy as np
import torch
from gym.spaces import Discrete, Dict, Box

import habitat
from habitat import Config
from habitat.core.agent import Agent
from habitat_baselines.common.utils import batch_obs

from src.default import get_config
from src import POLICY_CLASSES
from src.models.rednet import load_rednet
from src.encoder_dict import (
    get_vision_encoder_inputs
)

from src.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

class AuxAgent(Agent):
    def __init__(self, config: Config):
        max_object_value = 20 # from matterport
        spaces = {
            "objectgoal": Box(
                low=0, high=max_object_value,
                shape=(1,),
                dtype=np.int64
            ),
            "depth": Box(
                low=0,
                high=1,
                shape=(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 1),
                dtype=np.float32,
            ),
            "rgb": Box(
                low=0,
                high=255,
                shape=(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH, 3),
                dtype=np.uint8,
            ),
            "gps": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(3,), # Spoofed
                # shape=(2,),
                dtype=np.float32,
            ),
            "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)
        }

        observation_spaces = Dict(spaces)

        action_spaces = Discrete(4) # ! change to six if we're running action v1

        self.obs_transforms = get_active_obs_transforms(config)
        observation_spaces = apply_obs_transforms_obs_space(
            observation_spaces, self.obs_transforms
        )


        random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        self.hidden_size = config.RL.PPO.hidden_size

        aux_cfg = config.RL.AUX_TASKS
        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG.TASK

        # Prep agent
        is_objectnav = "ObjectNav" in task_cfg.TYPE
        additional_sensors = []
        embed_goal = False
        if is_objectnav:
            additional_sensors = ["gps", "compass"]
            embed_goal = True

        def _get_policy_head_count(config):
            reward_keys = config.RL.POLICIES
            if reward_keys[0] == "none" and len(reward_keys) == 1:
                return 1
            if config.RL.REWARD_FUSION.STRATEGY == "SPLIT":
                return 2
            return 1

        policy_class = POLICY_CLASSES[ppo_cfg.POLICY.name]

        policy_encoders = get_vision_encoder_inputs(ppo_cfg)

        self.actor_critic = policy_class(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            num_tasks=len(aux_cfg.tasks), # we pass this is in to support eval, where no aux modules are made
            additional_sensors=additional_sensors,
            embed_goal=embed_goal,
            device=self.device,
            config=ppo_cfg.POLICY,
            policy_encoders=policy_encoders,
            num_policy_heads=_get_policy_head_count(config),
            mock_objectnav=config.MOCK_OBJECTNAV
        ).to(self.device)

        self.actor_critic.to(self.device)

        self.semantic_predictor = None
        if ppo_cfg.POLICY.USE_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt="ckpts/rednet.pth", # ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT
                resize=not config.RL.POLICY.FULL_VISION
            )
            self.semantic_predictor.eval()

        self.num_recurrent_memories = self.actor_critic.net.num_tasks
        if self.actor_critic.IS_MULTIPLE_BELIEF:
            proposed_num_beliefs = ppo_cfg.POLICY.BELIEFS.NUM_BELIEFS
            self.num_recurrent_memories = len(aux_cfg.tasks) if proposed_num_beliefs == -1 else proposed_num_beliefs
            if self.actor_critic.IS_RECURRENT:
                self.num_recurrent_memories += 1

        if config.MODEL_PATH:
            ckpt_dict = torch.load(config.MODEL_PATH, map_location=self.device)
            self.actor_critic.load_state_dict(
                {
                    k.replace("actor_critic.", ""): v
                    for k, v in ckpt_dict["state_dict"].items()
                    if "actor_critic" in k
                }
            )
            self.behavioral_index = 0
            if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
                count_steps = ckpt_dict["extra_state"]["step"]
                if _get_policy_head_count(config) > 1 and count_steps > config.RL.REWARD_FUSION.SPLIT.TRANSITION:
                    self.behavioral_index = 1
        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None
        self.step = 0
        self.actor_critic.eval()


    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1, # num_processes
            self.num_recurrent_memories,
            self.hidden_size,
            device=self.device,
        )

        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )
        self.step = 0

    @torch.no_grad()
    def act(self, observations):
        # if self.step > 350: # 350: # 350 ~ 24.5 hours, 400 should be ok - just kidding.
        #     return 0
        batch = batch_obs([observations], device=self.device) # Why is this put in a list?
        if self.semantic_predictor is not None:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # print(batch['rgb'].mean(), batch['rgb'].std())
        # print(batch['depth'].mean(), batch['depth'].std())
        # print(batch['semantic'].size(), batch['semantic'].sum())
        # print("done")

        # import pdb
        # pdb.set_trace()

        # Substitute 3D GPS (2D provided noted in `nav.py`)
        if batch['gps'].size(-1) == 2:
            batch["gps"] = torch.stack([
                batch["gps"][:, 1],
                torch.zeros(batch["gps"].size(0), dtype=batch["gps"].dtype, device=self.device),
                -batch["gps"][:, 0],
            ], axis=-1)

        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                # deterministic=True,
                deterministic=False,
                behavioral_index=self.behavioral_index,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(1.0)
            self.prev_actions.copy_(actions)
        self.step += 1
        return actions[0][0].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--config-path", type=str, required=True, default="configs/aux_objectnav.yaml")
    args = parser.parse_args()

    DEFAULT_CONFIG = "configs/obj_base.yaml"
    config = get_config([DEFAULT_CONFIG, args.config_path],
                ['BASE_TASK_CONFIG_PATH', config_paths]).clone()
    config.defrost()
    config.TORCH_GPU_ID = 0
    config.MODEL_PATH = args.model_path

    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config.RANDOM_SEED = 7
    config.freeze()

    agent = AuxAgent(config)
    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()