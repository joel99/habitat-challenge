#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random
import os

import numpy as np
import torch
import PIL
from gym.spaces import Discrete, Dict, Box

import habitat
from src.default import get_config
from src import POLICY_CLASSES
from habitat_baselines.common.utils import batch_obs
from habitat import Config
from habitat.core.agent import Agent

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
                shape=(2,),
                dtype=np.float32,
            ),
            "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)
        }

        observation_spaces = Dict(spaces)

        action_spaces = Discrete(4) # ! change to six if we're running action v1

        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        self.hidden_size = config.RL.PPO.hidden_size

        self.aux_cfg = config.RL.AUX_TASKS
        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG.TASK

        is_objectnav = "ObjectNav" in task_cfg.TYPE
        additional_sensors = []
        embed_goal = False
        if is_objectnav:
            additional_sensors = ["gps", "compass"]
            embed_goal = True

        random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

        policy_class = POLICY_CLASSES[ppo_cfg.policy]

        self.actor_critic = policy_class(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            aux_tasks=self.aux_cfg.tasks,
            goal_sensor_uuid=task_cfg.GOAL_SENSOR_UUID,
            num_tasks=len(self.aux_cfg.tasks), # we pass this is in to support eval, where no aux modules are made
            additional_sensors=additional_sensors,
            embed_goal=embed_goal,
            device=self.device,
            config=ppo_cfg.POLICY
        ).to(self.device)

        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            # ! TODO - do our checkpoints start with actor critic?
            self.actor_critic.load_state_dict(
                {
                    k.replace("actor_critic.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None

    def reset(self):
        num_recurrent_memories = self.actor_critic.net.num_tasks
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1, # num_processes
            num_recurrent_memories,
            self.hidden_size,
            device=self.device,
        )

        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

    def act(self, observations):
        batch = batch_obs([observations])
        for sensor in batch:
            batch[sensor] = batch[sensor].to(self.device)

        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True, # False
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks = torch.ones(1, 1, device=self.device)
            self.prev_actions.copy_(actions)
        return actions[0][0].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--config-path", type=str, required=True, default="configs/aux_objectnav.yaml")
    args = parser.parse_args()

    config = get_config(args.config_path,
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