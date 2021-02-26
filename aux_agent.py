#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random
import os
import contextlib

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

AS_DETERMINISTIC_AS_POSSIBLE = False

class AuxAgent(Agent):
    def __init__(self, config: Config):
        if not config.MODEL_PATH:
            raise Exception(
                "Model checkpoint wasn't provided, quitting."
            )
        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        # if AS_DETERMINISTIC_AS_POSSIBLE: # This times out remote evalai
            # self.device = torch.device("cpu")
        ckpt_dict = torch.load(config.MODEL_PATH, map_location=self.device)

        # Config
        self.config = config
        if config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()
        aux_cfg = config.RL.AUX_TASKS
        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG.TASK

        self._fp16_autocast = self.config.RL.fp16_mode == "autocast"

        self._fp16_autocast = False # TEST THIS

        # ! Agent setup
        policy_encoders = get_vision_encoder_inputs(ppo_cfg)

        # Load spaces (manually)
        spaces = {
            "objectgoal": Box(
                low=0, high=20, # from matterport dataset
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
                shape=(3,), # Spoof for model to be shaped correctly
                dtype=np.float32,
            ),
            "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)
        }

        observation_spaces = Dict(spaces)
        if 'actor_critic.action_distribution.linear.bias' in ckpt_dict['state_dict']:
            num_acts = ckpt_dict['state_dict']['actor_critic.action_distribution.linear.bias'].size(0)
        else: # multipolicy
            num_acts = ckpt_dict['state_dict']['actor_critic.action_distribution.stack.1.linear.bias'].size(0)
        action_spaces = Discrete(num_acts)

        self.obs_transforms = get_active_obs_transforms(config)
        observation_spaces = apply_obs_transforms_obs_space(
            observation_spaces, self.obs_transforms
        )

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
        self.actor_critic = policy_class(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=ppo_cfg.hidden_size,
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

        self.num_recurrent_memories = self.actor_critic.net.num_tasks
        if self.actor_critic.IS_MULTIPLE_BELIEF:
            proposed_num_beliefs = ppo_cfg.POLICY.BELIEFS.NUM_BELIEFS
            self.num_recurrent_memories = len(aux_cfg.tasks) if proposed_num_beliefs == -1 else proposed_num_beliefs
            if self.actor_critic.IS_RECURRENT:
                self.num_recurrent_memories += 1

        self.actor_critic.load_state_dict(
            {
                k.replace("actor_critic.", ""): v
                for k, v in ckpt_dict["state_dict"].items()
                if "actor_critic" in k
            }
        )
        self.actor_critic.eval()

        self.semantic_predictor = None
        if ppo_cfg.POLICY.USE_SEMANTICS:
            self.semantic_predictor = load_rednet(
                self.device,
                ckpt="ckpts/rednet.pth", # ppo_cfg.POLICY.EVAL_SEMANTICS_CKPT
                resize=True # always to half size
            )
            self.semantic_predictor.eval()

        self.behavioral_index = 0
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            count_steps = ckpt_dict["extra_state"]["step"]
            if _get_policy_head_count(config) > 1 and count_steps > config.RL.REWARD_FUSION.SPLIT.TRANSITION:
                self.behavioral_index = 1

        # Load other items
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1, # num_processes
            self.num_recurrent_memories,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        self.not_done_masks = None
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

        # self.step = 0
        # self.ep = 0
        # self._POSSIBLE_ACTIONS = task_cfg.POSSIBLE_ACTIONS
        # self.actions = []

    def reset(self):
        # print(f'{self.ep} reset {self.step}')
        # We don't reset state because our rnn accounts for masks, and ignore actions because we don't use actions
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)

        # self.step = 0
        # self.ep += 1
        # self.actions = []

    @torch.no_grad()
    def act(self, observations):
        # RANDOM TEST
        # return {"action": np.random.choice(self._POSSIBLE_ACTIONS)} # Parity - baseline
        # return torch.randint(len(self._POSSIBLE_ACTIONS), (1,)).item() # Parity - checking whether it's torch or syntax

        batch = batch_obs([observations], device=self.device) # Why is this put in a list?
        if self.semantic_predictor is not None:
            batch["semantic"] = self.semantic_predictor(batch["rgb"], batch["depth"])
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        with torch.no_grad(), torch.cuda.amp.autocast() if self._fp16_autocast else contextlib.suppress():
            # Substitute 3D GPS (2D provided noted in `nav.py`)
            if batch['gps'].size(-1) == 2:
                batch["gps"] = torch.stack([
                    batch["gps"][:, 1],
                    torch.zeros(batch["gps"].size(0), dtype=batch["gps"].dtype, device=self.device),
                    -batch["gps"][:, 0],
                ], axis=-1)

            # Looks like `obs` was always different inside.
            # if self.step == 24 and self.ep == 2: # Crash it - check if obs or recurrent
            # if self.step == 283 and self.ep == 29: # Crash it - check what's wrong
            # if self.step == 283 and self.ep == 29: # Crash it - check what's wrong
                # print('inputs')
                # print({
                #     key: vec.sum().item() for key, vec in batch.items()
                # })
                # print('state pre step')
                # print({
                #     'hidden_mean': self.test_recurrent_hidden_states.mean(-1).cpu().tolist(),
                #     'hidden_std': self.test_recurrent_hidden_states.std(-1).cpu().tolist(),
                # })
            #     value, actions, _, self.test_recurrent_hidden_states, features, obs, _ = self.actor_critic.act(
            #         batch,
            #         self.test_recurrent_hidden_states,
            #         self.prev_actions,
            #         self.not_done_masks,
            #         deterministic=AS_DETERMINISTIC_AS_POSSIBLE,
            #         behavioral_index=self.behavioral_index,
            #         return_all_activations=True,
            #         debug=True
            #     )
            #     # Obs is different
            #     raise Exception({
            #         'features': features.mean().cpu().item(),
            #         'features_std': features.std().cpu().item(),
            #         'obs': obs.mean().cpu().item(),
            #         'obs_std': obs.std().cpu().item(),
            #         'hidden_mean': self.test_recurrent_hidden_states.mean(-1).cpu().tolist(),
            #         'hidden_std': self.test_recurrent_hidden_states.std(-1).cpu().tolist(),
            #         'actions': actions
            #     })
            # else:
            _, actions, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=AS_DETERMINISTIC_AS_POSSIBLE,
                behavioral_index=self.behavioral_index,
            )
            self.prev_actions.copy_(actions)

        self.not_done_masks = torch.ones(1, 1, device=self.device, dtype=torch.bool) # Reset called externally, we're not done until then

        # return torch.randint(len(self._POSSIBLE_ACTIONS), (1,)).item() # Parity - checking whether it's random state being affected by my agent

        # self.step += 1
        # if self.step > 30: # * PARITY (1 success) TODO check this again, I'm sus.
        # if self.step > 31: # ! NO PARITY
        # if self.step > 31 or self.ep > 15: # ! NO PARITY (whoops)
        # if self.step > 31 or self.ep > 8: # ! NO PARITY
        # if self.step > 31 or self.ep > 2: # ! NO PARITY
        # if self.step > 31 or self.ep > 1: # * PARITY
        # * So, if we restrict to first episode (which is 31 steps), we have parity. 31 steps of ep 2 is too much, though.
        # if (self.step > 31 or self.ep > 1) or (self.step > 50 and self.ep == 1): # * PARITY (restricted)
        # if (self.step > 31 or self.ep > 3) or (self.step > 4 and self.ep == 2): # * PARITY
        # if (self.step > 31 or self.ep > 3) or (self.step > 18 and self.ep == 2): # * PARITY feck you have a confound, ep down to 2 please
        # if (self.step > 31 or self.ep >= 3) or (self.step > 24 and self.ep == 2): # * PARITY
        # if (self.step > 31 or self.ep >= 3) or (self.step > 28 and self.ep == 2): # ! NO PARITY
        # if (self.step > 31 or self.ep >= 3) or (self.step > 26 and self.ep == 2): # ! NO PARITY
        # if (self.step > 31 or self.ep >= 3) or (self.step > 25 and self.ep == 2): # ! NO PARITY
        # In conclusion, self.step == 25 of self.ep == 2 diverges. Why? (Let's look at the hidden states!)

        # if self.step > 31:
        #     return torch.randint(len(self._POSSIBLE_ACTIONS), (1,)).item()
        # print(self.ep, len(self.actions[-1]))
        # self.actions.append(actions[0][0].item())
        # if self.step == 25 and self.ep == 2:
        # if self.actions[-1] == 0 or self.step == 500:
        #     print(len(self.actions), self.actions)
        #     if self.ep == 29:

        #         raise Exception('done')
        #     # Now, look at the internal states and observations.
        #     payload = {}
        #     # for key in batch:
        #     #     payload[key] = batch[key].float().std().item()
        #     # payload['hidden_std'] = self.test_recurrent_hidden_states.std().item()
        #     payload['hidden_mean'] = self.test_recurrent_hidden_states.mean().item() # Hidden mean is different, std is the same. Inputs are the same.
        #     # Now trace backwards. where in?
        #     payload['actions'] = self.actions # Yep. Action 3 locally, instead of action 1 remotely.
            # raise Exception(payload)

        return actions[0][0].item()

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        # From base trainer
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()
        config.defrost()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config, )
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.defrost()
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

def main():
    # ! Note, there's some additional config not ported from dev setup, but those choices shouldn't matter...
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
    config.RL.PPO.POLICY.EVAL_GT_SEMANTICS = False

    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    config.RANDOM_SEED = 7
    config.freeze()
    torch.set_deterministic(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    agent = AuxAgent(config)
    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()