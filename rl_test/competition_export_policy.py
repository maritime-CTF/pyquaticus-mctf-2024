# (C) 2021 Massachusetts Institute of Technology.

# Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

# The software/firmware is provided to you on an As-Is basis

# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

import argparse
import gymnasium as gym
import numpy as np
import pygame
from pygame import KEYDOWN, QUIT, K_ESCAPE
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTF1Policy, PPOTorchPolicy
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
import sys
import time
from pyquaticus.envs.pyquaticus import Team
import pyquaticus
from pyquaticus import pyquaticus_v0
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTF2Policy, PPOConfig
from ray.rllib.policy.policy import PolicySpec
import os
from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy


RENDER_MODE = None
#RENDER_MODE = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 3v3 policy in a 3v3 PyQuaticus environment')
    parser.add_argument('checkpoint', help='Please enter the path to the model you would like to load in')
    reward_config = {0:None, 1:None, 2:None, 3:None}#{0:rew.sparse, 1:rew.custom_v1, 2:None, 3:None} # Example Reward Config
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    
    e_normalizer = pyquaticus_v0.PyQuaticusEnv(render_mode=None, team_size=3)

    args = parser.parse_args()


    #RENDER_MODE = 'human' if args.render else None #set to 'human' if you want rendered output
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=3)
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=3))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    obs_space = env.observation_space
    act_space = env.action_space
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0 or agent_id == 'agent-0':
            return "agent-0-policy"
        elif agent_id == 1 or agent_id == 'agent-1':
            return "agent-1-policy"
        elif agent_id == 2 or agent_id == 'agent-2':
            return "agent-2-policy"
        elif agent_id == 3 or agent_id == 'agent-3':
            # change this to agent-1-policy to train both agents at once
            return "easy-defend-policy"
        elif agent_id == 4 or agent_id == 'agent-4':
            return "easy-attack-policy"
        else:
            return "easy-attack-policy"
    
    policies = {'agent-0-policy':(None, obs_space, act_space, {}), 
                'agent-1-policy':(None, obs_space, act_space, {}),
                'agent-2-policy':(None, obs_space, act_space, {}),
                'easy-defend-policy': (DefendGen(2, Team.RED_TEAM, 'competition_easy', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                'easy-attack-policy': (AttackGen(3, Team.RED_TEAM, 'competition_easy', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                'easy-attack-policy': (AttackGen(3, Team.RED_TEAM, 'nothing', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {})}
    env.close()
    ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=3).resources(num_cpus_per_worker=1, num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    #If your system allows changing the number of rollouts can significantly reduce training times (num_rollout_workers=15)
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy", "agent-1-policy", "agent-2-policy"],)
    algo = ppo_config.build()
    algo.restore(args.checkpoint)

    #Export Each individual agent policy from the rllib checkpoint (Recomended that you use this rather than the whole rllib checkpoint)
    agent_0_policy = algo.get_policy(policy_id="agent-0-policy")
    agent_1_policy = algo.get_policy(policy_id="agent-1-policy")
    agent_2_policy = algo.get_policy(policy_id="agent-2-policy")

    agent_0_policy.export_checkpoint("./policies/agent-0-policy")
    agent_1_policy.export_checkpoint("./policies/agent-1-policy")
    agent_2_policy.export_checkpoint("./policies/agent-2-policy")