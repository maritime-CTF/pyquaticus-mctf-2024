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
import ray
from ray.rllib.algorithms.ppo import PPOConfig
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
from ray.rllib.policy.policy import PolicySpec, Policy
import os
import pyquaticus.utils.rewards as rew
from pyquaticus.base_policies.base_policies import DefendGen, AttackGen
from pyquaticus.utils.dummy_policies import NoOpPolicy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 3v3 policy in a 3v3 PyQuaticus environment')
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    reward_config = {0:rew.custom_v1, 1:rew.custom_v1, 2:rew.custom_v1, 3:None, 4:None, 5:None} # Example Reward Config
    #Competitors: reward_config should be updated to reflect how you want to reward your learning agent
    
    args = parser.parse_args()

    RENDER_MODE = 'human' if args.render else None #set to 'human' if you want rendered output
    
    env_creator = lambda config: pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=3)
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(render_mode=RENDER_MODE, reward_config=reward_config, team_size=3))
    register_env('pyquaticus', lambda config: ParallelPettingZooWrapper(env_creator(config)))
    obs_space = env.observation_space
    act_space = env.action_space
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0 or agent_id == 'agent-0':
            return "agent-0-policy"
        if agent_id == 1 or agent_id == 'agent-1':
            return "agent-1-policy"
        elif agent_id == 2 or agent_id == 'agent-2':
            # change this to agent-1-policy to train both agents at once
            return "agent-2-policy"
        elif agent_id == 3 or agent_id == 'agent-3':
            return "noop-policy-3"
        elif agent_id == 4 or agent_id == 'agent-4':
            return "noop-policy-4"
        else:
            return "noop-policy-5"
    
    policies = {'agent-0-policy':(None, obs_space, act_space, {}), 
                'agent-1-policy':(None, obs_space, act_space, {}),
                'agent-2-policy':(None, obs_space, act_space, {}),
                'noop-policy-3':(AttackGen(3, Team.RED_TEAM, 'nothing', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                'noop-policy-4':(AttackGen(4, Team.RED_TEAM, 'nothing', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                'noop-policy-5':(AttackGen(5, Team.RED_TEAM, 'nothing', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                'easy-defend-policy': (DefendGen(4, Team.RED_TEAM, 'competition_easy', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {}),
                'easy-attack-policy': (AttackGen(3, Team.RED_TEAM, 'competition_easy', 3, env.par_env.agent_obs_normalizer), obs_space, act_space, {})}
    env.close()
    ppo_config = PPOConfig().environment(env='pyquaticus').rollouts(num_rollout_workers=1).resources(num_cpus_per_worker=1, num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    #If your system allows changing the number of rollouts can significantly reduce training times (num_rollout_workers=15)
    ppo_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent-0-policy", "agent-1-policy", "agent-2-policy"],)
    algo = ppo_config.build()

    for i in range(6):
        algo.train()
        if np.mod(i, 5) == 0:
            print("Saving Checkpoint: ", i)
            chkpt_file = algo.save('./ray_test/')
            print(f'Saved to {chkpt_file}', flush=True)
