import argparse
import sys
import os
import os.path
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.envs.pyquaticus import Team
from collections import OrderedDict
from pyquaticus.config import config_dict_std, ACTION_MAP
import copy
#Update this to the path of your solution.py file or ensure its on the same level as this (competition_test.py) script
from solution import solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test against the evaluation used on the submission platform (3v3)')
    parser.add_argument('--render', help='Enable rendering', action='store_true')
    args = parser.parse_args()
    RENDER_MODE = 'human' if args.render else None #set to 'human' if you want rendered output

    config_dict = config_dict_std
    config_dict["max_time"] = 600.0
    config_dict["max_score"] = 100


    easy_score = 0
    medium_score = 0
    hidden_score = 0

    step = 0
    #RED side Competition Easy Defender and Attacker vs Submission (Blue Side)
    env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict,render_mode=None)
    term_g = {0:False,1:False}
    truncated_g = {0:False,1:False}
    term = term_g
    trunc = truncated_g
    obs = env.reset()
    temp_score = copy.deepcopy(env.game_score)
    sol = solution()
    H_one = BaseDefender(3, Team.RED_TEAM, mode='competition_easy')
    H_two = BaseAttacker(4, Team.RED_TEAM, mode='competition_easy')
    H_three = BaseAttacker(5, Team.RED_TEAM, mode='competition_easy')
    while True:
        new_obs = {}
        #Get normalized observation (for heuristic approaches)
        for k in obs:
            new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])
        try:
            zero = sol.compute_action(0,obs[0], new_obs[0], new_obs)
            one = sol.compute_action(1,obs[1],new_obs[1], new_obs)
            two = sol.compute_action(2,obs[2],new_obs[2], new_obs)
        except:
            zero = sol.compute_action(0,obs[0], new_obs[0])
            one = sol.compute_action(1,obs[1],new_obs[1])
            two = sol.compute_action(2,obs[2],new_obs[2])
        three = H_one.compute_action(new_obs)
        four = H_two.compute_action(new_obs)
        five = H_three.compute_action(new_obs)
        
        obs, reward, term, trunc, info = env.step({0:zero,1:one, 2:two, 3:three, 4:four, 5:five})
        k =  list(term.keys())
        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            break
    for k in env.game_score:
        temp_score[k] += env.game_score[k]
    env.close()
    easy_score = temp_score['blue_captures'] - temp_score['red_captures'] - (temp_score['blue_collisions']/4)
    print("Final Easy Score: ", easy_score)

    step = 0
    #RED side Competition Medium Defender and Attacker vs Submission (Blue Side)
    env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict,render_mode=RENDER_MODE)
    term_g = {0:False,1:False}
    truncated_g = {0:False,1:False}
    term = term_g
    trunc = truncated_g
    obs = env.reset()
    temp_score = copy.deepcopy(env.game_score)
    sol = solution()
    H_one = BaseDefender(3, Team.RED_TEAM, mode='competition_medium')
    H_two = BaseAttacker(4, Team.RED_TEAM, mode='competition_medium')
    H_three = BaseAttacker(5, Team.RED_TEAM, mode='competition_medium')
    while True:
        new_obs = {}
        #Get normalized observation (for heuristic approaches)
        for k in obs:
            new_obs[k] = env.agent_obs_normalizer.unnormalized(obs[k])
        try:
            zero = sol.compute_action(0,obs[0], new_obs[0], new_obs)
            one = sol.compute_action(1,obs[1],new_obs[1], new_obs)
            two = sol.compute_action(2,obs[2],new_obs[2], new_obs)
        except:
            zero = sol.compute_action(0,obs[0], new_obs[0])
            one = sol.compute_action(1,obs[1],new_obs[1])
            two = sol.compute_action(2,obs[2],new_obs[2])
        three = H_one.compute_action(new_obs)
        four = H_two.compute_action(new_obs)
        five = H_three.compute_action(new_obs)
        
        obs, reward, term, trunc, info = env.step({0:zero,1:one, 2:two, 3:three, 4:four, 5:five})
        k =  list(term.keys())
        step += 1
        if term[k[0]] == True or trunc[k[0]]==True:
            break
    for k in env.game_score:
        temp_score[k] += env.game_score[k]
    env.close()
    print("Medium Detailed Results: ", temp_score)
    medium_score += temp_score['blue_captures'] - temp_score['red_captures'] - (temp_score['blue_collisions']/4)
    print("Final Medium Score: ", medium_score)
