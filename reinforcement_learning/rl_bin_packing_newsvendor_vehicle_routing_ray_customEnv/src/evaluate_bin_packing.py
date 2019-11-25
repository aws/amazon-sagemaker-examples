#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import numpy as np

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from gym.envs.registration import register
import bin_packing_env
from ray.tune.util import merge_dicts
import csv

from action_mask_model import ActionMaskModel, register_actor_mask_model

register_actor_mask_model()

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

def create_environment(env_config):
    # This import must happen inside the method so that worker processes import this code
    import bin_packing_env
    
    env_config = {
                   "bag_capacity": 100,
                   'item_sizes': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                   'item_probabilities': [0, 0, 0, 1 / 3, 0, 0, 0, 0, 2 / 3],  # linear waste
                    # 'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
                    # 'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33], #perfect pack
                   'time_horizon': 10000,
               }
    
    register(
            id='BinPacking-v1',
#             entry_point='bin_packing_env:BinPackingEnv',
#             entry_point='bin_packing_env:BinPackingIncrementalWasteEnv',
            entry_point='bin_packing_env:BinPackingActionMaskEnv',        
            kwargs = {'env_config': env_config}
        )
    return gym.make('BinPacking-v1')

def run():
    out = None #picke file to store output results
    render = False
    config = None

    run = "PPO"
    checkpoint = "models/perfect_pack_bin100/checkpoint_100/checkpoint-100"

    
    register_env("BinPacking-v1", create_environment)
    env = "BinPacking-v1"
    steps = 1000
    if not config:
        # Load configuration from file
        config_dir = os.path.dirname(checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.json")
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.json in either the checkpoint dir or "
                "its parent directory.")
        with open(config_path) as f:
            config = json.load(f)

    if not env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        env = config.get("env")

    config["num_workers"] = 1

    ray.init()

    cls = get_agent_class(run)
    agent = cls(env=env, config=config)
    agent.restore(checkpoint)
    num_steps = int(steps)

    env = gym.make(env)
    multiagent = False
    use_lstm = {'default': False}
    
    record_csv = False
    if record_csv:
        csv_file = 'bin_levels_rl_linear_waste.csv'
        header = ["Level " + str(x) for x in range(0,env.bag_capacity)]
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    total_rewards = []
    count = 0
    while count < 100:
        count += 1
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done:
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
#             print(state)
#             print(action)
#             print()
            reward_total += reward
            state = next_state
            
#             bin_levels = state[:-1]
            if record_csv:
                with open(csv_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(bin_levels)
#         print("Episode reward", reward_total)
#         print("Info: ", _)
        total_rewards.append(reward_total)

    print("Total Rewards, Mean: ", np.mean(total_rewards), "Min: ", np.min(total_rewards), "Max: ", np.max(total_rewards), "Std Dev: ", np.std(total_rewards))

if __name__ == "__main__":
    run()
