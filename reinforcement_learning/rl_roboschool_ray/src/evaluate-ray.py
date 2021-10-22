from __future__ import absolute_import, division, print_function

import argparse
import json
import os

import gym
import numpy as np
import ray
from gym import wrappers

from ray.tune.registry import register_env

from roboschool import RoboschoolReacher

OUTPUT_DIR = "/opt/ml/output/intermediate"


def create_parser(parser_creator=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/opt/ml/input/data/model/checkpoint",
        type=str,
        help="Checkpoint from which to roll out.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.",
    )
    parser.add_argument("--env", type=str, help="The gym environment to use.")
    parser.add_argument("--evaluate_episodes", default=None, help="Number of episodes to roll out.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.",
    )
    return parser


def run(args, parser):
    def create_environment(env_config={}):
        return RoboschoolReacher()

    if not args.config:
        # Load configuration from file
        config_dir = os.path.dirname(args.checkpoint)
        # params.json is saved in the model directory during ray training by default
        config_path = os.path.join(config_dir, "params.json")
        with open(config_path) as f:
            args.config = json.load(f)

    if not args.env:
        if not args.config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = args.config.get("env")

    ray.init()

    register_env(args.env, create_environment)

    if ray.__version__ >= "0.6.5":
        from ray.rllib.agents.registry import get_agent_class
    else:
        from ray.rllib.agents.agent import get_agent_class

    cls = get_agent_class(args.algorithm)
    config = args.config
    config["monitor"] = False
    config["num_workers"] = 1
    config["num_gpus"] = 0
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_episodes = int(args.evaluate_episodes)

    env = RoboschoolReacher()
    all_rewards = []
    max_steps = 100  # set a max_steps as stopping condition as this env does not return done=True

    for episode in range(num_episodes):
        steps = 0
        state = env.reset()
        done = False
        reward_total = 0.0
        while steps < max_steps:
            action = agent.compute_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            steps += 1
            state = next_state
        all_rewards.append(reward_total)
        print("Episode reward: %s. Episode steps: %s" % (reward_total, steps))
    print("Mean Reward:", np.mean(all_rewards))
    print("Max Reward:", np.max(all_rewards))
    print("Min Reward:", np.min(all_rewards))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
