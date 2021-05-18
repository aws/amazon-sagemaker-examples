from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import subprocess

import gym
import numpy as np
import ray
from gym import wrappers
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.registry import default_registry
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

OUTPUT_DIR = "/opt/ml/output/intermediate"


class UnityEnvWrapper(gym.Env):
    def __init__(self, env_config):
        self.worker_index = 0

        if "SM_CHANNEL_TRAIN" in os.environ:
            env_name = os.environ["SM_CHANNEL_TRAIN"] + "/" + env_config["env_name"]
            os.chmod(env_name, 0o755)
            print("Changed environment binary into executable mode.")
            # Try connecting to the Unity3D game instance.
            while True:
                try:
                    unity_env = UnityEnvironment(
                        env_name,
                        no_graphics=True,
                        worker_id=self.worker_index,
                        additional_args=["-logFile", "unity.log"],
                    )
                except UnityWorkerInUseException:
                    self.worker_index += 1
                else:
                    break
        else:
            env_name = env_config["env_name"]
            while True:
                try:
                    unity_env = default_registry[env_name].make(
                        no_graphics=True,
                        worker_id=self.worker_index,
                        additional_args=["-logFile", "unity.log"],
                    )
                except UnityWorkerInUseException:
                    self.worker_index += 1
                else:
                    break

        self.env = UnityToGymWrapper(unity_env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


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
    parser.add_argument("--env", type=str, help="The Unity environment to use.")
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

    ray.init(webui_host="127.0.0.1")

    agent_env_config = {"env_name": args.env}

    register_env("unity_env", lambda config: UnityEnvWrapper(agent_env_config))

    if ray.__version__ >= "0.6.5":
        from ray.rllib.agents.registry import get_agent_class
    else:
        from ray.rllib.agents.agent import get_agent_class

    cls = get_agent_class(args.algorithm)
    config = args.config
    config["monitor"] = False
    config["num_workers"] = 0
    config["num_gpus"] = 0
    agent = cls(env="unity_env", config=config)

    agent.restore(args.checkpoint)
    num_episodes = int(args.evaluate_episodes)

    env_config = {"env_name": args.env}

    if ray.__version__ >= "0.6.5":
        env = UnityEnvWrapper(env_config)
    else:
        from ray.rllib.agents.dqn.common.wrappers import wrap_dqn

        if args.algorithm == "DQN":
            env = UnityEnvWrapper(env_config)
            env = wrap_dqn(env, args.config.get("model", {}))
        else:
            env = ModelCatalog.get_preprocessor_as_wrapper(UnityEnvWrapper(env_config))

    env = wrappers.Monitor(env, OUTPUT_DIR, force=True, video_callable=lambda episode_id: True)
    all_rewards = []
    for episode in range(num_episodes):
        steps = 0
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done:
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
    import time

    time.sleep(10)
