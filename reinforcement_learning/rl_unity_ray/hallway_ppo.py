from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym


import ray
from ray import tune
# from ray.tune import grid_search
from ray.tune.registry import register_env

import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper



if __name__ == "__main__":

    class UnityEnvWrapper(gym.Env):
        def __init__(self, env_config):
            self.worker_index = env_config.worker_index
            # Name of the Unity environment binary to launch
            env_name = '/opt/code/hallway_env_linux'
            unity_env = UnityEnvironment(env_name, no_graphics=True, worker_id=self.worker_index)
            self.env = UnityToGymWrapper(unity_env, use_visual=False) #
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

        def reset(self):
            return self.env.reset()

        def step(self, action):
            return self.env.step(action)

    register_env("unity_env", lambda config: UnityEnvWrapper(config))

    ray.init()

    tune.run(
        "PPO",
        name="unity_hallway_ppo",
        stop={
            "timesteps_total": 100,
        },
        config={
            "env": "unity_env",
            "num_workers": 0,
            "env_config":{
            },
            "train_batch_size": 500,
            # "monitor": True
        },
        checkpoint_at_end=True,
    )
