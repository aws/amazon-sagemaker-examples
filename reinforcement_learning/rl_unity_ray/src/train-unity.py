import json
import os

import gym
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from sagemaker_rl.ray_launcher import SageMakerRayLauncher
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.registry import default_registry


class UnityEnvWrapper(gym.Env):
    def __init__(self, env_config):
        self.worker_index = env_config.worker_index
        if 'SM_CHANNEL_TRAIN' in os.environ:
            print(os.environ['SM_CHANNEL_TRAIN'])
            print(os.listdir(os.environ['SM_CHANNEL_TRAIN']))
            env_name = os.environ['SM_CHANNEL_TRAIN'] + env_config['env_name']
            unity_env = UnityEnvironment(env_name, no_graphics=True, worker_id=self.worker_index)
        else:
            unity_env = default_registry[env_config['env_name']].make(no_graphics=True, worker_id=self.worker_index) 
            
        self.env = UnityToGymWrapper(unity_env) 
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class MyLauncher(SageMakerRayLauncher):

    def register_env_creator(self):
        register_env("unity_env", lambda config: UnityEnvWrapper(config))

    def get_experiment_config(self):
        return {
          "training": {
            "run": "PPO",
            "stop": {
              "timesteps_total": 10000,
            },
            "config": {
              "env": "unity_env",
              "gamma": 0.995,
              "kl_coeff": 1.0,
              "num_sgd_iter": 20,
              "lr": 0.0001,
              "sgd_minibatch_size": 100,
              "train_batch_size": 500,
              "monitor": True,  # Record videos.
              "model": {
                "free_log_std": True
              },
              "env_config":{
                "env_name": "Basic"
              },
              "num_workers": (self.num_cpus-1),
              # "num_workers": 0,
            }
          }
        }

if __name__ == "__main__":
    # import time
    # time.sleep(10000)
    MyLauncher().train_main()
