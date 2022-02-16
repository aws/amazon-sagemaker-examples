import json
import os

import gym
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from sagemaker_rl.ray_launcher import SageMakerRayLauncher


def create_environment(env_config):
    return gym.make("Hopper-v3")


class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):
        register_env("Hopper-v3", create_environment)

    def get_experiment_config(self):
        return {
            "training": {
                "env": "Hopper-v3",
                "run": "PPO",
                "stop": {"training_iteration": 10},
                "config": {
                    "framework": "tf",
                    "gamma": 0.99,
                    "kl_coeff": 1.0,
                    "num_sgd_iter": 20,
                    "lr": 0.0001,
                    "sgd_minibatch_size": 1000,
                    "train_batch_size": 25000,
                    "model": {"free_log_std": True},
                    "num_workers": (self.num_cpus - 1),
                    "num_gpus": self.num_gpus,
                    "batch_mode": "truncate_episodes",
                },
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()
