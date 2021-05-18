import json
import os

import gym
import ray
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.registry import default_registry
from ray.tune import run_experiments
from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher


class UnityEnvWrapper(gym.Env):
    def __init__(self, env_config):
        self.worker_index = env_config.worker_index
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
                    "model": {"free_log_std": True},
                    "env_config": {"env_name": "Basic"},
                    "num_workers": (self.num_cpus - 1),
                    "ignore_worker_failures": True,
                },
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()
