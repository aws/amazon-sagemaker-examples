import json
import os
import sys

import gym
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher

env_config = {}


class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):
        from gameserver_env import GameServerEnv

        register_env("GameServers-v0", lambda env_config: GameServerEnv(env_config))

    def get_experiment_config(self):
        print("get_experiment_config")
        print(env_config)
        # allowing 4600 seconds to the job toto stop and save the model
        time_total_s = float(env_config["time_total_s"]) - 4600
        print("time_total_s=" + str(time_total_s))
        return {
            "training": {
                "env": "GameServers-v0",
                "run": "PPO",
                "stop": {"time_total_s": time_total_s},
                "config": {
                    "ignore_worker_failures": True,
                    "gamma": 0,
                    "kl_coeff": 1.0,
                    "num_sgd_iter": 10,
                    "lr": 0.0001,
                    "sgd_minibatch_size": 32,
                    "train_batch_size": 128,
                    "model": {
                        #                 "free_log_std": True,
                        #                  "fcnet_hiddens": [512, 512],
                    },
                    "use_gae": True,
                    # "num_workers": (self.num_cpus-1),
                    "num_gpus": self.num_gpus,
                    # "batch_mode": "complete_episodes",
                    "num_workers": 1,
                    "env_config": env_config,
                    #'observation_filter': 'MeanStdFilter',
                },
            }
        }


if __name__ == "__main__":
    for i in range(len(sys.argv)):
        if i == 0:
            continue
        if i % 2 > 0:
            env_config[sys.argv[i].split("--", 1)[1]] = sys.argv[i + 1]
    MyLauncher().train_main()
