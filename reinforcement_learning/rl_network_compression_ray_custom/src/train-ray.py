import json
import os

import gym
import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo
from environment import NetworkCompression
from ray.tune import run_experiments
from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher


def create_environment(env_config):
    # This import must happen inside the method so that worker processes import this code
    from environment import Compression

    return Compression()


class MyLauncher(SageMakerRayLauncher):
    def __init__(self):
        super(MyLauncher, self).__init__()
        self.num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))
        self.hosts_info = json.loads(os.environ.get("SM_RESOURCE_CONFIG"))["hosts"]
        self.num_total_gpus = self.num_gpus * len(self.hosts_info)

    def register_env_creator(self):
        register_env("NetworkCompression-v1", create_environment)

    def get_experiment_config(self):
        return {
            "training": {
                "env": "NetworkCompression-v1",
                "run": "A3C",
                "stop": {
                    "training_iteration": 20,
                },
                "local_dir": "/opt/ml/model/",
                "checkpoint_freq": 1,
                "config": {
                    "num_workers": max(self.num_total_gpus - 1, 1),
                    "use_gpu_for_workers": True,
                    "train_batch_size": 5,
                    "sample_batch_size": 1,
                    "gpu_fraction": 0.3,
                    "optimizer": {"grads_per_step": 10},
                },
                "trial_resources": {
                    "cpu": 0,
                    "gpu": 1,
                    "extra_gpu": max(self.num_total_gpus - 1, 1),
                    "extra_cpu": 0,
                },
            }
        }


if __name__ == "__main__":
    os.environ["LC_ALL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"
    os.environ["RAY_USE_XRAY"] = "1"
    print(a3c.DEFAULT_CONFIG)
    MyLauncher().train_main()
