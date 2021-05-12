# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import json
import multiprocessing
import os
import subprocess

from eplus.envs.medium_office_env import MediumOfficeEnv
from hvac_ray_launcher import HVACSageMakerRayLauncher
from ray.tune.registry import register_env
from ray_experiment_builder import RayExperimentBuilder


class MyLauncher(HVACSageMakerRayLauncher):
    def __init__(self, args):
        super().__init__()

        self.n_days = args.n_days
        self.n_iter = args.n_iter
        self.algorithm = args.algorithm
        self.multi_zone_control = args.multi_zone_control
        self.energy_temp_penalty_ratio = args.energy_temp_penalty_ratio

        self.num_cpus = multiprocessing.cpu_count()

        try:
            self.num_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count("UUID")
        except:
            self.num_gpus = 0

    def register_env_creator(self):
        register_env("medium_office_env", lambda env_config: MediumOfficeEnv(env_config))

    def _get_ray_config(self):
        return {
            "ray_num_cpus": self.num_cpus,  # adjust based on selected instance type
            "ray_num_gpus": self.num_gpus,
            "eager": False,
            "v": True,  # requried for CW to catch the progress
        }

    def _get_rllib_config(self):
        return {
            "experiment_name": "training",
            "run": self.algorithm,
            "env": "medium_office_env",
            "stop": {
                "training_iteration": self.n_iter,
            },
            "checkpoint_freq": 20,
            "checkpoint_at_end": True,
            "keep_checkpoints_num": 5,
            "queue_trials": False,
            "config": {
                "tau": 1,
                # === Settings for Rollout Worker processes ===
                "num_workers": self.num_cpus - 1,
                #                 "rollout_fragment_length": 140,
                #                 "batch_mode": "truncate_episodes",
                # === Advanced Resource Settings ===
                "num_envs_per_worker": 1,
                "num_cpus_per_worker": 1,
                "num_cpus_for_driver": 1,
                "num_gpus_per_worker": 0,
                # === Settings for the Trainer process ===
                "num_gpus": self.num_gpus,  # adjust based on number of GPUs available in a single node, e.g., p3.2xlarge has 1 GPU
                # === Exploration Settings ===
                #                 "explore": True,
                #                 "exploration_config": {
                #                     "type": "StochasticSampling",
                #                 },
                "model": {
                    # == LSTM ==
                    # Whether to wrap the model with an LSTM.
                    "use_lstm": True,
                    "fcnet_hiddens": [256, 256],
                    # Max seq len for training the LSTM, defaults to 20.
                    "max_seq_len": 20,
                    # Size of the LSTM cell.
                    "lstm_cell_size": 256,
                    # Whether to feed a_{t-1}, r_{t-1} to LSTM.
                    "lstm_use_prev_action_reward": True,
                },
                # === Settings for the Procgen Environment ===
                "env_config": {
                    "eplus_path": "/usr/local/EnergyPlus-9-3-0/",
                    "weather_file": "weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
                    "days": self.n_days,
                    "timestep": 4,
                    "multi_zone_control": self.multi_zone_control,
                    "energy_temp_penalty_ratio": self.energy_temp_penalty_ratio,
                },
            },
        }

    def register_algorithms_and_preprocessors(self):
        pass

    def get_experiment_config(self):
        params = dict(self._get_ray_config())
        params.update(self._get_rllib_config())
        reb = RayExperimentBuilder(**params)

        print(reb)
        return reb.get_experiment_definition()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_days", type=int, default=365, help="Number of days to simulate.")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of days to simulate.")
    parser.add_argument(
        "--algorithm", type=str, default="APEX_DDPG", help="Number of days to simulate."
    )
    parser.add_argument(
        "--multi_zone_control", type=bool, default=True, help="Number of days to simulate."
    )
    parser.add_argument(
        "--energy_temp_penalty_ratio", type=float, default=10, help="Number of days to simulate."
    )

    args = parser.parse_args()

    MyLauncher(args).launch()
