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
import os
from pathlib import Path

import yaml
from ray.tune.resources import json_to_resources, resources_to_json
from ray.tune.result import DEFAULT_RESULTS_DIR

try:
    from custom.callbacks import CustomCallbacks
except ModuleNotFoundError:
    from callbacks import CustomCallbacks


class RayExperimentBuilder:
    EXAMPLE_USAGE = """
        Training example:
            python ./train.py --run DQN --env CartPole-v0

        Training with Config:
            python ./train.py -f experiments/simple-corridor-0.yaml


        Note that -f overrides all other trial-specific command-line options.
        """

    def __init__(self, **kwargs):
        parser = self.create_parser()
        self.args, _ = parser.parse_known_args()

        if kwargs is not None:
            for k, v in kwargs.items():
                self.args.__dict__[k] = v

        # Convert jsons to dicts in local mode
        self.args.scheduler_config = self.try_convert_json_to_dict(self.args.scheduler_config)
        self.args.config = self.try_convert_json_to_dict(self.args.config)
        self.args.stop = self.try_convert_json_to_dict(self.args.stop)

    def try_convert_json_to_dict(self, json_string):
        try:
            return json.loads(json_string)
        except TypeError:
            return json_string

    def make_parser(self, **kwargs):
        # TODO import method from starter-kit
        # Taken from https://github.com/ray-project/ray/blob/5303c3abe322cbd90f75bcf03ee1f9c3dad23aae/python/ray/tune/config_parser.py
        parser = argparse.ArgumentParser(**kwargs)

        parser.add_argument(
            "--run",
            default=None,
            type=str,
            help="The algorithm or model to train. This may refer to the name "
            "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
            "user-defined trainable function or class registered in the "
            "tune registry.",
        )
        parser.add_argument(
            "--stop",
            default="{}",
            help="The stopping criteria, specified in JSON. The keys may be any "
            "field returned by 'train()' e.g. "
            '\'{"time_total_s": 600, "training_iteration": 100000}\' to stop '
            "after 600 seconds or 100k iterations, whichever is reached first.",
        )
        parser.add_argument(
            "--config",
            default="{}",
            help="Algorithm-specific configuration (e.g. env, hyperparams), " "specified in JSON.",
        )
        parser.add_argument(
            "--resources-per-trial",
            default=None,
            type=json_to_resources,
            help="Override the machine resources to allocate per trial, e.g. "
            '\'{"cpu": 64, "gpu": 8}\'. Note that GPUs will not be assigned '
            "unless you specify them here. For RLlib, you probably want to "
            "leave this alone and use RLlib configs to control parallelism.",
        )
        parser.add_argument(
            "--num-samples", default=1, type=int, help="Number of times to repeat each trial."
        )
        parser.add_argument(
            "--checkpoint-freq",
            default=0,
            type=int,
            help="How many training iterations between checkpoints. "
            "A value of 0 (default) disables checkpointing.",
        )
        parser.add_argument(
            "--checkpoint-at-end",
            action="store_true",
            help="Whether to checkpoint at the end of the experiment. " "Default is False.",
        )
        parser.add_argument(
            "--sync-on-checkpoint",
            action="store_true",
            help="Enable sync-down of trial checkpoint to guarantee "
            "recoverability. If unset, checkpoint syncing from worker "
            "to driver is asynchronous, so unset this only if synchronous "
            "checkpointing is too slow and trial restoration failures "
            "can be tolerated.",
        )
        parser.add_argument(
            "--keep-checkpoints-num",
            default=None,
            type=int,
            help="Number of best checkpoints to keep. Others get "
            "deleted. Default (None) keeps all checkpoints.",
        )
        parser.add_argument(
            "--checkpoint-score-attr",
            default="training_iteration",
            type=str,
            help="Specifies by which attribute to rank the best checkpoint. "
            "Default is increasing order. If attribute starts with min- it "
            "will rank attribute in decreasing order. Example: "
            "min-validation_loss",
        )
        parser.add_argument(
            "--export-formats",
            default=None,
            help="List of formats that exported at the end of the experiment. "
            "Default is None. For RLlib, 'checkpoint' and 'model' are "
            "supported for TensorFlow policy graphs.",
        )
        parser.add_argument(
            "--max-failures",
            default=3,
            type=int,
            help="Try to recover a trial from its last checkpoint at least this "
            "many times. Only applies if checkpointing is enabled.",
        )
        parser.add_argument(
            "--scheduler",
            default="FIFO",
            type=str,
            help="FIFO (default), MedianStopping, AsyncHyperBand, " "HyperBand, or HyperOpt.",
        )
        parser.add_argument(
            "--scheduler-config", default="{}", help="Config options to pass to the scheduler."
        )

        # Note: this currently only makes sense when running a single trial
        parser.add_argument(
            "--restore", default=None, type=str, help="If specified, restore from this checkpoint."
        )

        return parser

    def create_parser(self):
        parser = self.make_parser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Train a reinforcement learning agent.",
            epilog=self.EXAMPLE_USAGE,
        )

        # See also the base parser definition in ray/tune/config_parser.py
        parser.add_argument(
            "--ray-address",
            default=None,
            type=str,
            help="Connect to an existing Ray cluster at this address instead "
            "of starting a new one.",
        )
        parser.add_argument(
            "--ray-num-cpus",
            default=None,
            type=int,
            help="--num-cpus to use if starting a new cluster.",
        )
        parser.add_argument(
            "--ray-num-gpus",
            default=None,
            type=int,
            help="--num-gpus to use if starting a new cluster.",
        )
        parser.add_argument(
            "--ray-num-nodes",
            default=None,
            type=int,
            help="Emulate multiple cluster nodes for debugging.",
        )
        parser.add_argument(
            "--ray-redis-max-memory",
            default=None,
            type=int,
            help="--redis-max-memory to use if starting a new cluster.",
        )
        parser.add_argument(
            "--ray-memory",
            default=None,
            type=int,
            help="--memory to use if starting a new cluster.",
        )
        parser.add_argument(
            "--ray-object-store-memory",
            default=None,
            type=int,
            help="--object-store-memory to use if starting a new cluster.",
        )
        parser.add_argument(
            "--experiment-name",
            default="default",
            type=str,
            help="Name of the subdirectory under `local_dir` to put results in.",
        )
        parser.add_argument(
            "--local-dir",
            default=DEFAULT_RESULTS_DIR,
            type=str,
            help="Local dir to save training results to. Defaults to '{}'.".format(
                DEFAULT_RESULTS_DIR
            ),
        )
        parser.add_argument(
            "--upload-dir",
            default="",
            type=str,
            help="Optional URI to sync training results to (e.g. s3://bucket).",
        )
        parser.add_argument("-v", action="store_true", help="Whether to use INFO level logging.")
        parser.add_argument("-vv", action="store_true", help="Whether to use DEBUG level logging.")
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Whether to attempt to resume previous Tune experiments.",
        )
        parser.add_argument(
            "--torch",
            action="store_true",
            help="Whether to use PyTorch (instead of tf) as the DL framework.",
        )
        parser.add_argument(
            "--eager", action="store_true", help="Whether to attempt to enable TF eager execution."
        )
        parser.add_argument(
            "--trace",
            action="store_true",
            help="Whether to attempt to enable tracing for eager mode.",
        )
        parser.add_argument("--env", default=None, type=str, help="The gym environment to use.")
        parser.add_argument(
            "--queue-trials",
            action="store_true",
            help=(
                "Whether to queue trials when the cluster does not currently have "
                "enough resources to launch one. This should be set to True when "
                "running on an autoscaling cluster to enable automatic scale-up."
            ),
        )
        parser.add_argument(
            "-f",
            "--config-file",
            default=None,
            type=str,
            help="If specified, use config options from this file. Note that this "
            "overrides any trial-specific options set via flags above.",
        )

        return parser

    def get_experiment_definition(self):
        if self.args.config_file:
            with open(self.args.config_file) as f:
                experiments = yaml.safe_load(f)
                exp_name_list = list(experiments.keys())
                assert len(exp_name_list) == 1
                # overwrite experiment name for SageMaker to recognize
                experiments["training"] = experiments.pop(exp_name_list[0])
        else:
            experiments = {
                self.args.experiment_name: {  # i.e. log to ~/ray_results/default
                    "run": self.args.run,
                    "checkpoint_freq": self.args.checkpoint_freq,
                    "keep_checkpoints_num": self.args.keep_checkpoints_num,
                    "checkpoint_score_attr": self.args.checkpoint_score_attr,
                    "local_dir": self.args.local_dir,
                    "resources_per_trial": (
                        self.args.resources_per_trial
                        and resources_to_json(self.args.resources_per_trial)
                    ),
                    "stop": self.args.stop,
                    "config": dict(self.args.config, env=self.args.env),
                    "restore": self.args.restore,
                    "num_samples": self.args.num_samples,
                    "upload_dir": self.args.upload_dir,
                }
            }

        verbose = 1
        for exp in experiments.values():
            # Bazel makes it hard to find files specified in `args` (and `data`).
            # Look for them here.
            # NOTE: Some of our yaml files don't have a `config` section.
            if exp.get("config", {}).get("input") and not os.path.exists(exp["config"]["input"]):
                # This script runs in the ray/rllib dir.
                rllib_dir = Path(__file__).parent
                input_file = rllib_dir.absolute().joinpath(exp["config"]["input"])
                exp["config"]["input"] = str(input_file)

            if not exp.get("run"):
                raise ValueError("The following arguments are required: run")
            if not exp.get("env") and not exp.get("config", {}).get("env"):
                raise ValueError("The following arguments are required: env")

            if self.args.eager:
                exp["config"]["eager"] = True
            if self.args.torch:
                exp["config"]["use_pytorch"] = True
            if self.args.v:
                exp["config"]["log_level"] = "INFO"
                verbose = 2
            if self.args.vv:
                exp["config"]["log_level"] = "DEBUG"
                verbose = 3
            if self.args.trace:
                if not exp["config"].get("eager"):
                    raise ValueError("Must enable --eager to enable tracing.")
                exp["config"]["eager_tracing"] = True

            # Add Custom Callbacks
            exp["config"]["callbacks"] = CustomCallbacks
        return experiments, self.args, verbose
