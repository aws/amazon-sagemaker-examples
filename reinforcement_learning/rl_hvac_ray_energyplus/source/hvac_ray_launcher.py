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
import ast
import json
import os
import subprocess

import ray
from ray.tune.tune import _make_scheduler, run, run_experiments
from sagemaker_rl.ray_launcher import SageMakerRayLauncher
from sagemaker_rl.tf_serving_utils import export_tf_serving, natural_keys

TERMINATION_SIGNAL = "JOB_TERMINATED"
MODEL_OUTPUT_DIR = "/opt/ml/model"
CHECKPOINTS_DIR = "/opt/ml/checkpoints"


def custom_sync_func(source, target):
    """Custom rsync cmd to sync experiment artifact from remote nodes to driver node."""
    sync_cmd = (
        'rsync -havP --inplace --stats -e "ssh -i /root/.ssh/id_rsa" {source} {target}'.format(
            source=source, target=target
        )
    )

    sync_process = subprocess.Popen(sync_cmd, shell=True)
    sync_process.wait()


class HVACSageMakerRayLauncher(SageMakerRayLauncher):
    """Launcher class for Procgen experiments using Ray-RLLib.
    Customers should sub-class this, fill in the required methods, and
    call .train_main() to start a training process.

    Example::

        class MyLauncher(ProcgenSageMakerRayLauncher):
            def register_env_creator(self):
                register_env(
                    "stacked_procgen_env",  # This should be different from procgen_env_wrapper
                    lambda config: gym.wrappers.FrameStack(ProcgenEnvWrapper(config), 4)
                )

            def get_experiment_config(self):
                return {
                  "training": {
                    "env": "procgen_env_wrapper",
                    "run": "PPO",
                    ...
                  }
                }

        if __name__ == "__main__":
            MyLauncher().train_main()
    """

    def register_algorithms_and_preprocessors(self):
        raise NotImplementedError()

    def create_tf_serving_model(self, algorithm=None, env_string=None):
        self.register_env_creator()
        self.register_algorithms_and_preprocessors()
        if ray.__version__ >= "0.6.5":
            from ray.rllib.agents.registry import get_agent_class
        else:
            from ray.rllib.agents.agent import get_agent_class
        cls = get_agent_class(algorithm)
        with open(os.path.join(MODEL_OUTPUT_DIR, "params.json")) as config_json:
            config = json.load(config_json)
        use_torch = config.get("use_pytorch", False)
        if not use_torch:
            if "callbacks" in config:
                callback_cls_str = config["callbacks"]
                callback_cls = callback_cls_str.split("'")[-2].split(".")[-1]
                config["callbacks"] = ast.literal_eval()(callback_cls)
            print("Loaded config for TensorFlow serving.")
            config["monitor"] = False
            config["num_workers"] = 1
            config["num_gpus"] = 0
            agent = cls(env=env_string, config=config)
            checkpoint = os.path.join(MODEL_OUTPUT_DIR, "checkpoint")
            agent.restore(checkpoint)
            export_tf_serving(agent, MODEL_OUTPUT_DIR)

    def find_checkpoint_path_for_spot(self, prefix):
        ckpts = []
        ckpts_prefix = ""
        for root, directories, files in os.walk(prefix):
            for directory in directories:
                if directory.startswith("checkpoint"):
                    if not ckpts_prefix:
                        ckpts_prefix = root
                    ckpts.append(directory)
        return ckpts_prefix, ckpts

    def find_checkpoint_file_for_spot(self, prefix):
        ckpts_prefix, ckpts = self.find_checkpoint_path_for_spot(prefix)
        if not ckpts:
            return ""
        else:
            ckpts.sort(key=natural_keys)
            ckpt_name = ckpts[-1].replace("_", "-")
            return os.path.join(ckpts_prefix, ckpts[-1], ckpt_name)

    def launch(self):
        """Actual entry point into the class instance where everything happens."""
        self.register_env_creator()
        self.register_algorithms_and_preprocessors()
        experiment_config, args, verbose = self.get_experiment_config()

        # All worker nodes will block at this step during training
        ray_cluster_config = self.ray_init_config()
        if not self.is_master_node:
            return
        ray_custom_cluster_config = {
            "object_store_memory": args.ray_object_store_memory,
            "memory": args.ray_memory,
            "redis_max_memory": args.ray_redis_max_memory,
            "num_cpus": args.ray_num_cpus,
            "num_gpus": args.ray_num_gpus,
        }
        all_workers_host_names = self.get_all_host_names()[1:]
        # Overwrite redis address for single instance job
        if len(all_workers_host_names) == 0:
            ray_custom_cluster_config.update({"address": args.ray_address})
        ray_cluster_config.update(ray_custom_cluster_config)

        # Start the driver on master node
        ray.init(**ray_cluster_config)

        # Spot instance is back
        if os.path.exists(CHECKPOINTS_DIR) and os.listdir(CHECKPOINTS_DIR):
            print("Instance is back. Local checkpoint path detected.")
            checkpoint_file = self.find_checkpoint_file_for_spot(CHECKPOINTS_DIR)
            print("Setting checkpoint path to {}".format(checkpoint_file))
            if checkpoint_file:
                experiment_config["training"]["restore"] = checkpoint_file  # Overwrite
        experiment_config = self.customize_experiment_config(experiment_config)
        experiment_config = self.set_up_checkpoint(experiment_config)
        experiment_config["training"]["sync_to_driver"] = custom_sync_func

        run_experiments(
            experiment_config,
            scheduler=_make_scheduler(args),
            queue_trials=args.queue_trials,
            resume=args.resume,
            verbose=verbose,
            concurrent=True,
        )
        # If distributed job, send TERMINATION_SIGNAL to all workers.
        if len(all_workers_host_names) > 0:
            self.sage_cluster_communicator.create_s3_signal(TERMINATION_SIGNAL)

    @classmethod
    def train_main(cls, args):
        """main function that kicks things off"""
        launcher = cls(args)
        launcher.launch()
