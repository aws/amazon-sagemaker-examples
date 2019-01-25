import json
import os
import subprocess
import sys
import time
from shutil import copyfile
from enum import Enum

import boto3

import ray
from ray.tune import run_experiments
from ray.rllib.agents.agent import get_agent_class

from .tf_serving_utils import export_tf_serving, natural_keys, change_permissions_recursive
from .configuration_list import ConfigurationList
from .sage_cluster_communicator import SageClusterCommunicator
from .docker_utils import get_ip_from_host

TERMINATION_SIGNAL = "JOB_TERMINATED"
INTERMEDIATE_DIR = "/opt/ml/output/intermediate"
MODEL_OUTPUT_DIR = "/opt/ml/model"


class Cluster(Enum):
    """
    Used when training is done in heterogeneous mode, i.e. 2 SageMaker jobs are launched with
    different instance types. Usually, primary cluster has a single GPU instance responsible
    for Neural Network training and secondary cluster has CPU instances for rollouts.
    For single machine or homogeneous cluster, primary is the default type.
    """
    Primary = "primary"
    Secondary = "secondary"


class SageMakerRayLauncher(object):
    """Base class for SageMaker RL applications using Ray-RLLib.
    Customers should sub-class this, fill in the required methods, and 
    call .train_main() to start a training process.

    Example::

        def create_environment(env_config):
            # Import must happen inside the method so workers re-import
            import roboschool
            return gym.make('RoboschoolHumanoid-v1')

        class MyLauncher(SageMakerRayLauncher):
            def register_env_creator(self):
                register_env("RoboschoolHumanoid-v1", create_environment)
                
            def get_experiment_config(self):
                return {
                  "training": {
                    "env": "RoboschoolHumanoid-v1",
                    "run": "PPO",
                    ...
                  }
                }

        if __name__ == "__main__":
            MyLauncher().train_main()
    """

    def __init__(self):
        self.num_cpus = int(os.environ.get("SM_NUM_CPUS", 1))
        self.num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))

        self.cluster_type = self._get_cluster_type()
        self.num_instances_secondary_cluster = int(os.environ.get("SM_HP_RL_NUM_INSTANCES_SECONDARY", 0))
        self.host_name = os.environ.get("SM_CURRENT_HOST", "algo-1")
        self.hosts_info = json.loads(os.environ.get("SM_RESOURCE_CONFIG"))["hosts"]
        self.is_master_node = self.hosts_info[0] == self.host_name and self.cluster_type == Cluster.Primary

        self.sage_cluster_communicator = SageClusterCommunicator()

    def _get_cluster_type(self):
        cluster_str = os.environ.get("SM_HP_RL_CLUSTER_TYPE", "primary")
        if cluster_str.lower() == "primary":
            return Cluster.Primary
        else:
            return Cluster.Secondary

    def register_env_creator(self):
        """Sub-classes must implement this.
        """
        raise NotImplementedError("Subclasses should implement this to call ray.tune.registry.register_env")

    def get_experiment_config(self):
        raise NotImplementedError("Subclasses must define the experiment config to pass to ray.tune.run_experiments")

    def customize_experiment_config(self, config):
        """Applies command-line hyperparameters to the config.
        """
        # TODO: use ConfigList from Coach launcher, and share customization code.
        hyperparams_dict = json.loads(os.environ.get("SM_HPS", "{}"))

        # Set output dir to intermediate
        # TODO: move this to before customer-specified so they can override
        hyperparams_dict["rl.training.local_dir"] = INTERMEDIATE_DIR
        hyperparams_dict["rl.training.checkpoint_at_end"] = True
        hyperparams_dict["rl.training.checkpoint_freq"] = 10
        self.hyperparameters = ConfigurationList()  # TODO: move to shared
        for name, value in hyperparams_dict.items():
            # self.map_hyperparameter(name, val) #TODO
            if name.startswith("rl."):
                # self.apply_hyperparameter(name, value)  #TODO
                self.hyperparameters.store(name, value)
                #             else:
                #                 raise ValueError("Unknown hyperparameter %s" % name)
        self.hyperparameters.apply_subset(config, "rl.")
        return config

    def get_all_host_names(self):
        all_workers_host_names = []
        for host in self.hosts_info:
            # All primary cluster instances' hostnames. Prefix with "primary"
            all_workers_host_names.append("%s:%s" % (self.cluster_type.value, host))
        for i in range(self.num_instances_secondary_cluster):
            # All secondary cluster instances' hostnames. Prefix with "secondary"
            all_workers_host_names.append("%s:algo-%s" % (Cluster.Secondary.value, i + 1))
        return all_workers_host_names

    def ray_init_config(self):
        num_workers = max(self.num_cpus, 3)
        config = {"num_cpus": num_workers, "num_gpus": self.num_gpus}

        if self.is_master_node:
            all_wokers_host_names = self.get_all_host_names()[1:]
            # Single machine job
            if len(all_wokers_host_names) == 0:
                return config
            master_ip = get_ip_from_host(host_name=self.host_name)
            self.start_ray_cluster(master_ip)
            self.sage_cluster_communicator.write_host_config(ip=master_ip,
                                                             host_name="%s:%s" % (
                                                             self.cluster_type.value, self.host_name))
            self.sage_cluster_communicator.create_s3_signal("%s:%s" % (self.cluster_type.value, self.host_name))
            print("Waiting for %s worker nodes to join!" % (len(all_wokers_host_names)))
            self.sage_cluster_communicator.wait_for_signals(all_wokers_host_names)
            print("All worker nodes have joined the cluster. Now training...")
            config = {"redis_address": "%s:6379" % master_ip}
        else:
            master_ip, master_hostname = self.sage_cluster_communicator.get_master_config()
            node_ip = get_ip_from_host(host_name=self.host_name)
            self.sage_cluster_communicator.wait_for_signals([master_hostname])
            print("Attempting to join ray cluster.")
            self.join_ray_cluster(master_ip, node_ip)
            self.sage_cluster_communicator.create_s3_signal("%s:%s" % (self.cluster_type.value, self.host_name))
            print("Joined ray cluster at %s successfully!" % master_ip)
            self.sage_cluster_communicator.wait_for_signals([TERMINATION_SIGNAL], timeout=sys.maxsize)
            print("Received job termination signal. Shutting down.")

        return config

    def start_ray_cluster(self, master_ip):
        p = subprocess.Popen("ray start --head --redis-port=6379 --no-ui --node-ip-address=%s" % master_ip,
                             shell=True,
                             stderr=subprocess.STDOUT)
        time.sleep(3)
        if p.poll() != 0:
            raise RuntimeError("Could not start Ray server.")

    def join_ray_cluster(self, master_ip, node_ip):
        p = subprocess.Popen("ray start --redis-address=%s:6379 --node-ip-address=%s" % (master_ip, node_ip),
                             shell=True, stderr=subprocess.STDOUT)
        time.sleep(3)
        if p.poll() != 0:
            raise RuntimeError("Could not join Ray server running at %s:6379" % master_ip)

    def copy_checkpoints_to_model_output(self):
        checkpoints = []
        for root, directories, filenames in os.walk(INTERMEDIATE_DIR):
            for filename in filenames:
                if filename.startswith("checkpoint"):
                    checkpoints.append(os.path.join(root, filename))
        checkpoints.sort(key=natural_keys)
        latest_checkpoints = checkpoints[-2:]
        validation = sum(1 if x.endswith("tune_metadata") or x.endswith("extra_data") else 0 for x in
                         latest_checkpoints)
        if validation is not 2:
            raise RuntimeError("Failed to save checkpoint files - .tune_metadata or .extra_data")
        for source_path in latest_checkpoints:
            _, ext = os.path.splitext(source_path)
            destination_path = os.path.join(MODEL_OUTPUT_DIR, "checkpoint%s" % ext)
            copyfile(source_path, destination_path)
            print("Saved the checkpoint file %s as %s" % (source_path, destination_path))

    def save_experiment_config(self, config):
        with open(os.path.join(MODEL_OUTPUT_DIR, "params.json"), "w") as f:
            json.dump(config, f, indent=2)
        print("Saved model configuration.")

    def create_tf_serving_model(self, algorithm=None, env_string=None, config=None):
        self.register_env_creator()
        cls = get_agent_class(algorithm)
        config["monitor"] = False
        config["num_workers"] = 1
        config["num_gpus"] = 0
        agent = cls(env=env_string, config=config)
        checkpoint = os.path.join(MODEL_OUTPUT_DIR, "checkpoint")
        agent.restore(checkpoint)
        export_tf_serving(agent, MODEL_OUTPUT_DIR)

    def save_checkpoint_and_serving_model(self, algorithm=None, env_string=None, config=None):
        self.save_experiment_config(config)
        self.copy_checkpoints_to_model_output()
        self.create_tf_serving_model(algorithm, env_string, config)

        # To ensure SageMaker local mode works fine
        change_permissions_recursive(INTERMEDIATE_DIR, 0o777)
        change_permissions_recursive(MODEL_OUTPUT_DIR, 0o777)

    def launch(self):
        """Actual entry point into the class instance where everything happens.
        Lots of delegating to classes that are in subclass or can be over-ridden.
        """
        self.register_env_creator()

        # All worker nodes will block at this step during training
        ray_cluster_config = self.ray_init_config()
        if not self.is_master_node:
            return

        # Start the driver on master node
        ray.init(**ray_cluster_config)
        experiment_config = self.get_experiment_config()
        experiment_config = self.customize_experiment_config(experiment_config)
        print("Running experiment with config %s" % json.dumps(experiment_config, indent=2))
        run_experiments(experiment_config)
        all_wokers_host_names = self.get_all_host_names()[1:]
        # If distributed job, send TERMINATION_SIGNAL to all workers.
        if len(all_wokers_host_names) > 0:
            self.sage_cluster_communicator.create_s3_signal(TERMINATION_SIGNAL)

        algo = experiment_config["training"]["run"]
        env_string = experiment_config["training"]["config"]["env"]
        config = experiment_config["training"]["config"]
        self.save_checkpoint_and_serving_model(algorithm=algo,
                                               env_string=env_string,
                                               config=config)


    @classmethod
    def train_main(cls):
        """main function that kicks things off
        """
        launcher = cls()
        launcher.launch()
