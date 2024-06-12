# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Placeholder docstring"""
from __future__ import absolute_import

import base64
import copy
import errno
import json
import logging
import os
import platform
import random
import re
import shlex
import shutil
import string
import subprocess
import sys
import tarfile
import tempfile

from distutils.spawn import find_executable
from threading import Thread
from six.moves.urllib.parse import urlparse

import sagemaker
import sagemaker.local.data
import sagemaker.local.utils
import sagemaker.utils

CONTAINER_PREFIX = "algo"
DOCKER_COMPOSE_FILENAME = "docker-compose.yaml"
DOCKER_COMPOSE_HTTP_TIMEOUT_ENV = "COMPOSE_HTTP_TIMEOUT"
DOCKER_COMPOSE_HTTP_TIMEOUT = "120"

# Environment variables to be set during training
REGION_ENV_NAME = "AWS_REGION"
TRAINING_JOB_NAME_ENV_NAME = "TRAINING_JOB_NAME"
S3_ENDPOINT_URL_ENV_NAME = "S3_ENDPOINT_URL"

# SELinux Enabled
SELINUX_ENABLED = os.environ.get("SAGEMAKER_LOCAL_SELINUX_ENABLED", "False").lower() in [
    "1",
    "true",
    "yes",
]

logger = logging.getLogger(__name__)


class _SageMakerContainer(object):
    """Handle the lifecycle and configuration of a local container execution.

    This class is responsible for creating the directories and configuration
    files that the docker containers will use for either training or serving.
    """

    def __init__(
        self,
        instance_type,
        instance_count,
        image,
        sagemaker_session=None,
        container_entrypoint=None,
        container_arguments=None,
    ):
        """Initialize a SageMakerContainer instance

        It uses a :class:`sagemaker.session.Session` for general interaction
        with user configuration such as getting the default sagemaker S3 bucket.
        However this class does not call any of the SageMaker APIs.

        Args:
            instance_type (str): The instance type to use. Either 'local' or
                'local_gpu'
            instance_count (int): The number of instances to create.
            image (str): docker image to use.
            sagemaker_session (sagemaker.session.Session): a sagemaker session
                to use when interacting with SageMaker.
            container_entrypoint (str): the container entrypoint to execute
            container_arguments (str): the container entrypoint arguments
        """
        from sagemaker.local.local_session import LocalSession

        # check if docker-compose is installed
        if find_executable("docker-compose") is None:
            raise ImportError(
                "'docker-compose' is not installed. "
                "Local Mode features will not work without docker-compose. "
                "For more information on how to install 'docker-compose', please, see "
                "https://docs.docker.com/compose/install/"
            )

        self.sagemaker_session = sagemaker_session or LocalSession()
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.image = image
        self.container_entrypoint = container_entrypoint
        self.container_arguments = container_arguments
        # Since we are using a single docker network, Generate a random suffix to attach to the
        # container names. This way multiple jobs can run in parallel.
        suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
        self.hosts = [
            "{}-{}-{}".format(CONTAINER_PREFIX, i, suffix)
            for i in range(1, self.instance_count + 1)
        ]
        self.container_root = None
        self.container = None

    def process(
        self,
        processing_inputs,
        processing_output_config,
        environment,
        processing_job_name,
    ):
        """Run a processing job locally using docker-compose.

        Args:
            processing_inputs (dict): The processing input specification.
            processing_output_config (dict): The processing output configuration specification.
            environment (dict): The environment collection for the processing job.
            processing_job_name (str): Name of the local processing job being run.
        """

        self.container_root = self._create_tmp_folder()

        # A shared directory for all the containers;
        # it is only mounted if the processing script is Local.
        shared_dir = os.path.join(self.container_root, "shared")
        os.mkdir(shared_dir)

        data_dir = self._create_tmp_folder()
        volumes = self._prepare_processing_volumes(
            data_dir, processing_inputs, processing_output_config
        )

        # Create the configuration files for each container that we will create.
        for host in self.hosts:
            _create_processing_config_file_directories(self.container_root, host)
            self.write_processing_config_files(
                host,
                environment,
                processing_inputs,
                processing_output_config,
                processing_job_name,
            )

        self._generate_compose_file(
            "process", additional_volumes=volumes, additional_env_vars=environment
        )
        compose_command = self._compose()

        if _ecr_login_if_needed(self.sagemaker_session.boto_session, self.image):
            _pull_image(self.image)

        process = subprocess.Popen(
            compose_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        try:
            _stream_output(process)
        except RuntimeError as e:
            # _stream_output() doesn't have the command line. We will handle the exception
            # which contains the exit code and append the command line to it.
            msg = f"Failed to run: {compose_command}"
            raise RuntimeError(msg) from e
        finally:
            # Uploading processing outputs back to Amazon S3.
            self._upload_processing_outputs(data_dir, processing_output_config)

            try:
                # Deleting temporary directories.
                dirs_to_delete = [shared_dir, data_dir]
                self._cleanup(dirs_to_delete)
            except OSError:
                pass

        # Print our Job Complete line to have a similar experience to training on SageMaker where
        # you see this line at the end.
        print("===== Job Complete =====")

    def train(self, input_data_config, output_data_config, hyperparameters, environment, job_name):
        """Run a training job locally using docker-compose.

        Args:
            input_data_config (dict): The Input Data Configuration, this contains data such as the
                channels to be used for training.
            output_data_config: The configuration of the output data.
            hyperparameters (dict): The HyperParameters for the training job.
            environment (dict): The environment collection for the training job.
            job_name (str): Name of the local training job being run.

        Returns (str): Location of the trained model.
        """
        self.container_root = self._create_tmp_folder()
        os.mkdir(os.path.join(self.container_root, "output"))
        # create output/data folder since sagemaker-containers 2.0 expects it
        os.mkdir(os.path.join(self.container_root, "output", "data"))
        # A shared directory for all the containers. It is only mounted if the training script is
        # Local.
        shared_dir = os.path.join(self.container_root, "shared")
        os.mkdir(shared_dir)

        data_dir = self._create_tmp_folder()
        volumes = self._prepare_training_volumes(
            data_dir, input_data_config, output_data_config, hyperparameters
        )
        # If local, source directory needs to be updated to mounted /opt/ml/code path
        hyperparameters = self._update_local_src_path(
            hyperparameters, key=sagemaker.estimator.DIR_PARAM_NAME
        )

        # Create the configuration files for each container that we will create
        # Each container will map the additional local volumes (if any).
        for host in self.hosts:
            _create_config_file_directories(self.container_root, host)
            self.write_config_files(host, hyperparameters, input_data_config)
            shutil.copytree(data_dir, os.path.join(self.container_root, host, "input", "data"))

        training_env_vars = {
            REGION_ENV_NAME: self.sagemaker_session.boto_region_name,
            TRAINING_JOB_NAME_ENV_NAME: job_name,
        }
        training_env_vars.update(environment)
        if self.sagemaker_session.s3_resource is not None:
            training_env_vars[
                S3_ENDPOINT_URL_ENV_NAME
            ] = self.sagemaker_session.s3_resource.meta.client._endpoint.host

        compose_data = self._generate_compose_file(
            "train", additional_volumes=volumes, additional_env_vars=training_env_vars
        )
        compose_command = self._compose()

        if _ecr_login_if_needed(self.sagemaker_session.boto_session, self.image):
            _pull_image(self.image)

        process = subprocess.Popen(
            compose_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        try:
            _stream_output(process)
        except RuntimeError as e:
            # _stream_output() doesn't have the command line. We will handle the exception
            # which contains the exit code and append the command line to it.
            msg = "Failed to run: %s, %s" % (compose_command, str(e))
            raise RuntimeError(msg)
        finally:
            artifacts = self.retrieve_artifacts(compose_data, output_data_config, job_name)

            # free up the training data directory as it may contain
            # lots of data downloaded from S3. This doesn't delete any local
            # data that was just mounted to the container.
            dirs_to_delete = [data_dir, shared_dir]
            self._cleanup(dirs_to_delete)

        # Print our Job Complete line to have a similar experience to training on SageMaker where
        # you see this line at the end.
        print("===== Job Complete =====")
        return artifacts

    def serve(self, model_dir, environment):
        """Host a local endpoint using docker-compose.

        Args:
            primary_container (dict): dictionary containing the container runtime settings
                for serving. Expected keys:
                - 'ModelDataUrl' pointing to a file or s3:// location.
                - 'Environment' a dictionary of environment variables to be passed to the
                    hosting container.
        """
        logger.info("serving")

        self.container_root = self._create_tmp_folder()
        logger.info("creating hosting dir in %s", self.container_root)

        volumes = self._prepare_serving_volumes(model_dir)

        # If the user script was passed as a file:// mount it to the container.
        if sagemaker.estimator.DIR_PARAM_NAME.upper() in environment:
            script_dir = environment[sagemaker.estimator.DIR_PARAM_NAME.upper()]
            parsed_uri = urlparse(script_dir)
            if parsed_uri.scheme == "file":
                host_dir = os.path.abspath(parsed_uri.netloc + parsed_uri.path)
                volumes.append(_Volume(host_dir, "/opt/ml/code"))
                # Update path to mount location
                environment = environment.copy()
                environment[sagemaker.estimator.DIR_PARAM_NAME.upper()] = "/opt/ml/code"

        if _ecr_login_if_needed(self.sagemaker_session.boto_session, self.image):
            _pull_image(self.image)

        self._generate_compose_file(
            "serve", additional_env_vars=environment, additional_volumes=volumes
        )
        compose_command = self._compose()

        self.container = _HostingContainer(compose_command)
        self.container.start()

    def stop_serving(self):
        """Stop the serving container.

        The serving container runs in async mode to allow the SDK to do other
        tasks.
        """
        if self.container:
            self.container.down()
            self.container.join()
            self._cleanup()
        # for serving we can delete everything in the container root.
        _delete_tree(self.container_root)

    def retrieve_artifacts(self, compose_data, output_data_config, job_name):
        """Get the model artifacts from all the container nodes.

        Used after training completes to gather the data from all the
        individual containers. As the official SageMaker Training Service, it
        will override duplicate files if multiple containers have the same file
        names.

        Args:
            compose_data (dict): Docker-Compose configuration in dictionary
                format.
            output_data_config: The configuration of the output data.
            job_name: The name of the job.

        Returns: Local path to the collected model artifacts.
        """
        # We need a directory to store the artfiacts from all the nodes
        # and another one to contained the compressed final artifacts
        artifacts = os.path.join(self.container_root, "artifacts")
        compressed_artifacts = os.path.join(self.container_root, "compressed_artifacts")
        os.mkdir(artifacts)

        model_artifacts = os.path.join(artifacts, "model")
        output_artifacts = os.path.join(artifacts, "output")

        artifact_dirs = [model_artifacts, output_artifacts, compressed_artifacts]
        for d in artifact_dirs:
            os.mkdir(d)

        # Gather the artifacts from all nodes into artifacts/model and artifacts/output
        for host in self.hosts:
            volumes = compose_data["services"][str(host)]["volumes"]
            volumes = [v[:-2] if v.endswith(":z") else v for v in volumes]
            for volume in volumes:
                if re.search(r"^[A-Za-z]:", volume):
                    unit, host_dir, container_dir = volume.split(":")
                    host_dir = unit + ":" + host_dir
                else:
                    host_dir, container_dir = volume.split(":")
                if container_dir == "/opt/ml/model":
                    sagemaker.local.utils.recursive_copy(host_dir, model_artifacts)
                elif container_dir == "/opt/ml/output":
                    sagemaker.local.utils.recursive_copy(host_dir, output_artifacts)

        # Tar Artifacts -> model.tar.gz and output.tar.gz
        model_files = [os.path.join(model_artifacts, name) for name in os.listdir(model_artifacts)]
        output_files = [
            os.path.join(output_artifacts, name) for name in os.listdir(output_artifacts)
        ]
        sagemaker.utils.create_tar_file(
            model_files, os.path.join(compressed_artifacts, "model.tar.gz")
        )
        sagemaker.utils.create_tar_file(
            output_files, os.path.join(compressed_artifacts, "output.tar.gz")
        )

        if output_data_config["S3OutputPath"] == "":
            output_data = "file://%s" % compressed_artifacts
        else:
            # Now we just need to move the compressed artifacts to wherever they are required
            output_data = sagemaker.local.utils.move_to_destination(
                compressed_artifacts,
                output_data_config["S3OutputPath"],
                job_name,
                self.sagemaker_session,
            )

        _delete_tree(model_artifacts)
        _delete_tree(output_artifacts)

        return os.path.join(output_data, "model.tar.gz")

    def write_processing_config_files(
        self,
        host,
        environment,
        processing_inputs,
        processing_output_config,
        processing_job_name,
    ):
        """Write the config files for the processing containers.

        This method writes the hyperparameters, resources and input data
        configuration files.

        Args:
            host (str): Host to write the configuration for
            environment (dict): Environment variable collection.
            processing_inputs (dict): Processing inputs.
            processing_output_config (dict): Processing output configuration.
            processing_job_name (str): Processing job name.
        """
        config_path = os.path.join(self.container_root, host, "config")

        resource_config = {"current_host": host, "hosts": self.hosts}
        _write_json_file(os.path.join(config_path, "resourceconfig.json"), resource_config)

        processing_job_config = {
            "ProcessingJobArn": processing_job_name,
            "ProcessingJobName": processing_job_name,
            "AppSpecification": {
                "ImageUri": self.image,
                "ContainerEntrypoint": self.container_entrypoint,
                "ContainerArguments": self.container_arguments,
            },
            "Environment": environment,
            "ProcessingInputs": processing_inputs,
            "ProcessingOutputConfig": processing_output_config,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": self.instance_count,
                    "InstanceType": self.instance_type,
                    "VolumeSizeInGB": 30,
                    "VolumeKmsKeyId": None,
                }
            },
            "RoleArn": "<no_role>",
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        }

        _write_json_file(
            os.path.join(config_path, "processingjobconfig.json"), processing_job_config
        )

    def write_config_files(self, host, hyperparameters, input_data_config):
        """Write the config files for the training containers.

        This method writes the hyperparameters, resources and input data
        configuration files.

        Returns: None

        Args:
            host (str): Host to write the configuration for
            hyperparameters (dict): Hyperparameters for training.
            input_data_config (dict): Training input channels to be used for
                training.
        """
        config_path = os.path.join(self.container_root, host, "input", "config")

        resource_config = {"current_host": host, "hosts": self.hosts}

        json_input_data_config = {}
        for c in input_data_config:
            channel_name = c["ChannelName"]
            json_input_data_config[channel_name] = {"TrainingInputMode": "File"}
            if "ContentType" in c:
                json_input_data_config[channel_name]["ContentType"] = c["ContentType"]

        _write_json_file(os.path.join(config_path, "hyperparameters.json"), hyperparameters)
        _write_json_file(os.path.join(config_path, "resourceconfig.json"), resource_config)
        _write_json_file(os.path.join(config_path, "inputdataconfig.json"), json_input_data_config)

    def _prepare_training_volumes(
        self, data_dir, input_data_config, output_data_config, hyperparameters
    ):
        """Prepares the training volumes based on input and output data configs.

        Args:
            data_dir:
            input_data_config:
            output_data_config:
            hyperparameters:
        """
        shared_dir = os.path.join(self.container_root, "shared")
        model_dir = os.path.join(self.container_root, "model")
        volumes = []

        volumes.append(_Volume(model_dir, "/opt/ml/model"))

        # Mount the metadata directory if present.
        # Only expected to be present on SM notebook instances.
        # This is used by some DeepEngine libraries
        metadata_dir = "/opt/ml/metadata"
        if os.path.isdir(metadata_dir):
            volumes.append(_Volume(metadata_dir, metadata_dir))

        # Set up the channels for the containers. For local data we will
        # mount the local directory to the container. For S3 Data we will download the S3 data
        # first.
        for channel in input_data_config:
            uri = channel["DataUri"]
            channel_name = channel["ChannelName"]
            channel_dir = os.path.join(data_dir, channel_name)
            os.mkdir(channel_dir)

            data_source = sagemaker.local.data.get_data_source_instance(uri, self.sagemaker_session)
            volumes.append(_Volume(data_source.get_root_dir(), channel=channel_name))

        # If there is a training script directory and it is a local directory,
        # mount it to the container.
        if sagemaker.estimator.DIR_PARAM_NAME in hyperparameters:
            training_dir = json.loads(hyperparameters[sagemaker.estimator.DIR_PARAM_NAME])
            parsed_uri = urlparse(training_dir)
            if parsed_uri.scheme == "file":
                host_dir = os.path.abspath(parsed_uri.netloc + parsed_uri.path)
                volumes.append(_Volume(host_dir, "/opt/ml/code"))
                # Also mount a directory that all the containers can access.
                volumes.append(_Volume(shared_dir, "/opt/ml/shared"))

        parsed_uri = urlparse(output_data_config["S3OutputPath"])
        if (
            parsed_uri.scheme == "file"
            and sagemaker.model.SAGEMAKER_OUTPUT_LOCATION in hyperparameters
        ):
            dir_path = os.path.abspath(parsed_uri.netloc + parsed_uri.path)
            intermediate_dir = os.path.join(dir_path, "output", "intermediate")
            if not os.path.exists(intermediate_dir):
                os.makedirs(intermediate_dir)
            volumes.append(_Volume(intermediate_dir, "/opt/ml/output/intermediate"))

        return volumes

    def _prepare_processing_volumes(self, data_dir, processing_inputs, processing_output_config):
        """Prepares local container volumes for the processing job.

        Args:
            data_dir: The local data directory.
            processing_inputs: The configuration of processing inputs.
            processing_output_config: The configuration of processing outputs.

        Returns:
            The volumes configuration.
        """
        shared_dir = os.path.join(self.container_root, "shared")
        volumes = []

        # Set up the input/outputs for the container.

        for item in processing_inputs:
            uri = item["DataUri"]
            input_container_dir = item["S3Input"]["LocalPath"]

            data_source = sagemaker.local.data.get_data_source_instance(uri, self.sagemaker_session)
            volumes.append(_Volume(data_source.get_root_dir(), input_container_dir))

        if processing_output_config and "Outputs" in processing_output_config:
            for item in processing_output_config["Outputs"]:
                output_name = item["OutputName"]
                output_container_dir = item["S3Output"]["LocalPath"]

                output_dir = os.path.join(data_dir, "output", output_name)
                os.makedirs(output_dir)

                volumes.append(_Volume(output_dir, output_container_dir))

        volumes.append(_Volume(shared_dir, "/opt/ml/shared"))

        return volumes

    def _upload_processing_outputs(self, data_dir, processing_output_config):
        """Uploads processing outputs to Amazon S3.

        Args:
            data_dir: The local data directory.
            processing_output_config: The processing output configuration.
        """
        if processing_output_config and "Outputs" in processing_output_config:
            for item in processing_output_config["Outputs"]:
                output_name = item["OutputName"]
                output_s3_uri = item["S3Output"]["S3Uri"]
                output_dir = os.path.join(data_dir, "output", output_name)

                sagemaker.local.utils.move_to_destination(
                    output_dir, output_s3_uri, "", self.sagemaker_session
                )

    def _update_local_src_path(self, params, key):
        """Updates the local path of source code.

        Args:
            params: Existing configuration parameters.
            key: Lookup key for the path of the source code in the configuration parameters.

        Returns:
            The updated parameters.
        """
        if key in params:
            src_dir = json.loads(params[key])
            parsed_uri = urlparse(src_dir)
            if parsed_uri.scheme == "file":
                new_params = params.copy()
                new_params[key] = json.dumps("/opt/ml/code")
                return new_params
        return params

    def _prepare_serving_volumes(self, model_location):
        """Prepares the serving volumes.

        Args:
            model_location: Location of the models.
        """
        volumes = []
        host = self.hosts[0]
        # Make the model available to the container. If this is a local file just mount it to
        # the container as a volume. If it is an S3 location, the DataSource will download it, we
        # just need to extract the tar file.
        host_dir = os.path.join(self.container_root, host)
        os.makedirs(host_dir)

        model_data_source = sagemaker.local.data.get_data_source_instance(
            model_location, self.sagemaker_session
        )

        for filename in model_data_source.get_file_list():
            if tarfile.is_tarfile(filename):
                with tarfile.open(filename) as tar:
                    tar.extractall(path=model_data_source.get_root_dir())

        volumes.append(_Volume(model_data_source.get_root_dir(), "/opt/ml/model"))

        return volumes

    def _generate_compose_file(self, command, additional_volumes=None, additional_env_vars=None):
        """Writes a config file describing a training/hosting environment.

        This method generates a docker compose configuration file, it has an
        entry for each container that will be created (based on self.hosts). it
        calls
        :meth:~sagemaker.local_session.SageMakerContainer._create_docker_host to
        generate the config for each individual container.

        Args:
            command (str): either 'train' or 'serve'
            additional_volumes (list): a list of volumes that will be mapped to
                the containers
            additional_env_vars (dict): a dictionary with additional environment
                variables to be passed on to the containers.

        Returns: (dict) A dictionary representation of the configuration that was written.
        """
        boto_session = self.sagemaker_session.boto_session
        additional_volumes = additional_volumes or []
        additional_env_vars = additional_env_vars or {}
        environment = []
        optml_dirs = set()

        aws_creds = _aws_credentials(boto_session)
        if aws_creds is not None:
            environment.extend(aws_creds)

        additional_env_var_list = ["{}={}".format(k, v) for k, v in additional_env_vars.items()]
        environment.extend(additional_env_var_list)

        if os.environ.get(DOCKER_COMPOSE_HTTP_TIMEOUT_ENV) is None:
            os.environ[DOCKER_COMPOSE_HTTP_TIMEOUT_ENV] = DOCKER_COMPOSE_HTTP_TIMEOUT

        if command == "train":
            optml_dirs = {"output", "output/data", "input"}
        elif command == "process":
            optml_dirs = {"output", "config"}

        services = {
            h: self._create_docker_host(h, environment, optml_dirs, command, additional_volumes)
            for h in self.hosts
        }

        content = {
            # Use version 2.3 as a minimum so that we can specify the runtime
            "version": "2.3",
            "services": services,
            "networks": {"sagemaker-local": {"name": "sagemaker-local"}},
        }

        docker_compose_path = os.path.join(self.container_root, DOCKER_COMPOSE_FILENAME)

        try:
            import yaml
        except ImportError as e:
            logger.error(sagemaker.utils._module_import_error("yaml", "Local mode", "local"))
            raise e

        yaml_content = yaml.dump(content, default_flow_style=False)
        # Mask all environment vars for logging, could contain secrects.
        masked_content = copy.deepcopy(content)
        for _, service_data in masked_content["services"].items():
            service_data["environment"] = ["[Masked]" for _ in service_data["environment"]]

        masked_content_for_logging = yaml.dump(masked_content, default_flow_style=False)
        logger.info("docker compose file: \n%s", masked_content_for_logging)
        with open(docker_compose_path, "w") as f:
            f.write(yaml_content)

        return content

    def _compose(self, detached=False):
        """Invokes the docker compose command.

        Args:
            detached:
        """
        compose_cmd = "docker-compose"

        command = [
            compose_cmd,
            "-f",
            os.path.join(self.container_root, DOCKER_COMPOSE_FILENAME),
            "up",
            "--build",
            "--abort-on-container-exit" if not detached else "--detach",  # mutually exclusive
        ]

        logger.info("docker command: %s", " ".join(command))
        return command

    def _create_docker_host(self, host, environment, optml_subdirs, command, volumes):
        """Creates the docker host configuration.

        Args:
            host:
            environment:
            optml_subdirs:
            command:
            volumes:
        """
        optml_volumes = self._build_optml_volumes(host, optml_subdirs)
        optml_volumes.extend(volumes)

        container_name_prefix = "".join(
            random.choice(string.ascii_lowercase + string.digits) for _ in range(10)
        )

        host_config = {
            "image": self.image,
            "container_name": f"{container_name_prefix}-{host}",
            "stdin_open": True,
            "tty": True,
            "volumes": [v.map for v in optml_volumes],
            "environment": environment,
            "networks": {"sagemaker-local": {"aliases": [host]}},
        }

        if command != "process":
            host_config["command"] = command
        else:
            if self.container_entrypoint:
                host_config["entrypoint"] = self.container_entrypoint
            if self.container_arguments:
                host_config["entrypoint"] = host_config["entrypoint"] + self.container_arguments

        # for GPU support pass in nvidia as the runtime, this is equivalent
        # to setting --runtime=nvidia in the docker commandline.
        if self.instance_type == "local_gpu":
            host_config["deploy"] = {
                "resources": {"reservations": {"devices": [{"capabilities": ["gpu"]}]}}
            }

        if command == "serve":
            serving_port = (
                sagemaker.utils.get_config_value(
                    "local.serving_port", self.sagemaker_session.config
                )
                or 8080
            )
            host_config.update({"ports": ["%s:8080" % serving_port]})

        return host_config

    def _create_tmp_folder(self):
        """Placeholder docstring"""
        root_dir = sagemaker.utils.get_config_value(
            "local.container_root", self.sagemaker_session.config
        )
        if root_dir:
            root_dir = os.path.abspath(root_dir)

        working_dir = tempfile.mkdtemp(dir=root_dir)

        # Docker cannot mount Mac OS /var folder properly see
        # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
        # Only apply this workaround if the user didn't provide an alternate storage root dir.
        if root_dir is None and platform.system() == "Darwin":
            working_dir = "/private{}".format(working_dir)

        return os.path.abspath(working_dir)

    def _build_optml_volumes(self, host, subdirs):
        """Generate a list of :class:`~sagemaker.local_session.Volume`.

        These are required for the container to start. It takes a folder with
        the necessary files for training and creates a list of opt volumes
        that the Container needs to start.

        Args:
            host (str): container for which the volumes will be generated.
            subdirs (list): list of subdirectories that will be mapped. For
                example: ['input', 'output', 'model']

        Returns: (list) List of :class:`~sagemaker.local_session.Volume`
        """
        volumes = []

        for subdir in subdirs:
            host_dir = os.path.join(self.container_root, host, subdir)
            container_dir = "/opt/ml/{}".format(subdir)
            volume = _Volume(host_dir, container_dir)
            volumes.append(volume)

        return volumes

    def _cleanup(self, dirs_to_delete=None):
        """Cleans up directories and the like.

        Args:
            dirs_to_delete:
        """
        if dirs_to_delete:
            for d in dirs_to_delete:
                _delete_tree(d)

        # Free the container config files.
        for host in self.hosts:
            container_config_path = os.path.join(self.container_root, host)
            _delete_tree(container_config_path)


class _HostingContainer(Thread):
    """Placeholder docstring."""

    def __init__(self, command):
        """Creates a new threaded hosting container.

        Args:
            command:
        """
        Thread.__init__(self)
        self.command = command
        self.process = None

    def run(self):
        """Placeholder docstring"""
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            _stream_output(self.process)
        except RuntimeError as e:
            # _stream_output() doesn't have the command line. We will handle the exception
            # which contains the exit code and append the command line to it.
            msg = "Failed to run: %s, %s" % (self.command, str(e))
            raise RuntimeError(msg)

    def down(self):
        """Placeholder docstring"""
        if os.name != "nt":
            sagemaker.local.utils.kill_child_processes(self.process.pid)
        self.process.terminate()


class _Volume(object):
    """Represent a Volume that will be mapped to a container."""

    def __init__(self, host_dir, container_dir=None, channel=None):
        """Create a Volume instance.

        The container path can be provided as a container_dir or as a channel name but not both.

        Args:
            host_dir (str): path to the volume data in the host
            container_dir (str): path inside the container that host_dir will be mapped to
            channel (str): channel name that the host_dir represents. It will be mapped as
                /opt/ml/input/data/<channel> in the container.
        """
        if not container_dir and not channel:
            raise ValueError("Either container_dir or channel must be declared.")

        if container_dir and channel:
            raise ValueError("container_dir and channel cannot be declared together.")

        self.container_dir = container_dir if container_dir else "/opt/ml/input/data/" + channel
        self.host_dir = host_dir
        map_format = "{}:{}"
        if platform.system() == "Linux" and SELINUX_ENABLED:
            # Support mounting shared volumes in SELinux enabled hosts
            map_format += ":z"
        if platform.system() == "Darwin" and host_dir.startswith("/var"):
            self.host_dir = os.path.join("/private", host_dir)

        self.map = map_format.format(self.host_dir, self.container_dir)


def _stream_output(process):
    """Stream the output of a process to stdout

    This function takes an existing process that will be polled for output.
    Only stdout will be polled and sent to sys.stdout.

    Args:
        process (subprocess.Popen): a process that has been started with
            stdout=PIPE and stderr=STDOUT

    Returns (int): process exit code
    """
    exit_code = None

    while exit_code is None:
        stdout = process.stdout.readline().decode("utf-8")
        sys.stdout.write(stdout)
        exit_code = process.poll()

    if exit_code != 0:
        raise RuntimeError("Process exited with code: %s" % exit_code)

    return exit_code


def _check_output(cmd, *popenargs, **kwargs):
    """Makes a call to `subprocess.check_output` for the given command and args.

    Args:
        cmd:
        *popenargs:
        **kwargs:
    """
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    success = True
    try:
        output = subprocess.check_output(cmd, *popenargs, **kwargs)
    except subprocess.CalledProcessError as e:
        output = e.output
        success = False

    output = output.decode("utf-8")
    if not success:
        logger.error("Command output: %s", output)
        raise Exception("Failed to run %s" % ",".join(cmd))

    return output


def _create_processing_config_file_directories(root, host):
    """Creates the directory for the processing config files.

    Args:
        root: The root path.
        host: The current host.
    """
    for d in ["config"]:
        os.makedirs(os.path.join(root, host, d))


def _create_config_file_directories(root, host):
    """Creates the directories for the config files.

    Args:
        root:
        host:
    """
    for d in ["input", "input/config", "output", "model"]:
        os.makedirs(os.path.join(root, host, d))


def _delete_tree(path):
    """Makes a call to `shutil.rmtree` for the given path.

    Args:
        path:
    """
    try:
        shutil.rmtree(path)
    except OSError as exc:
        # on Linux, when docker writes to any mounted volume, it uses the container's user. In most
        # cases this is root. When the container exits and we try to delete them we can't because
        # root owns those files. We expect this to happen, so we handle EACCESS. Any other error
        # we will raise the exception up.
        if exc.errno == errno.EACCES:
            logger.warning("Failed to delete: %s Please remove it manually.", path)
        else:
            logger.error("Failed to delete: %s", path)
            raise


def _aws_credentials(session):
    """Provides the AWS credentials of the session as a paired list of strings.

    These can be used to set environment variables on command execution.

    Args:
        session:
    """
    try:
        creds = session.get_credentials()
        access_key = creds.access_key
        secret_key = creds.secret_key
        token = creds.token

        # The presence of a token indicates the credentials are short-lived and as such are risky
        # to be used as they might expire while running.
        # Long-lived credentials are available either through
        # 1. boto session
        # 2. EC2 Metadata Service (SageMaker Notebook instances or EC2 instances with roles
        #       attached them)
        # Short-lived credentials available via boto session are permitted to support running on
        # machines with no EC2 Metadata Service but a warning is provided about their danger
        if token is None:
            logger.info("Using the long-lived AWS credentials found in session")
            return [
                "AWS_ACCESS_KEY_ID=%s" % (str(access_key)),
                "AWS_SECRET_ACCESS_KEY=%s" % (str(secret_key)),
            ]
        if _use_short_lived_credentials() or not _aws_credentials_available_in_metadata_service():
            logger.warning(
                "Using the short-lived AWS credentials found in session. They might expire while "
                "running."
            )
            return [
                "AWS_ACCESS_KEY_ID=%s" % (str(access_key)),
                "AWS_SECRET_ACCESS_KEY=%s" % (str(secret_key)),
                "AWS_SESSION_TOKEN=%s" % (str(token)),
            ]
        logger.info(
            "No AWS credentials found in session but credentials from EC2 Metadata Service are "
            "available."
        )
        return None
    except Exception as e:  # pylint: disable=broad-except
        logger.info("Could not get AWS credentials: %s", e)

    return None


def _aws_credentials_available_in_metadata_service():
    """Placeholder docstring"""
    import botocore
    from botocore.credentials import InstanceMetadataProvider
    from botocore.utils import InstanceMetadataFetcher

    session = botocore.session.Session()
    instance_metadata_provider = InstanceMetadataProvider(
        iam_role_fetcher=InstanceMetadataFetcher(
            timeout=session.get_config_variable("metadata_service_timeout"),
            num_attempts=session.get_config_variable("metadata_service_num_attempts"),
            user_agent=session.user_agent(),
        )
    )
    return not instance_metadata_provider.load() is None


def _use_short_lived_credentials():
    """Use short-lived AWS credentials found in session."""
    return os.environ.get("USE_SHORT_LIVED_CREDENTIALS") == "1"


def _write_json_file(filename, content):
    """Write the contents dict as json to the file.

    Args:
        filename:
        content:
    """
    with open(filename, "w") as f:
        json.dump(content, f)


def _ecr_login_if_needed(boto_session, image):
    """Log into ECR, if needed.

    Of note, only ECR images need login.

    Args:
        boto_session:
        image:
    """
    sagemaker_pattern = re.compile(sagemaker.utils.ECR_URI_PATTERN)
    sagemaker_match = sagemaker_pattern.match(image)
    if not sagemaker_match:
        return False

    # do we have the image?
    if _check_output("docker images -q %s" % image).strip():
        return False

    if not boto_session:
        raise RuntimeError(
            "A boto session is required to login to ECR."
            "Please pull the image: %s manually." % image
        )

    ecr = boto_session.client("ecr")
    auth = ecr.get_authorization_token(registryIds=[image.split(".")[0]])
    authorization_data = auth["authorizationData"][0]

    raw_token = base64.b64decode(authorization_data["authorizationToken"])
    token = raw_token.decode("utf-8").strip("AWS:")
    ecr_url = auth["authorizationData"][0]["proxyEndpoint"]

    # Log in to ecr, but use communicate to not print creds to the console
    cmd = f"docker login {ecr_url} -u AWS --password-stdin".split()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
    )

    proc.communicate(input=token.encode())

    return True


def _pull_image(image):
    """Invokes the docker pull command for the given image.

    Args:
        image:
    """
    pull_image_command = ("docker pull %s" % image).strip()
    logger.info("docker command: %s", pull_image_command)

    subprocess.check_output(pull_image_command.split())
    logger.info("image pulled: %s", image)
