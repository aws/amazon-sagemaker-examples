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
"""This module is the entry to run spark processing script.

This module contains code related to Spark Processors, which are used
for Processing jobs. These jobs let customers perform data pre-processing,
post-processing, feature engineering, data validation, and model evaluation
on SageMaker using Spark and PySpark.
"""
from __future__ import absolute_import

import json
import logging
import os.path
import shutil
import subprocess
import tempfile
import time
import urllib.request
from enum import Enum
from io import BytesIO
from urllib.parse import urlparse
from copy import copy

from typing import Union, List, Dict, Optional

from sagemaker import image_uris, s3
from sagemaker.local.image import _ecr_login_if_needed, _pull_image
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.s3 import S3Uploader
from sagemaker.session import Session
from sagemaker.network import NetworkConfig
from sagemaker.spark import defaults

from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.pipeline_context import runnable_by_pipeline
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.functions import Join

logger = logging.getLogger(__name__)


class _SparkProcessorBase(ScriptProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using Spark.

    Base class for either PySpark or SparkJars.
    """

    _default_command = "smspark-submit"
    _conf_container_base_path = "/opt/ml/processing/input/"
    _conf_container_input_name = "conf"
    _conf_file_name = "configuration.json"

    _submit_jars_input_channel_name = "jars"
    _submit_files_input_channel_name = "files"
    _submit_py_files_input_channel_name = "py-files"
    _submit_deps_error_message = (
        "Please specify a list of one or more S3 URIs, "
        "local file paths, and/or local directory paths"
    )

    # history server vars
    _history_server_port = "15050"
    _history_server_url_suffix = f"/proxy/{_history_server_port}"
    _spark_event_log_default_local_path = "/opt/ml/processing/spark-events/"

    def __init__(
        self,
        role=None,
        instance_type=None,
        instance_count=None,
        framework_version=None,
        py_version=None,
        container_version=None,
        image_uri=None,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        configuration_location: Optional[str] = None,
        dependency_location: Optional[str] = None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initialize a ``_SparkProcessorBase`` instance.

        The _SparkProcessorBase handles Amazon SageMaker processing tasks for
        jobs using SageMaker Spark.

        Args:
            framework_version (str): The version of SageMaker PySpark.
            py_version (str): The version of python.
            container_version (str): The version of spark container.
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3 (default: None).
                If not specified, the value from the defaults configuration file
                will be used.
            instance_type (str): Type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            instance_count (int): The number of instances to run
                the Processing job with. Defaults to 1.
            volume_size_in_gb (int): Size in GB of the EBS volume to
                use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume.
            output_kms_key (str): The KMS key id for all ProcessingOutputs.
            configuration_location (str): The S3 prefix URI where the user-provided EMR
                application configuration will be uploaded (default: None). If not specified,
                the default ``configuration location`` is 's3://{sagemaker-default-bucket}'.
            dependency_location (str): The S3 prefix URI where Spark dependencies will be
                uploaded (default: None). If not specified, the default ``dependency location``
                is 's3://{sagemaker-default-bucket}'.
            max_runtime_in_seconds (int): Timeout in seconds.
                After this amount of time Amazon SageMaker terminates the job
                regardless of its current status.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon
                SageMaker APIs and any other AWS services needed. If not specified,
                the processor creates one using the default AWS configuration chain.
            env (dict): Environment variables to be passed to the processing job.
            tags ([dict]): List of tags to be passed to the processing job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        self.configuration_location = configuration_location
        self.dependency_location = dependency_location
        self.history_server = None
        self._spark_event_logs_s3_uri = None

        session = sagemaker_session or Session()
        region = session.boto_region_name

        self.image_uri = self._retrieve_image_uri(
            image_uri, framework_version, py_version, container_version, region, instance_type
        )

        env = env or {}
        command = [_SparkProcessorBase._default_command]

        super(_SparkProcessorBase, self).__init__(
            role=role,
            image_uri=self.image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            command=command,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=session,
            env=env,
            tags=tags,
            network_config=network_config,
        )

    def get_run_args(
        self,
        code,
        inputs=None,
        outputs=None,
        arguments=None,
    ):
        """Returns a RunArgs object.

        For processors (:class:`~sagemaker.spark.processing.PySparkProcessor`,
            :class:`~sagemaker.spark.processing.SparkJar`) that have special
            run() arguments, this object contains the normalized arguments for passing to
            :class:`~sagemaker.workflow.steps.ProcessingStep`.

        Args:
            code (str): This can be an S3 URI or a local path to a file with the framework
                script to run.
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str]): A list of string arguments to be passed to a
                processing job (default: None).
        """
        return super().get_run_args(
            code=code,
            inputs=inputs,
            outputs=outputs,
            arguments=arguments,
        )

    @runnable_by_pipeline
    def run(
        self,
        submit_app,
        inputs=None,
        outputs=None,
        arguments=None,
        wait=True,
        logs=True,
        job_name=None,
        experiment_config=None,
        kms_key=None,
    ):
        """Runs a processing job.

        Args:
            submit_app (str): .py or .jar file to submit to Spark as the primary application
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str]): A list of string arguments to be passed to a
                processing job (default: None).
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the base job name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        if is_pipeline_variable(submit_app):
            raise ValueError(
                "submit_app argument has to be a valid S3 URI or local file path "
                + "rather than a pipeline variable"
            )

        return super().run(
            submit_app,
            inputs,
            outputs,
            arguments,
            wait,
            logs,
            job_name,
            experiment_config,
            kms_key,
        )

    def _extend_processing_args(self, inputs, outputs, **kwargs):
        """Extends processing job args such as inputs."""

        # make a shallow copy of user outputs
        outputs = outputs or []
        extended_outputs = copy(outputs)

        if kwargs.get("spark_event_logs_s3_uri"):
            spark_event_logs_s3_uri = kwargs.get("spark_event_logs_s3_uri")
            SparkConfigUtils.validate_s3_uri(spark_event_logs_s3_uri)

            self._spark_event_logs_s3_uri = spark_event_logs_s3_uri
            self.command.extend(
                [
                    "--local-spark-event-logs-dir",
                    _SparkProcessorBase._spark_event_log_default_local_path,
                ]
            )

            output = ProcessingOutput(
                source=_SparkProcessorBase._spark_event_log_default_local_path,
                destination=spark_event_logs_s3_uri,
                s3_upload_mode="Continuous",
            )

            extended_outputs.append(output)

        # make a shallow copy of user inputs
        inputs = inputs or []
        extended_inputs = copy(inputs)

        if kwargs.get("configuration"):
            configuration = kwargs.get("configuration")
            SparkConfigUtils.validate_configuration(configuration)
            extended_inputs.append(self._stage_configuration(configuration))

        return (
            extended_inputs if extended_inputs else None,
            extended_outputs if extended_outputs else None,
        )

    def start_history_server(self, spark_event_logs_s3_uri=None):
        """Starts a Spark history server.

        Args:
            spark_event_logs_s3_uri (str): S3 URI where Spark events are stored.
        """
        if _ecr_login_if_needed(self.sagemaker_session.boto_session, self.image_uri):
            logger.info("Pulling spark history server image...")
            _pull_image(self.image_uri)
        history_server_env_variables = self._prepare_history_server_env_variables(
            spark_event_logs_s3_uri
        )
        self.history_server = _HistoryServer(
            history_server_env_variables, self.image_uri, self._get_network_config()
        )
        self.history_server.run()
        self._check_history_server()

    def terminate_history_server(self):
        """Terminates the Spark history server."""
        if self.history_server:
            logger.info("History server is running, terminating history server")
            self.history_server.down()
            self.history_server = None

    def _retrieve_image_uri(
        self, image_uri, framework_version, py_version, container_version, region, instance_type
    ):
        """Builds an image URI."""
        if not image_uri:
            if (py_version is None) != (container_version is None):
                raise ValueError(
                    "Both or neither of py_version and container_version should be set"
                )

            if container_version:
                container_version = f"v{container_version}"

            return image_uris.retrieve(
                defaults.SPARK_NAME,
                region,
                version=framework_version,
                instance_type=instance_type,
                py_version=py_version,
                container_version=container_version,
            )

        return image_uri

    def _stage_configuration(self, configuration):
        """Serializes and uploads the user-provided EMR application configuration to S3.

        This method prepares an input channel.

        Args:
            configuration (Dict): the configuration dict for the EMR application configuration.
        """
        from sagemaker.workflow.utilities import _pipeline_config

        if self.configuration_location:
            if self.configuration_location.endswith("/"):
                s3_prefix_uri = self.configuration_location[:-1]
            else:
                s3_prefix_uri = self.configuration_location
        else:
            s3_prefix_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
            )

        serialized_configuration = BytesIO(json.dumps(configuration).encode("utf-8"))

        if _pipeline_config and _pipeline_config.config_hash:
            s3_uri = (
                f"{s3_prefix_uri}/{_pipeline_config.pipeline_name}/{_pipeline_config.step_name}/"
                f"input/{self._conf_container_input_name}/{_pipeline_config.config_hash}/"
                f"{self._conf_file_name}"
            )
        else:
            s3_uri = (
                f"{s3_prefix_uri}/{self._current_job_name}/"
                f"input/{self._conf_container_input_name}/"
                f"{self._conf_file_name}"
            )

        S3Uploader.upload_string_as_file_body(
            body=serialized_configuration,
            desired_s3_uri=s3_uri,
            sagemaker_session=self.sagemaker_session,
        )

        conf_input = ProcessingInput(
            source=s3_uri,
            destination=f"{self._conf_container_base_path}{self._conf_container_input_name}",
            input_name=_SparkProcessorBase._conf_container_input_name,
        )
        return conf_input

    def _stage_submit_deps(self, submit_deps, input_channel_name):
        """Prepares a list of paths to jars, py-files, or files dependencies.

        This prepared list of paths is provided as `spark-submit` options.
        The submit_deps list may include a combination of S3 URIs and local paths.
        Any S3 URIs are appended to the `spark-submit` option value without modification.
        Any local file paths are copied to a temp directory, uploaded to ``dependency location``,
        and included as a ProcessingInput channel to provide as local files to the SageMaker
        Spark container.

        :param submit_deps (list[str]): List of one or more dependency paths to include.
        :param input_channel_name (str): The `spark-submit` option name associated with
                    the input channel.
        :return (Optional[ProcessingInput], str): Tuple of (left) optional ProcessingInput
                    for the input channel, and (right) comma-delimited value for
                    `spark-submit` option.
        """
        if not submit_deps:
            raise ValueError(
                f"submit_deps value may not be empty. {self._submit_deps_error_message}"
            )
        if not input_channel_name:
            raise ValueError("input_channel_name value may not be empty.")

        use_input_channel = False
        spark_opt_s3_uris = []
        spark_opt_s3_uris_has_pipeline_var = False

        with tempfile.TemporaryDirectory() as tmpdir:
            for dep_path in submit_deps:
                if is_pipeline_variable(dep_path):
                    spark_opt_s3_uris.append(dep_path)
                    spark_opt_s3_uris_has_pipeline_var = True
                    continue
                dep_url = urlparse(dep_path)
                # S3 URIs are included as-is in the spark-submit argument
                if dep_url.scheme in ["s3", "s3a"]:
                    spark_opt_s3_uris.append(dep_path)
                # Local files are copied to temp directory to be uploaded to S3
                elif not dep_url.scheme or dep_url.scheme == "file":
                    if not os.path.isfile(dep_path):
                        raise ValueError(
                            f"submit_deps path {dep_path} is not a valid local file. "
                            f"{self._submit_deps_error_message}"
                        )
                    logger.info(
                        "Copying dependency from local path %s to tmpdir %s", dep_path, tmpdir
                    )
                    shutil.copy(dep_path, tmpdir)
                else:
                    raise ValueError(
                        f"submit_deps path {dep_path} references unsupported filesystem "
                        f"scheme: {dep_url.scheme} {self._submit_deps_error_message}"
                    )

            # If any local files were found and copied, upload the temp directory to S3
            if os.listdir(tmpdir):
                from sagemaker.workflow.utilities import _pipeline_config

                if self.dependency_location:
                    if self.dependency_location.endswith("/"):
                        s3_prefix_uri = self.dependency_location[:-1]
                    else:
                        s3_prefix_uri = self.dependency_location
                else:
                    s3_prefix_uri = s3.s3_path_join(
                        "s3://",
                        self.sagemaker_session.default_bucket(),
                        self.sagemaker_session.default_bucket_prefix,
                    )

                if _pipeline_config and _pipeline_config.code_hash:
                    input_channel_s3_uri = (
                        f"{s3_prefix_uri}/{_pipeline_config.pipeline_name}/"
                        f"code/{_pipeline_config.code_hash}/{input_channel_name}"
                    )
                else:
                    input_channel_s3_uri = (
                        f"{s3_prefix_uri}/{self._current_job_name}/input/{input_channel_name}"
                    )
                logger.info(
                    "Uploading dependencies from tmpdir %s to S3 %s", tmpdir, input_channel_s3_uri
                )
                S3Uploader.upload(
                    local_path=tmpdir,
                    desired_s3_uri=input_channel_s3_uri,
                    sagemaker_session=self.sagemaker_session,
                )
                use_input_channel = True

        # If any local files were uploaded, construct a ProcessingInput to provide
        # them to the Spark container  and form the spark-submit option from a
        # combination of S3 URIs and container's local input path
        if use_input_channel:
            input_channel = ProcessingInput(
                source=input_channel_s3_uri,
                destination=f"{self._conf_container_base_path}{input_channel_name}",
                input_name=input_channel_name,
            )
            spark_opt = (
                Join(on=",", values=spark_opt_s3_uris + [input_channel.destination])
                if spark_opt_s3_uris_has_pipeline_var
                else ",".join(spark_opt_s3_uris + [input_channel.destination])
            )
        # If no local files were uploaded, form the spark-submit option from a list of S3 URIs
        else:
            input_channel = None
            spark_opt = (
                Join(on=",", values=spark_opt_s3_uris)
                if spark_opt_s3_uris_has_pipeline_var
                else ",".join(spark_opt_s3_uris)
            )

        return input_channel, spark_opt

    def _get_network_config(self):
        """Runs container with different network config based on different env."""
        if self._is_notebook_instance():
            return "--network host"

        return f"-p 80:80 -p {self._history_server_port}:{self._history_server_port}"

    def _prepare_history_server_env_variables(self, spark_event_logs_s3_uri):
        """Gets all parameters required to run history server."""
        # prepare env varibles
        history_server_env_variables = {}

        if spark_event_logs_s3_uri:
            history_server_env_variables[
                _HistoryServer.arg_event_logs_s3_uri
            ] = spark_event_logs_s3_uri
        # this variable will be previously set by run() method
        elif self._spark_event_logs_s3_uri is not None:
            history_server_env_variables[
                _HistoryServer.arg_event_logs_s3_uri
            ] = self._spark_event_logs_s3_uri
        else:
            raise ValueError(
                "SPARK_EVENT_LOGS_S3_URI not present. You can specify spark_event_logs_s3_uri "
                "either in run() or start_history_server()"
            )

        history_server_env_variables.update(self._config_aws_credentials())
        region = self.sagemaker_session.boto_region_name
        history_server_env_variables["AWS_REGION"] = region

        if self._is_notebook_instance():
            history_server_env_variables[
                _HistoryServer.arg_remote_domain_name
            ] = self._get_notebook_instance_domain()

        return history_server_env_variables

    def _is_notebook_instance(self):
        """Determine whether it is a notebook instance."""
        return os.path.isfile("/opt/ml/metadata/resource-metadata.json")

    def _get_notebook_instance_domain(self):
        """Get the instance's domain."""
        region = self.sagemaker_session.boto_region_name
        with open("/opt/ml/metadata/resource-metadata.json") as file:
            data = json.load(file)
            notebook_name = data["ResourceName"]

        return f"https://{notebook_name}.notebook.{region}.sagemaker.aws"

    def _check_history_server(self, ping_timeout=40):
        """Print message indicating the status of history server.

        Pings port 15050 to check whether the history server is up.
        Times out after `ping_timeout`.

        Args:
            ping_timeout (int): Timeout in seconds (defaults to 40).
        """
        # ping port 15050 to check history server is up
        timeout = time.time() + ping_timeout

        while True:
            if self._is_history_server_started():
                if self._is_notebook_instance():
                    logger.info(
                        "History server is up on %s%s",
                        self._get_notebook_instance_domain(),
                        self._history_server_url_suffix,
                    )
                else:
                    logger.info(
                        "History server is up on http://0.0.0.0%s", self._history_server_url_suffix
                    )
                break
            if time.time() > timeout:
                logger.error(
                    "History server failed to start. Please run 'docker logs history_server' "
                    "to see logs"
                )
                break

            time.sleep(1)

    def _is_history_server_started(self):
        """Check if history server is started."""
        try:
            response = urllib.request.urlopen(f"http://localhost:{self._history_server_port}")
            return response.status == 200
        except Exception:  # pylint: disable=W0703
            return False

    # TODO (guoqioa@): method only checks urlparse scheme, need to perform deep s3 validation
    def _validate_s3_uri(self, spark_output_s3_path):
        """Validate whether the URI uses an S3 scheme.

        In the future, this validation will perform deeper S3 validation.

        Args:
            spark_output_s3_path (str): The URI of the Spark output S3 Path.
        """
        if is_pipeline_variable(spark_output_s3_path):
            return

        if urlparse(spark_output_s3_path).scheme != "s3":
            raise ValueError(
                f"Invalid s3 path: {spark_output_s3_path}. Please enter something like "
                "s3://bucket-name/folder-name"
            )

    def _config_aws_credentials(self):
        """Configure AWS credentials."""
        try:
            creds = self.sagemaker_session.boto_session.get_credentials()
            access_key = creds.access_key
            secret_key = creds.secret_key
            token = creds.token

            return {
                "AWS_ACCESS_KEY_ID": str(access_key),
                "AWS_SECRET_ACCESS_KEY": str(secret_key),
                "AWS_SESSION_TOKEN": str(token),
            }
        except Exception as e:  # pylint: disable=W0703
            logger.info("Could not get AWS credentials: %s", e)
            return {}

    def _handle_script_dependencies(self, inputs, submit_files, file_type):
        """Handle script dependencies

        The method extends inputs and command based on input files and file_type
        """

        if not submit_files:
            return inputs

        input_channel_name_dict = {
            FileType.PYTHON: self._submit_py_files_input_channel_name,
            FileType.JAR: self._submit_jars_input_channel_name,
            FileType.FILE: self._submit_files_input_channel_name,
        }

        files_input, files_opt = self._stage_submit_deps(
            submit_files, input_channel_name_dict[file_type]
        )

        inputs = inputs or []

        if files_input:
            inputs.append(files_input)

        if files_opt:
            self.command.extend([f"--{input_channel_name_dict[file_type]}", files_opt])

        return inputs


class PySparkProcessor(_SparkProcessorBase):
    """Handles Amazon SageMaker processing tasks for jobs using PySpark."""

    def __init__(
        self,
        role: str = None,
        instance_type: Union[str, PipelineVariable] = None,
        instance_count: Union[int, PipelineVariable] = None,
        framework_version: Optional[str] = None,
        py_version: Optional[str] = None,
        container_version: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        configuration_location: Optional[str] = None,
        dependency_location: Optional[str] = None,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initialize an ``PySparkProcessor`` instance.

        The PySparkProcessor handles Amazon SageMaker processing tasks for jobs
        using SageMaker PySpark.

        Args:
            framework_version (str): The version of SageMaker PySpark.
            py_version (str): The version of python.
            container_version (str): The version of spark container.
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3 (default: None).
                If not specified, the value from the defaults configuration file
                will be used.
            instance_type (str or PipelineVariable): Type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            instance_count (int or PipelineVariable): The number of instances to run
                the Processing job with. Defaults to 1.
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume to
                use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing
                volume.
            output_kms_key (str or PipelineVariable): The KMS key id for all ProcessingOutputs.
            configuration_location (str): The S3 prefix URI where the user-provided EMR
                application configuration will be uploaded (default: None). If not specified,
                the default ``configuration location`` is 's3://{sagemaker-default-bucket}'.
            dependency_location (str): The S3 prefix URI where Spark dependencies will be
                uploaded (default: None). If not specified, the default ``dependency location``
                is 's3://{sagemaker-default-bucket}'.
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds.
                After this amount of time Amazon SageMaker terminates the job
                regardless of its current status.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the processor creates one
                using the default AWS configuration chain.
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables to
                be passed to the processing job.
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): List of tags to
                be passed to the processing job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """

        super(PySparkProcessor, self).__init__(
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            framework_version=framework_version,
            py_version=py_version,
            container_version=container_version,
            image_uri=image_uri,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            configuration_location=configuration_location,
            dependency_location=dependency_location,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )

    def get_run_args(
        self,
        submit_app,
        submit_py_files=None,
        submit_jars=None,
        submit_files=None,
        inputs=None,
        outputs=None,
        arguments=None,
        job_name=None,
        configuration=None,
        spark_event_logs_s3_uri=None,
    ):
        """Returns a RunArgs object.

        This object contains the normalized inputs, outputs and arguments
        needed when using a ``PySparkProcessor`` in a
        :class:`~sagemaker.workflow.steps.ProcessingStep`.

        Args:
            submit_app (str): Path (local or S3) to Python file to submit to Spark
                as the primary application. This is translated to the `code`
                property on the returned `RunArgs` object.
            submit_py_files (list[str]): List of paths (local or S3) to provide for
                `spark-submit --py-files` option
            submit_jars (list[str]): List of paths (local or S3) to provide for
                `spark-submit --jars` option
            submit_files (list[str]): List of paths (local or S3) to provide for
                `spark-submit --files` option
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str]): A list of string arguments to be passed to a
                processing job (default: None).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the base job name and current timestamp.
            configuration (list[dict] or dict): Configuration for Hadoop, Spark, or Hive.
                List or dictionary of EMR-style classifications.
                https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
            spark_event_logs_s3_uri (str): S3 path where spark application events will
                be published to.
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        if not submit_app:
            raise ValueError("submit_app is required")

        extended_inputs, extended_outputs = self._extend_processing_args(
            inputs=inputs,
            outputs=outputs,
            submit_py_files=submit_py_files,
            submit_jars=submit_jars,
            submit_files=submit_files,
            configuration=configuration,
            spark_event_logs_s3_uri=spark_event_logs_s3_uri,
        )

        return super().get_run_args(
            code=submit_app,
            inputs=extended_inputs,
            outputs=extended_outputs,
            arguments=arguments,
        )

    @runnable_by_pipeline
    def run(
        self,
        submit_app: str,
        submit_py_files: Optional[List[Union[str, PipelineVariable]]] = None,
        submit_jars: Optional[List[Union[str, PipelineVariable]]] = None,
        submit_files: Optional[List[Union[str, PipelineVariable]]] = None,
        inputs: Optional[List[ProcessingInput]] = None,
        outputs: Optional[List[ProcessingOutput]] = None,
        arguments: Optional[List[Union[str, PipelineVariable]]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        configuration: Optional[Union[List[Dict], Dict]] = None,
        spark_event_logs_s3_uri: Optional[Union[str, PipelineVariable]] = None,
        kms_key: Optional[str] = None,
    ):
        """Runs a processing job.

        Args:
            submit_app (str): Path (local or S3) to Python file to submit to Spark
                as the primary application
            submit_py_files (list[str] or list[PipelineVariable]): List of paths (local or S3)
                to provide for `spark-submit --py-files` option
            submit_jars (list[str] or list[PipelineVariable]): List of paths (local or S3)
                to provide for `spark-submit --jars` option
            submit_files (list[str] or list[PipelineVariable]): List of paths (local or S3)
                to provide for `spark-submit --files` option
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str] or list[PipelineVariable]): A list of string arguments to
                be passed to a processing job (default: None).
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the base job name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
            configuration (list[dict] or dict): Configuration for Hadoop, Spark, or Hive.
                List or dictionary of EMR-style classifications.
                https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
            spark_event_logs_s3_uri (str or PipelineVariable): S3 path where spark application
                events will be published to.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        if not submit_app:
            raise ValueError("submit_app is required")

        extended_inputs, extended_outputs = self._extend_processing_args(
            inputs=inputs,
            outputs=outputs,
            submit_py_files=submit_py_files,
            submit_jars=submit_jars,
            submit_files=submit_files,
            configuration=configuration,
            spark_event_logs_s3_uri=spark_event_logs_s3_uri,
        )

        return super().run(
            submit_app=submit_app,
            inputs=extended_inputs,
            outputs=extended_outputs,
            arguments=arguments,
            wait=wait,
            logs=logs,
            job_name=self._current_job_name,
            experiment_config=experiment_config,
            kms_key=kms_key,
        )

    def _extend_processing_args(self, inputs, outputs, **kwargs):
        """Extends inputs and outputs.

        Args:
            inputs: Processing inputs.
            outputs: Processing outputs.
            kwargs: Additional keyword arguments passed to `super()`.
        """

        if inputs is None:
            inputs = []

        # make a shallow copy of user inputs
        extended_inputs = copy(inputs)

        self.command = [_SparkProcessorBase._default_command]
        extended_inputs = self._handle_script_dependencies(
            extended_inputs, kwargs.get("submit_py_files"), FileType.PYTHON
        )
        extended_inputs = self._handle_script_dependencies(
            extended_inputs, kwargs.get("submit_jars"), FileType.JAR
        )
        extended_inputs = self._handle_script_dependencies(
            extended_inputs, kwargs.get("submit_files"), FileType.FILE
        )

        return super()._extend_processing_args(extended_inputs, outputs, **kwargs)


class SparkJarProcessor(_SparkProcessorBase):
    """Handles Amazon SageMaker processing tasks for jobs using Spark with Java or Scala Jars."""

    def __init__(
        self,
        role: str = None,
        instance_type: Union[str, PipelineVariable] = None,
        instance_count: Union[int, PipelineVariable] = None,
        framework_version: Optional[str] = None,
        py_version: Optional[str] = None,
        container_version: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        configuration_location: Optional[str] = None,
        dependency_location: Optional[str] = None,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initialize a ``SparkJarProcessor`` instance.

        The SparkProcessor handles Amazon SageMaker processing tasks for jobs
        using SageMaker Spark.

        Args:
            framework_version (str): The version of SageMaker PySpark.
            py_version (str): The version of python.
            container_version (str): The version of spark container.
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3 (default: None).
                If not specified, the value from the defaults configuration file
                will be used.
            instance_type (str or PipelineVariable): Type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            instance_count (int or PipelineVariable): The number of instances to run
                the Processing job with. Defaults to 1.
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume to
                use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing
                volume.
            output_kms_key (str or PipelineVariable): The KMS key id for all ProcessingOutputs.
            configuration_location (str): The S3 prefix URI where the user-provided EMR
                application configuration will be uploaded (default: None). If not specified,
                the default ``configuration location`` is 's3://{sagemaker-default-bucket}'.
            dependency_location (str): The S3 prefix URI where Spark dependencies will be
                uploaded (default: None). If not specified, the default ``dependency location``
                is 's3://{sagemaker-default-bucket}'.
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds.
                After this amount of time Amazon SageMaker terminates the job
                regardless of its current status.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the processor creates one
                using the default AWS configuration chain.
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables to
                be passed to the processing job.
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): List of tags to
                be passed to the processing job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """

        super(SparkJarProcessor, self).__init__(
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            framework_version=framework_version,
            py_version=py_version,
            container_version=container_version,
            image_uri=image_uri,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            configuration_location=configuration_location,
            dependency_location=dependency_location,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )

    def get_run_args(
        self,
        submit_app,
        submit_class=None,
        submit_jars=None,
        submit_files=None,
        inputs=None,
        outputs=None,
        arguments=None,
        job_name=None,
        configuration=None,
        spark_event_logs_s3_uri=None,
    ):
        """Returns a RunArgs object.

        This object contains the normalized inputs, outputs and arguments
        needed when using a ``SparkJarProcessor`` in a
        :class:`~sagemaker.workflow.steps.ProcessingStep`.

        Args:
            submit_app (str): Path (local or S3) to Python file to submit to Spark
                as the primary application. This is translated to the `code`
                property on the returned `RunArgs` object
            submit_class (str): Java class reference to submit to Spark as the primary
                application
            submit_jars (list[str]): List of paths (local or S3) to provide for
                `spark-submit --jars` option
            submit_files (list[str]): List of paths (local or S3) to provide for
                `spark-submit --files` option
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str]): A list of string arguments to be passed to a
                processing job (default: None).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the base job name and current timestamp.
            configuration (list[dict] or dict): Configuration for Hadoop, Spark, or Hive.
                List or dictionary of EMR-style classifications.
                https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
            spark_event_logs_s3_uri (str): S3 path where spark application events will
                be published to.
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        if not submit_app:
            raise ValueError("submit_app is required")

        extended_inputs, extended_outputs = self._extend_processing_args(
            inputs=inputs,
            outputs=outputs,
            submit_class=submit_class,
            submit_jars=submit_jars,
            submit_files=submit_files,
            configuration=configuration,
            spark_event_logs_s3_uri=spark_event_logs_s3_uri,
        )

        return super().get_run_args(
            code=submit_app,
            inputs=extended_inputs,
            outputs=extended_outputs,
            arguments=arguments,
        )

    @runnable_by_pipeline
    def run(
        self,
        submit_app: str,
        submit_class: Union[str, PipelineVariable],
        submit_jars: Optional[List[Union[str, PipelineVariable]]] = None,
        submit_files: Optional[List[Union[str, PipelineVariable]]] = None,
        inputs: Optional[List[ProcessingInput]] = None,
        outputs: Optional[List[ProcessingOutput]] = None,
        arguments: Optional[List[Union[str, PipelineVariable]]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        configuration: Optional[Union[List[Dict], Dict]] = None,
        spark_event_logs_s3_uri: Optional[Union[str, PipelineVariable]] = None,
        kms_key: Optional[str] = None,
    ):
        """Runs a processing job.

        Args:
            submit_app (str): Path (local or S3) to Jar file to submit to Spark as
                the primary application
            submit_class (str or PipelineVariable): Java class reference to submit to Spark
                as the primary application
            submit_jars (list[str] or list[PipelineVariable]): List of paths (local or S3)
                to provide for `spark-submit --jars` option
            submit_files (list[str] or list[PipelineVariable]): List of paths (local or S3)
                to provide for `spark-submit --files` option
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str] or list[PipelineVariable]): A list of string arguments to
                be passed to a processing job (default: None).
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the base job name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
            configuration (list[dict] or dict): Configuration for Hadoop, Spark, or Hive.
                List or dictionary of EMR-style classifications.
                https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
            spark_event_logs_s3_uri (str or PipelineVariable): S3 path where spark application
                events will be published to.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        if not submit_app:
            raise ValueError("submit_app is required")

        extended_inputs, extended_outputs = self._extend_processing_args(
            inputs=inputs,
            outputs=outputs,
            submit_class=submit_class,
            submit_jars=submit_jars,
            submit_files=submit_files,
            configuration=configuration,
            spark_event_logs_s3_uri=spark_event_logs_s3_uri,
        )

        return super().run(
            submit_app=submit_app,
            inputs=extended_inputs,
            outputs=extended_outputs,
            arguments=arguments,
            wait=wait,
            logs=logs,
            job_name=self._current_job_name,
            experiment_config=experiment_config,
            kms_key=kms_key,
        )

    def _extend_processing_args(self, inputs, outputs, **kwargs):
        self.command = [_SparkProcessorBase._default_command]
        if kwargs.get("submit_class"):
            self.command.extend(["--class", kwargs.get("submit_class")])
        else:
            raise ValueError("submit_class is required")

        if inputs is None:
            inputs = []

        # make a shallow copy of user inputs
        extended_inputs = copy(inputs)

        extended_inputs = self._handle_script_dependencies(
            extended_inputs, kwargs.get("submit_jars"), FileType.JAR
        )
        extended_inputs = self._handle_script_dependencies(
            extended_inputs, kwargs.get("submit_files"), FileType.FILE
        )

        return super()._extend_processing_args(extended_inputs, outputs, **kwargs)


class _HistoryServer:
    """History server class that is responsible for starting history server."""

    _container_name = "history_server"
    _entry_point = "smspark-history-server"
    arg_event_logs_s3_uri = "event_logs_s3_uri"
    arg_remote_domain_name = "remote_domain_name"

    _history_server_args_format_map = {
        arg_event_logs_s3_uri: "--event-logs-s3-uri {} ",
        arg_remote_domain_name: "--remote-domain-name {} ",
    }

    def __init__(self, cli_args, image_uri, network_config):
        self.cli_args = cli_args
        self.image_uri = image_uri
        self.network_config = network_config
        self.run_history_server_command = self._get_run_history_server_cmd()

    def run(self):
        """Runs the history server."""
        self.down()
        logger.info("Starting history server...")
        subprocess.Popen(
            self.run_history_server_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def down(self):
        """Stops and removes the container."""
        subprocess.call(["docker", "stop", self._container_name])
        subprocess.call(["docker", "rm", self._container_name])
        logger.info("History server terminated")

    # This method belongs to _HistoryServer because _CONTAINER_NAME(app name) belongs
    # to _HistoryServer. In the future, dynamically creating new app name, available
    # port should also belong to _HistoryServer rather than PySparkProcessor
    def _get_run_history_server_cmd(self):
        """Gets the history server command."""
        env_options = ""
        ser_cli_args = ""
        for key, value in self.cli_args.items():
            if key in self._history_server_args_format_map:
                ser_cli_args += self._history_server_args_format_map[key].format(value)
            else:
                env_options += f"--env {key}={value} "

        cmd = (
            f"docker run {env_options.strip()} --name {self._container_name} "
            f"{self.network_config} --entrypoint {self._entry_point} {self.image_uri} "
            f"{ser_cli_args.strip()}"
        )

        return cmd


class FileType(Enum):
    """Enum of file type"""

    JAR = 1
    PYTHON = 2
    FILE = 3


class SparkConfigUtils:
    """Util class for spark configurations"""

    _valid_configuration_keys = ["Classification", "Properties", "Configurations"]
    _valid_configuration_classifications = [
        "core-site",
        "hadoop-env",
        "hadoop-log4j",
        "hive-env",
        "hive-log4j",
        "hive-exec-log4j",
        "hive-site",
        "spark-defaults",
        "spark-env",
        "spark-log4j",
        "spark-hive-site",
        "spark-metrics",
        "yarn-env",
        "yarn-site",
        "export",
    ]

    @staticmethod
    def validate_configuration(configuration: Dict):
        """Validates the user-provided Hadoop/Spark/Hive configuration.

        This ensures that the list or dictionary the user provides will serialize to
        JSON matching the schema of EMR's application configuration

        Args:
            configuration (Dict): A dict that contains the configuration overrides to
                the default values. For more information, please visit:
                https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
        """
        emr_configure_apps_url = (
            "https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html"
        )
        if isinstance(configuration, dict):
            keys = configuration.keys()
            if "Classification" not in keys or "Properties" not in keys:
                raise ValueError(
                    f"Missing one or more required keys in configuration dictionary "
                    f"{configuration} Please see {emr_configure_apps_url} for more information"
                )

            for key in keys:
                if key not in SparkConfigUtils._valid_configuration_keys:
                    raise ValueError(
                        f"Invalid key: {key}. "
                        f"Must be one of {SparkConfigUtils._valid_configuration_keys}. "
                        f"Please see {emr_configure_apps_url} for more information."
                    )
                if key == "Classification":
                    if (
                        configuration[key]
                        not in SparkConfigUtils._valid_configuration_classifications
                    ):
                        raise ValueError(
                            f"Invalid classification: {key}. Must be one of "
                            f"{SparkConfigUtils._valid_configuration_classifications}"
                        )

        if isinstance(configuration, list):
            for item in configuration:
                SparkConfigUtils.validate_configuration(item)

    # TODO (guoqioa@): method only checks urlparse scheme, need to perform deep s3 validation
    @staticmethod
    def validate_s3_uri(spark_output_s3_path):
        """Validate whether the URI uses an S3 scheme.

        In the future, this validation will perform deeper S3 validation.

        Args:
            spark_output_s3_path (str): The URI of the Spark output S3 Path.
        """
        if is_pipeline_variable(spark_output_s3_path):
            return

        if urlparse(spark_output_s3_path).scheme != "s3":
            raise ValueError(
                f"Invalid s3 path: {spark_output_s3_path}. Please enter something like "
                "s3://bucket-name/folder-name"
            )
