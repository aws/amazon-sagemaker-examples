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

import enum
import datetime
import json
import logging
import os
import tempfile
import time
from uuid import uuid4
from copy import deepcopy
from botocore.exceptions import ClientError

import sagemaker.local.data

from sagemaker.local.image import _SageMakerContainer
from sagemaker.local.utils import copy_directory_structure, move_to_destination, get_docker_host
from sagemaker.utils import DeferredError, get_config_value
from sagemaker.local.exceptions import StepExecutionException

logger = logging.getLogger(__name__)

try:
    import urllib3
except ImportError as e:
    logger.warning("urllib3 failed to import. Local mode features will be impaired or broken.")
    # Any subsequent attempt to use urllib3 will raise the ImportError
    urllib3 = DeferredError(e)

_UNUSED_ARN = "local:arn-does-not-matter"
HEALTH_CHECK_TIMEOUT_LIMIT = 120


class _LocalProcessingJob:
    """Defines and starts a local processing job."""

    _STARTING = "Starting"
    _PROCESSING = "Processing"
    _COMPLETED = "Completed"

    def __init__(self, container):
        """Creates a local processing job.

        Args:
            container: the local container object.
        """
        self.container = container
        self.state = "Created"
        self.start_time = None
        self.end_time = None
        self.processing_job_name = ""
        self.processing_inputs = None
        self.processing_output_config = None
        self.environment = None

    def start(self, processing_inputs, processing_output_config, environment, processing_job_name):
        """Starts a local processing job.

        Args:
            processing_inputs: The processing input configuration.
            processing_output_config: The processing input configuration.
            environment: The collection of environment variables passed to the job.
            processing_job_name: The processing job name.
        """
        self.state = self._STARTING

        for item in processing_inputs:
            if "DatasetDefinition" in item:
                raise RuntimeError("DatasetDefinition is not currently supported in Local Mode")

            try:
                s3_input = item["S3Input"]
            except KeyError:
                raise ValueError("Processing input must have a valid ['S3Input']")

            item["DataUri"] = s3_input["S3Uri"]

            if "S3InputMode" in s3_input and s3_input["S3InputMode"] != "File":
                raise RuntimeError(
                    "S3InputMode: %s is not currently supported in Local Mode"
                    % s3_input["S3InputMode"]
                )

            if (
                "S3DataDistributionType" in s3_input
                and s3_input["S3DataDistributionType"] != "FullyReplicated"
            ):
                raise RuntimeError(
                    "DataDistribution: %s is not currently supported in Local Mode"
                    % s3_input["S3DataDistributionType"]
                )

            if "S3CompressionType" in s3_input and s3_input["S3CompressionType"] != "None":
                raise RuntimeError(
                    "CompressionType: %s is not currently supported in Local Mode"
                    % s3_input["S3CompressionType"]
                )

        if processing_output_config and "Outputs" in processing_output_config:
            processing_outputs = processing_output_config["Outputs"]

            for item in processing_outputs:
                if "FeatureStoreOutput" in item:
                    raise RuntimeError(
                        "FeatureStoreOutput is not currently supported in Local Mode"
                    )

                try:
                    s3_output = item["S3Output"]
                except KeyError:
                    raise ValueError("Processing output must have a valid ['S3Output']")

                if s3_output["S3UploadMode"] != "EndOfJob":
                    raise RuntimeError(
                        "UploadMode: %s is not currently supported in Local Mode."
                        % s3_output["S3UploadMode"]
                    )

        self.start_time = datetime.datetime.now()
        self.state = self._PROCESSING

        self.processing_job_name = processing_job_name
        self.processing_inputs = processing_inputs
        self.processing_output_config = processing_output_config
        self.environment = environment

        self.container.process(
            processing_inputs, processing_output_config, environment, processing_job_name
        )

        self.end_time = datetime.datetime.now()
        self.state = self._COMPLETED

    def describe(self):
        """Describes a local processing job.

        Returns:
            An object describing the processing job.
        """

        response = {
            "ProcessingJobArn": self.processing_job_name,
            "ProcessingJobName": self.processing_job_name,
            "AppSpecification": {
                "ImageUri": self.container.image,
                "ContainerEntrypoint": self.container.container_entrypoint,
                "ContainerArguments": self.container.container_arguments,
            },
            "Environment": self.environment,
            "ProcessingInputs": self.processing_inputs,
            "ProcessingOutputConfig": self.processing_output_config,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": self.container.instance_count,
                    "InstanceType": self.container.instance_type,
                    "VolumeSizeInGB": 30,
                    "VolumeKmsKeyId": None,
                }
            },
            "RoleArn": "<no_role>",
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "ProcessingJobStatus": self.state,
            "ProcessingStartTime": self.start_time,
            "ProcessingEndTime": self.end_time,
        }

        return response


class _LocalTrainingJob(object):
    """Defines and starts a local training job."""

    _STARTING = "Starting"
    _TRAINING = "Training"
    _COMPLETED = "Completed"
    _states = ["Starting", "Training", "Completed"]

    def __init__(self, container):
        """Creates a local training job.

        Args:
            container: the local container object.
        """
        self.container = container
        self.model_artifacts = None
        self.state = "created"
        self.start_time = None
        self.end_time = None
        self.environment = None
        self.training_job_name = ""

    def start(self, input_data_config, output_data_config, hyperparameters, environment, job_name):
        """Starts a local training job.

        Args:
            input_data_config (dict): The Input Data Configuration, this contains data such as the
                channels to be used for training.
            output_data_config (dict): The configuration of the output data.
            hyperparameters (dict): The HyperParameters for the training job.
            environment (dict): The collection of environment variables passed to the job.
            job_name (str): Name of the local training job being run.
        """
        for channel in input_data_config:
            if channel["DataSource"] and "S3DataSource" in channel["DataSource"]:
                data_distribution = channel["DataSource"]["S3DataSource"]["S3DataDistributionType"]
                data_uri = channel["DataSource"]["S3DataSource"]["S3Uri"]
            elif channel["DataSource"] and "FileDataSource" in channel["DataSource"]:
                data_distribution = channel["DataSource"]["FileDataSource"][
                    "FileDataDistributionType"
                ]
                data_uri = channel["DataSource"]["FileDataSource"]["FileUri"]
            else:
                raise ValueError(
                    "Need channel['DataSource'] to have ['S3DataSource'] or ['FileDataSource']"
                )

            # use a single Data URI - this makes handling S3 and File Data easier down the stack
            channel["DataUri"] = data_uri

            if data_distribution != "FullyReplicated":
                raise RuntimeError(
                    "DataDistribution: %s is not currently supported in Local Mode"
                    % data_distribution
                )

        self.start_time = datetime.datetime.now()
        self.state = self._TRAINING
        self.environment = environment

        self.model_artifacts = self.container.train(
            input_data_config, output_data_config, hyperparameters, environment, job_name
        )
        self.end_time = datetime.datetime.now()
        self.state = self._COMPLETED
        self.training_job_name = job_name

    def describe(self):
        """Placeholder docstring"""
        response = {
            "TrainingJobName": self.training_job_name,
            "TrainingJobArn": _UNUSED_ARN,
            "ResourceConfig": {"InstanceCount": self.container.instance_count},
            "TrainingJobStatus": self.state,
            "TrainingStartTime": self.start_time,
            "TrainingEndTime": self.end_time,
            "ModelArtifacts": {"S3ModelArtifacts": self.model_artifacts},
        }
        return response


class _LocalTransformJob(object):
    """Placeholder docstring"""

    _CREATING = "Creating"
    _COMPLETED = "Completed"

    def __init__(self, transform_job_name, model_name, local_session=None):
        from sagemaker.local import LocalSession

        self.local_session = local_session or LocalSession()
        local_client = self.local_session.sagemaker_client

        self.name = transform_job_name
        self.model_name = model_name

        # TODO - support SageMaker Models not just local models. This is not
        # ideal but it may be a good thing to do.
        self.primary_container = local_client.describe_model(model_name)["PrimaryContainer"]
        self.container = None
        self.start_time = None
        self.end_time = None
        self.batch_strategy = None
        self.transform_resources = None
        self.input_data = None
        self.output_data = None
        self.environment = {}
        self.state = _LocalTransformJob._CREATING

    def start(self, input_data, output_data, transform_resources, **kwargs):
        """Start the Local Transform Job

        Args:
            input_data (dict): Describes the dataset to be transformed and the
                location where it is stored.
            output_data (dict): Identifies the location where to save the
                results from the transform job
            transform_resources (dict): compute instances for the transform job.
                Currently only supports local or local_gpu
            **kwargs: additional arguments coming from the boto request object
        """
        self.transform_resources = transform_resources
        self.input_data = input_data
        self.output_data = output_data

        image = self.primary_container["Image"]
        instance_type = transform_resources["InstanceType"]
        instance_count = 1

        environment = self._get_container_environment(**kwargs)

        # Start the container, pass the environment and wait for it to start up
        self.container = _SageMakerContainer(
            instance_type, instance_count, image, self.local_session
        )
        self.container.serve(self.primary_container["ModelDataUrl"], environment)

        serving_port = get_config_value("local.serving_port", self.local_session.config) or 8080
        _wait_for_serving_container(serving_port)

        # Get capabilities from Container if needed
        endpoint_url = "http://%s:%d/execution-parameters" % (get_docker_host(), serving_port)
        response, code = _perform_request(endpoint_url)
        if code == 200:
            execution_parameters = json.loads(response.data.decode("utf-8"))
            # MaxConcurrentTransforms is ignored because we currently only support 1
            for setting in ("BatchStrategy", "MaxPayloadInMB"):
                if setting not in kwargs and setting in execution_parameters:
                    kwargs[setting] = execution_parameters[setting]

        # Apply Defaults if none was provided
        kwargs.update(self._get_required_defaults(**kwargs))

        self.start_time = datetime.datetime.now()
        self.batch_strategy = kwargs["BatchStrategy"]
        if "Environment" in kwargs:
            self.environment = kwargs["Environment"]

        # run the batch inference requests
        self._perform_batch_inference(input_data, output_data, **kwargs)
        self.end_time = datetime.datetime.now()
        self.state = self._COMPLETED

    def describe(self):
        """Describe this _LocalTransformJob

        The response is a JSON-like dictionary that follows the response of
        the boto describe_transform_job() API.

        Returns:
            dict: description of this _LocalTransformJob
        """
        response = {
            "TransformJobStatus": self.state,
            "ModelName": self.model_name,
            "TransformJobName": self.name,
            "TransformJobArn": _UNUSED_ARN,
            "TransformEndTime": self.end_time,
            "CreationTime": self.start_time,
            "TransformStartTime": self.start_time,
            "Environment": {},
            "BatchStrategy": self.batch_strategy,
        }

        if self.transform_resources:
            response["TransformResources"] = self.transform_resources

        if self.output_data:
            response["TransformOutput"] = self.output_data

        if self.input_data:
            response["TransformInput"] = self.input_data

        return response

    def _get_container_environment(self, **kwargs):
        """Get all the Environment variables that will be passed to the container.

        Certain input fields such as BatchStrategy have different values for
        the API vs the Environment variables, such as SingleRecord vs
        SINGLE_RECORD. This method also handles this conversion.

        Args:
            **kwargs: existing transform arguments

        Returns:
            dict: All the environment variables that should be set in the
            container
        """
        environment = {}
        environment.update(self.primary_container["Environment"])
        environment["SAGEMAKER_BATCH"] = "True"
        if "MaxPayloadInMB" in kwargs:
            environment["SAGEMAKER_MAX_PAYLOAD_IN_MB"] = str(kwargs["MaxPayloadInMB"])

        if "BatchStrategy" in kwargs:
            if kwargs["BatchStrategy"] == "SingleRecord":
                strategy_env_value = "SINGLE_RECORD"
            elif kwargs["BatchStrategy"] == "MultiRecord":
                strategy_env_value = "MULTI_RECORD"
            else:
                raise ValueError("Invalid BatchStrategy, must be 'SingleRecord' or 'MultiRecord'")
            environment["SAGEMAKER_BATCH_STRATEGY"] = strategy_env_value

        # we only do 1 max concurrent transform in Local Mode
        if "MaxConcurrentTransforms" in kwargs and int(kwargs["MaxConcurrentTransforms"]) > 1:
            logger.warning(
                "Local Mode only supports 1 ConcurrentTransform. Setting MaxConcurrentTransforms "
                "to 1"
            )
        environment["SAGEMAKER_MAX_CONCURRENT_TRANSFORMS"] = "1"

        # if there were environment variables passed to the Transformer we will pass them to the
        # container as well.
        if "Environment" in kwargs:
            environment.update(kwargs["Environment"])
        return environment

    def _get_required_defaults(self, **kwargs):
        """Return the default values.

         The values might be anything that was not provided by either the user or the container

        Args:
            **kwargs: current transform arguments

        Returns:
            dict: key/values for the default parameters that are missing.
        """
        defaults = {}
        if "BatchStrategy" not in kwargs:
            defaults["BatchStrategy"] = "MultiRecord"

        if "MaxPayloadInMB" not in kwargs:
            defaults["MaxPayloadInMB"] = 6

        return defaults

    def _get_working_directory(self):
        """Placeholder docstring"""
        # Root dir to use for intermediate data location. To make things simple we will write here
        # regardless of the final destination. At the end the files will either be moved or
        # uploaded to S3 and deleted.
        root_dir = get_config_value("local.container_root", self.local_session.config)
        if root_dir:
            root_dir = os.path.abspath(root_dir)

        working_dir = tempfile.mkdtemp(dir=root_dir)
        return working_dir

    def _prepare_data_transformation(self, input_data, batch_strategy):
        """Prepares the data for transformation.

        Args:
            input_data: Input data source.
            batch_strategy: Strategy for batch transformation to get.

        Returns:
            A (data source, batch provider) pair.
        """
        input_path = input_data["DataSource"]["S3DataSource"]["S3Uri"]
        data_source = sagemaker.local.data.get_data_source_instance(input_path, self.local_session)

        split_type = input_data["SplitType"] if "SplitType" in input_data else None
        splitter = sagemaker.local.data.get_splitter_instance(split_type)

        batch_provider = sagemaker.local.data.get_batch_strategy_instance(batch_strategy, splitter)
        return data_source, batch_provider

    def _perform_batch_inference(self, input_data, output_data, **kwargs):
        """Perform batch inference on the given input data.

        Transforms the input data to feed the serving container. It first gathers
        the files from S3 or Local FileSystem. It then splits the files as required
        (Line, RecordIO, None), and finally, it batch them according to the batch
        strategy and limit the request size.

        Args:
            input_data: Input data source.
            output_data: Output data source.
            **kwargs: Additional configuration arguments.
        """
        batch_strategy = kwargs["BatchStrategy"]
        max_payload = int(kwargs["MaxPayloadInMB"])
        data_source, batch_provider = self._prepare_data_transformation(input_data, batch_strategy)

        # Output settings
        accept = output_data["Accept"] if "Accept" in output_data else None

        working_dir = self._get_working_directory()
        dataset_dir = data_source.get_root_dir()

        for fn in data_source.get_file_list():

            relative_path = os.path.dirname(os.path.relpath(fn, dataset_dir))
            filename = os.path.basename(fn)
            copy_directory_structure(working_dir, relative_path)
            destination_path = os.path.join(working_dir, relative_path, filename + ".out")

            with open(destination_path, "wb") as f:
                for item in batch_provider.pad(fn, max_payload):
                    # call the container and add the result to inference.
                    response = self.local_session.sagemaker_runtime_client.invoke_endpoint(
                        item, "", input_data["ContentType"], accept
                    )

                    response_body = response["Body"]
                    data = response_body.read()
                    response_body.close()
                    f.write(data)
                    if "AssembleWith" in output_data and output_data["AssembleWith"] == "Line":
                        f.write(b"\n")

        move_to_destination(working_dir, output_data["S3OutputPath"], self.name, self.local_session)
        self.container.stop_serving()


class _LocalModel(object):
    """Placeholder docstring"""

    def __init__(self, model_name, primary_container):
        self.model_name = model_name
        self.primary_container = primary_container
        self.creation_time = datetime.datetime.now()

    def describe(self):
        """Placeholder docstring"""
        response = {
            "ModelName": self.model_name,
            "CreationTime": self.creation_time,
            "ExecutionRoleArn": _UNUSED_ARN,
            "ModelArn": _UNUSED_ARN,
            "PrimaryContainer": self.primary_container,
        }
        return response


class _LocalEndpointConfig(object):
    """Placeholder docstring"""

    def __init__(self, config_name, production_variants, tags=None):
        self.name = config_name
        self.production_variants = production_variants
        self.tags = tags
        self.creation_time = datetime.datetime.now()

    def describe(self):
        """Placeholder docstring"""
        response = {
            "EndpointConfigName": self.name,
            "EndpointConfigArn": _UNUSED_ARN,
            "Tags": self.tags,
            "CreationTime": self.creation_time,
            "ProductionVariants": self.production_variants,
        }
        return response


class _LocalEndpoint(object):
    """Placeholder docstring"""

    _CREATING = "Creating"
    _IN_SERVICE = "InService"
    _FAILED = "Failed"

    def __init__(self, endpoint_name, endpoint_config_name, tags=None, local_session=None):
        # runtime import since there is a cyclic dependency between entities and local_session
        from sagemaker.local import LocalSession

        self.local_session = local_session or LocalSession()
        local_client = self.local_session.sagemaker_client

        self.name = endpoint_name
        self.endpoint_config = local_client.describe_endpoint_config(endpoint_config_name)
        self.production_variant = self.endpoint_config["ProductionVariants"][0]
        self.tags = tags

        model_name = self.production_variant["ModelName"]
        self.primary_container = local_client.describe_model(model_name)["PrimaryContainer"]

        self.container = None
        self.create_time = None
        self.state = _LocalEndpoint._CREATING

    def serve(self):
        """Placeholder docstring"""
        image = self.primary_container["Image"]
        instance_type = self.production_variant["InstanceType"]
        instance_count = self.production_variant["InitialInstanceCount"]

        accelerator_type = self.production_variant.get("AcceleratorType")
        if accelerator_type == "local_sagemaker_notebook":
            self.primary_container["Environment"][
                "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
            ] = "true"

        self.create_time = datetime.datetime.now()
        self.container = _SageMakerContainer(
            instance_type, instance_count, image, self.local_session
        )
        self.container.serve(
            self.primary_container["ModelDataUrl"], self.primary_container["Environment"]
        )

        serving_port = get_config_value("local.serving_port", self.local_session.config) or 8080
        _wait_for_serving_container(serving_port)
        # the container is running and it passed the healthcheck status is now InService
        self.state = _LocalEndpoint._IN_SERVICE

    def stop(self):
        """Placeholder docstring"""
        if self.container:
            self.container.stop_serving()

    def describe(self):
        """Placeholder docstring"""
        response = {
            "EndpointConfigName": self.endpoint_config["EndpointConfigName"],
            "CreationTime": self.create_time,
            "ProductionVariants": self.endpoint_config["ProductionVariants"],
            "Tags": self.tags,
            "EndpointName": self.name,
            "EndpointArn": _UNUSED_ARN,
            "EndpointStatus": self.state,
        }
        return response


class _LocalPipeline(object):
    """Class representing a local SageMaker Pipeline"""

    _executions = {}

    def __init__(
        self,
        pipeline,
        pipeline_description=None,
        local_session=None,
    ):
        from sagemaker.local import LocalSession

        self.local_session = local_session or LocalSession()
        self.pipeline = pipeline
        self.pipeline_description = pipeline_description
        self.creation_time = datetime.datetime.now().timestamp()
        self.last_modified_time = self.creation_time

    def describe(self):
        """Describe Pipeline"""
        response = {
            "PipelineArn": self.pipeline.name,
            "PipelineDefinition": self.pipeline.definition(),
            "PipelineDescription": self.pipeline_description,
            "PipelineName": self.pipeline.name,
            "PipelineStatus": "Active",
            "RoleArn": "<no_role>",
            "CreationTime": self.creation_time,
            "LastModifiedTime": self.last_modified_time,
        }
        return response

    def start(self, **kwargs):
        """Start a pipeline execution. Returns a _LocalPipelineExecution object."""
        from sagemaker.local.pipeline import LocalPipelineExecutor

        execution_id = str(uuid4())
        execution = _LocalPipelineExecution(execution_id, self.pipeline, **kwargs)

        self._executions[execution_id] = execution
        print(
            f"Starting execution for pipeline {self.pipeline.name}. Execution ID is {execution_id}"
        )
        self.last_modified_time = datetime.datetime.now().timestamp()

        return LocalPipelineExecutor(execution, self.local_session).execute()


class _LocalPipelineExecution(object):
    """Class representing a local SageMaker pipeline execution."""

    def __init__(
        self,
        execution_id,
        pipeline,
        PipelineParameters=None,
        PipelineExecutionDescription=None,
        PipelineExecutionDisplayName=None,
    ):
        from sagemaker.workflow.pipeline import PipelineGraph

        self.pipeline = pipeline
        self.pipeline_execution_name = execution_id
        self.pipeline_execution_description = PipelineExecutionDescription
        self.pipeline_execution_display_name = PipelineExecutionDisplayName
        self.status = _LocalExecutionStatus.EXECUTING.value
        self.failure_reason = None
        self.creation_time = datetime.datetime.now().timestamp()
        self.last_modified_time = self.creation_time
        self.step_execution = {}
        self.pipeline_dag = PipelineGraph.from_pipeline(self.pipeline)
        self._initialize_step_execution(self.pipeline_dag.step_map.values())
        self.pipeline_parameters = self._initialize_and_validate_parameters(PipelineParameters)
        self._blocked_steps = {}

    def describe(self):
        """Describe Pipeline Execution."""
        response = {
            "CreationTime": self.creation_time,
            "LastModifiedTime": self.last_modified_time,
            "FailureReason": self.failure_reason,
            "PipelineArn": self.pipeline.name,
            "PipelineExecutionArn": self.pipeline_execution_name,
            "PipelineExecutionDescription": self.pipeline_execution_description,
            "PipelineExecutionDisplayName": self.pipeline_execution_display_name,
            "PipelineExecutionStatus": self.status,
        }
        filtered_response = {k: v for k, v in response.items() if v is not None}
        return filtered_response

    def list_steps(self):
        """List pipeline execution steps."""
        return {
            "PipelineExecutionSteps": [
                step.to_list_steps_response()
                for step in self.step_execution.values()
                if step.status is not None
            ]
        }

    def update_execution_success(self):
        """Mark execution as succeeded."""
        self.status = _LocalExecutionStatus.SUCCEEDED.value
        self.last_modified_time = datetime.datetime.now().timestamp()
        print(f"Pipeline execution {self.pipeline_execution_name} SUCCEEDED")

    def update_execution_failure(self, step_name, failure_message):
        """Mark execution as failed."""
        self.status = _LocalExecutionStatus.FAILED.value
        self.failure_reason = f"Step '{step_name}' failed with message: {failure_message}"
        self.last_modified_time = datetime.datetime.now().timestamp()
        print(
            f"Pipeline execution {self.pipeline_execution_name} FAILED because step "
            f"'{step_name}' failed."
        )

    def update_step_properties(self, step_name, step_properties):
        """Update pipeline step execution output properties."""
        self.step_execution.get(step_name).update_step_properties(step_properties)
        print(f"Pipeline step '{step_name}' SUCCEEDED.")

    def update_step_failure(self, step_name, failure_message):
        """Mark step_name as failed."""
        print(f"Pipeline step '{step_name}' FAILED. Failure message is: {failure_message}")
        self.step_execution.get(step_name).update_step_failure(failure_message)

    def mark_step_executing(self, step_name):
        """Update pipelines step's status to EXECUTING and start_time to now."""
        print(f"Starting pipeline step: '{step_name}'")
        self.step_execution.get(step_name).mark_step_executing()

    def _initialize_step_execution(self, steps):
        """Initialize step_execution dict."""
        from sagemaker.workflow.steps import StepTypeEnum, Step

        supported_steps_types = (
            StepTypeEnum.TRAINING,
            StepTypeEnum.PROCESSING,
            StepTypeEnum.TRANSFORM,
            StepTypeEnum.CONDITION,
            StepTypeEnum.FAIL,
            StepTypeEnum.CREATE_MODEL,
        )

        for step in steps:
            if isinstance(step, Step):
                if step.step_type not in supported_steps_types:
                    error_msg = self._construct_validation_exception_message(
                        "Step type {} is not supported in local mode.".format(step.step_type.value)
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                self.step_execution[step.name] = _LocalPipelineExecutionStep(
                    step.name, step.step_type, step.description, step.display_name
                )
                if step.step_type == StepTypeEnum.CONDITION:
                    self._initialize_step_execution(step.if_steps + step.else_steps)

    def _initialize_and_validate_parameters(self, overridden_parameters):
        """Initialize and validate pipeline parameters."""
        merged_parameters = {}
        default_parameters = {parameter.name: parameter for parameter in self.pipeline.parameters}
        if overridden_parameters is not None:
            for (param_name, param_value) in overridden_parameters.items():
                if param_name not in default_parameters:
                    error_msg = self._construct_validation_exception_message(
                        "Unknown parameter '{}'".format(param_name)
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                parameter_type = default_parameters[param_name].parameter_type
                if type(param_value) != parameter_type.python_type:  # pylint: disable=C0123
                    error_msg = self._construct_validation_exception_message(
                        "Unexpected type for parameter '{}'. Expected {} but found "
                        "{}.".format(param_name, parameter_type.python_type, type(param_value))
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                merged_parameters[param_name] = param_value
        for param_name, default_parameter in default_parameters.items():
            if param_name not in merged_parameters:
                if default_parameter.default_value is None:
                    error_msg = self._construct_validation_exception_message(
                        "Parameter '{}' is undefined.".format(param_name)
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                merged_parameters[param_name] = default_parameter.default_value
        return merged_parameters

    @staticmethod
    def _construct_validation_exception_message(exception_msg):
        """Construct error response for botocore.exceptions.ClientError"""
        return {"Error": {"Code": "ValidationException", "Message": exception_msg}}


class _LocalPipelineExecutionStep(object):
    """Class representing a local pipeline execution step."""

    def __init__(
        self,
        name,
        step_type,
        description,
        display_name=None,
        start_time=None,
        end_time=None,
        status=None,
        properties=None,
        failure_reason=None,
    ):
        from sagemaker.workflow.steps import StepTypeEnum

        self.name = name
        self.type = step_type
        self.description = description
        self.display_name = display_name
        self.status = status
        self.failure_reason = failure_reason
        self.properties = properties or {}
        self.start_time = start_time
        self.end_time = end_time
        self._step_type_to_output_format_map = {
            StepTypeEnum.TRAINING: self._construct_training_metadata,
            StepTypeEnum.PROCESSING: self._construct_processing_metadata,
            StepTypeEnum.TRANSFORM: self._construct_transform_metadata,
            StepTypeEnum.CREATE_MODEL: self._construct_model_metadata,
            StepTypeEnum.CONDITION: self._construct_condition_metadata,
            StepTypeEnum.FAIL: self._construct_fail_metadata,
        }

    def update_step_properties(self, properties):
        """Update pipeline step execution output properties."""
        self.properties = deepcopy(properties)
        self.status = _LocalExecutionStatus.SUCCEEDED.value
        self.end_time = datetime.datetime.now().timestamp()

    def update_step_failure(self, failure_message):
        """Update pipeline step execution failure status and message."""
        self.failure_reason = failure_message
        self.status = _LocalExecutionStatus.FAILED.value
        self.end_time = datetime.datetime.now().timestamp()
        raise StepExecutionException(self.name, failure_message)

    def mark_step_executing(self):
        """Update pipelines step's status to EXECUTING and start_time to now"""
        self.status = _LocalExecutionStatus.EXECUTING.value
        self.start_time = datetime.datetime.now().timestamp()

    def to_list_steps_response(self):
        """Convert to response dict for list_steps calls."""
        response = {
            "EndTime": self.end_time,
            "FailureReason": self.failure_reason,
            "Metadata": self._construct_metadata(),
            "StartTime": self.start_time,
            "StepDescription": self.description,
            "StepDisplayName": self.display_name,
            "StepName": self.name,
            "StepStatus": self.status,
        }
        filtered_response = {k: v for k, v in response.items() if v is not None}
        return filtered_response

    def _construct_metadata(self):
        """Constructs the metadata shape for the list_steps_response."""
        if self.properties:
            return self._step_type_to_output_format_map[self.type]()
        return None

    def _construct_training_metadata(self):
        """Construct training job metadata response."""
        return {"TrainingJob": {"Arn": self.properties["TrainingJobName"]}}

    def _construct_processing_metadata(self):
        """Construct processing job metadata response."""
        return {"ProcessingJob": {"Arn": self.properties["ProcessingJobName"]}}

    def _construct_transform_metadata(self):
        """Construct transform job metadata response."""
        return {"TransformJob": {"Arn": self.properties["TransformJobName"]}}

    def _construct_model_metadata(self):
        """Construct create model step metadata response."""
        return {"Model": {"Arn": self.properties["ModelName"]}}

    def _construct_condition_metadata(self):
        """Construct condition step metadata response."""
        return {"Condition": {"Outcome": self.properties["Outcome"]}}

    def _construct_fail_metadata(self):
        """Construct fail step metadata response."""
        return {"Fail": {"ErrorMessage": self.properties["ErrorMessage"]}}


class _LocalExecutionStatus(enum.Enum):
    """Pipeline execution status."""

    EXECUTING = "Executing"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"


def _wait_for_serving_container(serving_port):
    """Placeholder docstring."""
    i = 0
    http = urllib3.PoolManager()

    endpoint_url = "http://%s:%d/ping" % (get_docker_host(), serving_port)
    while True:
        i += 5
        if i >= HEALTH_CHECK_TIMEOUT_LIMIT:
            raise RuntimeError("Giving up, endpoint didn't launch correctly")

        logger.info("Checking if serving container is up, attempt: %s", i)
        _, code = _perform_request(endpoint_url, http)
        if code != 200:
            logger.info("Container still not up, got: %s", code)
        else:
            return

        time.sleep(5)


def _perform_request(endpoint_url, pool_manager=None):
    """Placeholder docstring."""
    http = pool_manager or urllib3.PoolManager()
    try:
        r = http.request("GET", endpoint_url)
        code = r.status
    except urllib3.exceptions.RequestError:
        return None, -1
    return r, code
