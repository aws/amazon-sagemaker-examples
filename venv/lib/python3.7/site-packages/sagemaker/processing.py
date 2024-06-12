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
"""This module contains code related to the ``Processor`` class.

which is used for Amazon SageMaker Processing Jobs. These jobs let users perform
data pre-processing, post-processing, feature engineering, data validation, and model evaluation,
and interpretation on Amazon SageMaker.
"""
from __future__ import absolute_import

import os
import pathlib
import logging
from textwrap import dedent
from typing import Dict, List, Optional, Union
from copy import copy

import attr

from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import url2pathname
from sagemaker import s3
from sagemaker.config import (
    PROCESSING_JOB_KMS_KEY_ID_PATH,
    PROCESSING_JOB_SECURITY_GROUP_IDS_PATH,
    PROCESSING_JOB_SUBNETS_PATH,
    PROCESSING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    PROCESSING_JOB_VOLUME_KMS_KEY_ID_PATH,
    PROCESSING_JOB_ROLE_ARN_PATH,
    PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
    PROCESSING_JOB_ENVIRONMENT_PATH,
)
from sagemaker.job import _Job
from sagemaker.local import LocalSession
from sagemaker.network import NetworkConfig
from sagemaker.utils import (
    base_name_from_image,
    get_config_value,
    name_from_base,
    check_and_get_run_experiment_config,
    resolve_value_from_config,
    resolve_class_attribute_from_config,
)
from sagemaker.session import Session
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline_context import runnable_by_pipeline
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.dataset_definition.inputs import S3Input, DatasetDefinition
from sagemaker.apiutils._base_types import ApiObject
from sagemaker.s3 import S3Uploader

logger = logging.getLogger(__name__)


class Processor(object):
    """Handles Amazon SageMaker Processing tasks."""

    JOB_CLASS_NAME = "processing-job"

    def __init__(
        self,
        role: str = None,
        image_uri: Union[str, PipelineVariable] = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        entrypoint: Optional[List[Union[str, PipelineVariable]]] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initializes a ``Processor`` instance.

        The ``Processor`` handles Amazon SageMaker Processing tasks.

        Args:
            role (str or PipelineVariable): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            image_uri (str or PipelineVariable): The URI of the Docker image to use for the
                processing jobs.
            instance_count (int or PipelineVariable): The number of instances to run
                a processing job with.
            instance_type (str or PipelineVariable): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            entrypoint (list[str] or list[PipelineVariable]): The entrypoint for the
                processing job (default: None). This is in the form of a list of strings
                that make a command.
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing
                volume (default: None).
            output_kms_key (str or PipelineVariable): The KMS key ID for processing job
                outputs (default: None).
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            base_job_name (str): Prefix for processing job name. If not specified,
                the processor generates a default job name, based on the
                processing image name and current timestamp.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables
                to be passed to the processing jobs (default: None).
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): List of tags
                to be passed to the processing job (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        self.image_uri = image_uri
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.entrypoint = entrypoint
        self.volume_size_in_gb = volume_size_in_gb
        self.max_runtime_in_seconds = max_runtime_in_seconds
        self.base_job_name = base_job_name
        self.tags = tags

        self.jobs = []
        self.latest_job = None
        self._current_job_name = None
        self.arguments = None

        if self.instance_type in ("local", "local_gpu"):
            if not isinstance(sagemaker_session, LocalSession):
                # Until Local Mode Processing supports local code, we need to disable it:
                sagemaker_session = LocalSession(disable_local_code=True)

        self.sagemaker_session = sagemaker_session or Session()
        self.output_kms_key = resolve_value_from_config(
            output_kms_key, PROCESSING_JOB_KMS_KEY_ID_PATH, sagemaker_session=self.sagemaker_session
        )
        self.volume_kms_key = resolve_value_from_config(
            volume_kms_key,
            PROCESSING_JOB_VOLUME_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            network_config,
            "subnets",
            PROCESSING_JOB_SUBNETS_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "security_group_ids",
            PROCESSING_JOB_SECURITY_GROUP_IDS_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "enable_network_isolation",
            PROCESSING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "encrypt_inter_container_traffic",
            PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.role = resolve_value_from_config(
            role, PROCESSING_JOB_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        if not self.role:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create a Processing job.")

        self.env = resolve_value_from_config(
            env, PROCESSING_JOB_ENVIRONMENT_PATH, sagemaker_session=self.sagemaker_session
        )

    @runnable_by_pipeline
    def run(
        self,
        inputs: Optional[List["ProcessingInput"]] = None,
        outputs: Optional[List["ProcessingOutput"]] = None,
        arguments: Optional[List[Union[str, PipelineVariable]]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
    ):
        """Runs a processing job.

        Args:
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str] or list[PipelineVariable]): A list of string arguments
                to be passed to a processing job (default: None).
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
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
                * Both `ExperimentName` and `TrialName` will be ignored if the Processor instance
                is built with :class:`~sagemaker.workflow.pipeline_context.PipelineSession`.
                However, the value of `TrialComponentDisplayName` is honored for display in Studio.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        Returns:
            None or pipeline step arguments in case the Processor instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        Raises:
            ValueError: if ``logs`` is True but ``wait`` is False.
        """
        if logs and not wait:
            raise ValueError(
                """Logs can only be shown if wait is set to True.
                Please either set wait to True or set logs to False."""
            )

        normalized_inputs, normalized_outputs = self._normalize_args(
            job_name=job_name,
            arguments=arguments,
            inputs=inputs,
            kms_key=kms_key,
            outputs=outputs,
        )

        experiment_config = check_and_get_run_experiment_config(experiment_config)
        self.latest_job = ProcessingJob.start_new(
            processor=self,
            inputs=normalized_inputs,
            outputs=normalized_outputs,
            experiment_config=experiment_config,
        )
        self.jobs.append(self.latest_job)
        if wait:
            self.latest_job.wait(logs=logs)

    def _extend_processing_args(self, inputs, outputs, **kwargs):  # pylint: disable=W0613
        """Extend inputs and outputs based on extra parameters"""
        return inputs, outputs

    def _normalize_args(
        self,
        job_name=None,
        arguments=None,
        inputs=None,
        outputs=None,
        code=None,
        kms_key=None,
    ):
        """Normalizes the arguments so that they can be passed to the job run

        Args:
            job_name (str): Name of the processing job to be created. If not specified, one
                is generated, using the base name given to the constructor, if applicable
                (default: None).
            arguments (list[str]): A list of string arguments to be passed to a
                processing job (default: None).
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            code (str): This can be an S3 URI or a local path to a file with the framework
                script to run (default: None). A no op in the base class.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        """
        if code and is_pipeline_variable(code):
            raise ValueError(
                "code argument has to be a valid S3 URI or local file path "
                + "rather than a pipeline variable"
            )

        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        inputs_with_code = self._include_code_in_inputs(inputs, code, kms_key)
        normalized_inputs = self._normalize_inputs(inputs_with_code, kms_key)
        normalized_outputs = self._normalize_outputs(outputs)
        self.arguments = arguments

        return normalized_inputs, normalized_outputs

    def _include_code_in_inputs(self, inputs, _code, _kms_key):
        """A no op in the base class to include code in the processing job inputs.

        Args:
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects.
            _code (str): This can be an S3 URI or a local path to a file with the framework
                script to run (default: None). A no op in the base class.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            list[:class:`~sagemaker.processing.ProcessingInput`]: inputs
        """
        return inputs

    def _generate_current_job_name(self, job_name=None):
        """Generates the job name before running a processing job.

        Args:
            job_name (str): Name of the processing job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.

        Returns:
            str: The supplied or generated job name.
        """
        if job_name is not None:
            return job_name
        # Honor supplied base_job_name or generate it.
        if self.base_job_name:
            base_name = self.base_job_name
        else:
            base_name = base_name_from_image(
                self.image_uri, default_base_name=Processor.JOB_CLASS_NAME
            )

        return name_from_base(base_name)

    def _normalize_inputs(self, inputs=None, kms_key=None):
        """Ensures that all the ``ProcessingInput`` objects have names and S3 URIs.

        Args:
            inputs (list[sagemaker.processing.ProcessingInput]): A list of ``ProcessingInput``
                objects to be normalized (default: None). If not specified,
                an empty list is returned.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            list[sagemaker.processing.ProcessingInput]: The list of normalized
                ``ProcessingInput`` objects.

        Raises:
            TypeError: if the inputs are not ``ProcessingInput`` objects.
        """
        from sagemaker.workflow.utilities import _pipeline_config

        # Initialize a list of normalized ProcessingInput objects.
        normalized_inputs = []
        if inputs is not None:
            # Iterate through the provided list of inputs.
            for count, file_input in enumerate(inputs, 1):
                if not isinstance(file_input, ProcessingInput):
                    raise TypeError("Your inputs must be provided as ProcessingInput objects.")
                # Generate a name for the ProcessingInput if it doesn't have one.
                if file_input.input_name is None:
                    file_input.input_name = "input-{}".format(count)

                if is_pipeline_variable(file_input.source) or file_input.dataset_definition:
                    normalized_inputs.append(file_input)
                    continue
                if is_pipeline_variable(file_input.s3_input.s3_uri):
                    normalized_inputs.append(file_input)
                    continue
                # If the source is a local path, upload it to S3
                # and save the S3 uri in the ProcessingInput source.
                parse_result = urlparse(file_input.s3_input.s3_uri)
                if parse_result.scheme != "s3":
                    if _pipeline_config:
                        desired_s3_uri = s3.s3_path_join(
                            "s3://",
                            self.sagemaker_session.default_bucket(),
                            self.sagemaker_session.default_bucket_prefix,
                            _pipeline_config.pipeline_name,
                            _pipeline_config.step_name,
                            "input",
                            file_input.input_name,
                        )
                    else:
                        desired_s3_uri = s3.s3_path_join(
                            "s3://",
                            self.sagemaker_session.default_bucket(),
                            self.sagemaker_session.default_bucket_prefix,
                            self._current_job_name,
                            "input",
                            file_input.input_name,
                        )
                    s3_uri = s3.S3Uploader.upload(
                        local_path=file_input.s3_input.s3_uri,
                        desired_s3_uri=desired_s3_uri,
                        sagemaker_session=self.sagemaker_session,
                        kms_key=kms_key,
                    )
                    file_input.s3_input.s3_uri = s3_uri
                normalized_inputs.append(file_input)
        return normalized_inputs

    def _normalize_outputs(self, outputs=None):
        """Ensures that all the outputs are ``ProcessingOutput`` objects with names and S3 URIs.

        Args:
            outputs (list[sagemaker.processing.ProcessingOutput]): A list
                of outputs to be normalized (default: None). Can be either strings or
                ``ProcessingOutput`` objects. If not specified,
                an empty list is returned.

        Returns:
            list[sagemaker.processing.ProcessingOutput]: The list of normalized
                ``ProcessingOutput`` objects.

        Raises:
            TypeError: if the outputs are not ``ProcessingOutput`` objects.
        """
        # Initialize a list of normalized ProcessingOutput objects.
        from sagemaker.workflow.utilities import _pipeline_config

        normalized_outputs = []
        if outputs is not None:
            # Iterate through the provided list of outputs.
            for count, output in enumerate(outputs, 1):
                if not isinstance(output, ProcessingOutput):
                    raise TypeError("Your outputs must be provided as ProcessingOutput objects.")
                # Generate a name for the ProcessingOutput if it doesn't have one.
                if output.output_name is None:
                    output.output_name = "output-{}".format(count)
                if is_pipeline_variable(output.destination):
                    normalized_outputs.append(output)
                    continue
                # If the output's destination is not an s3_uri, create one.
                parse_result = urlparse(output.destination)
                if parse_result.scheme != "s3":
                    if _pipeline_config:
                        s3_uri = Join(
                            on="/",
                            values=[
                                "s3:/",
                                self.sagemaker_session.default_bucket(),
                                *(
                                    # don't include default_bucket_prefix if it is None or ""
                                    [self.sagemaker_session.default_bucket_prefix]
                                    if self.sagemaker_session.default_bucket_prefix
                                    else []
                                ),
                                _pipeline_config.pipeline_name,
                                ExecutionVariables.PIPELINE_EXECUTION_ID,
                                _pipeline_config.step_name,
                                "output",
                                output.output_name,
                            ],
                        )
                    else:
                        s3_uri = s3.s3_path_join(
                            "s3://",
                            self.sagemaker_session.default_bucket(),
                            self.sagemaker_session.default_bucket_prefix,
                            self._current_job_name,
                            "output",
                            output.output_name,
                        )
                    output.destination = s3_uri
                normalized_outputs.append(output)
        return normalized_outputs


class ScriptProcessor(Processor):
    """Handles Amazon SageMaker processing tasks for jobs using a machine learning framework."""

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        image_uri: Union[str, PipelineVariable] = None,
        command: List[str] = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initializes a ``ScriptProcessor`` instance.

        The ``ScriptProcessor`` handles Amazon SageMaker Processing tasks for jobs
        using a machine learning framework, which allows for providing a script to be
        run as part of the Processing Job.

        Args:
            role (str or PipelineVariable): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            image_uri (str or PipelineVariable): The URI of the Docker image to use for the
                processing jobs.
            command ([str]): The command to run, along with any command-line flags.
                Example: ["python3", "-v"].
            instance_count (int or PipelineVariable): The number of instances to run
                a processing job with.
            instance_type (str or PipelineVariable): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing
                volume (default: None).
            output_kms_key (str or PipelineVariable): The KMS key ID for processing
                job outputs (default: None).
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                processing image name and current timestamp.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            env (dict[str, str] or dict[str, PipelineVariable])): Environment variables to
                be passed to the processing jobs (default: None).
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): List of tags to
                be passed to the processing job (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        self._CODE_CONTAINER_BASE_PATH = "/opt/ml/processing/input/"
        self._CODE_CONTAINER_INPUT_NAME = "code"
        self.command = command

        super(ScriptProcessor, self).__init__(
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
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
        logger.warning(
            "This function has been deprecated and could break pipeline step caching. "
            "We recommend using the run() function directly with pipeline sessions"
            "to access step arguments."
        )
        return RunArgs(code=code, inputs=inputs, outputs=outputs, arguments=arguments)

    @runnable_by_pipeline
    def run(
        self,
        code: str,
        inputs: Optional[List["ProcessingInput"]] = None,
        outputs: Optional[List["ProcessingOutput"]] = None,
        arguments: Optional[List[Union[str, PipelineVariable]]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
    ):
        """Runs a processing job.

        Args:
            code (str): This can be an S3 URI or a local path to
                a file with the framework script to run.
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
                * Both `ExperimentName` and `TrialName` will be ignored if the Processor instance
                is built with :class:`~sagemaker.workflow.pipeline_context.PipelineSession`.
                However, the value of `TrialComponentDisplayName` is honored for display in Studio.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        Returns:
            None or pipeline step arguments in case the Processor instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        """
        normalized_inputs, normalized_outputs = self._normalize_args(
            job_name=job_name,
            arguments=arguments,
            inputs=inputs,
            outputs=outputs,
            code=code,
            kms_key=kms_key,
        )

        experiment_config = check_and_get_run_experiment_config(experiment_config)
        self.latest_job = ProcessingJob.start_new(
            processor=self,
            inputs=normalized_inputs,
            outputs=normalized_outputs,
            experiment_config=experiment_config,
        )
        self.jobs.append(self.latest_job)
        if wait:
            self.latest_job.wait(logs=logs)

    def _include_code_in_inputs(self, inputs, code, kms_key=None):
        """Converts code to appropriate input and includes in input list.

        Side effects include:
            * uploads code to S3 if the code is a local file.
            * sets the entrypoint attribute based on the command and user script name from code.

        Args:
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects.
            code (str): This can be an S3 URI or a local path to a file with the framework
                script to run (default: None).
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            list[:class:`~sagemaker.processing.ProcessingInput`]: inputs together with the
                code as `ProcessingInput`.
        """
        user_code_s3_uri = self._handle_user_code_url(code, kms_key)
        user_script_name = self._get_user_code_name(code)

        inputs_with_code = self._convert_code_and_add_to_inputs(inputs, user_code_s3_uri)

        self._set_entrypoint(self.command, user_script_name)
        return inputs_with_code

    def _get_user_code_name(self, code):
        """Gets the basename of the user's code from the URL the customer provided.

        Args:
            code (str): A URL to the user's code.

        Returns:
            str: The basename of the user's code.

        """
        code_url = urlparse(code)
        return os.path.basename(code_url.path)

    def _handle_user_code_url(self, code, kms_key=None):
        """Gets the S3 URL containing the user's code.

           Inspects the scheme the customer passed in ("s3://" for code in S3, "file://" or nothing
           for absolute or local file paths. Uploads the code to S3 if the code is a local file.

        Args:
            code (str): A URL to the customer's code.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            str: The S3 URL to the customer's code.

        Raises:
            ValueError: if the code isn't found, is a directory, or
                does not have a valid URL scheme.
        """
        code_url = urlparse(code)
        if code_url.scheme == "s3":
            user_code_s3_uri = code
        elif code_url.scheme == "" or code_url.scheme == "file":
            # Validate that the file exists locally and is not a directory.
            code_path = url2pathname(code_url.path)
            if not os.path.exists(code_path):
                raise ValueError(
                    """code {} wasn't found. Please make sure that the file exists.
                    """.format(
                        code
                    )
                )
            if not os.path.isfile(code_path):
                raise ValueError(
                    """code {} must be a file, not a directory. Please pass a path to a file.
                    """.format(
                        code
                    )
                )
            user_code_s3_uri = self._upload_code(code_path, kms_key)
        else:
            raise ValueError(
                "code {} url scheme {} is not recognized. Please pass a file path or S3 url".format(
                    code, code_url.scheme
                )
            )
        return user_code_s3_uri

    def _upload_code(self, code, kms_key=None):
        """Uploads a code file or directory specified as a string and returns the S3 URI.

        Args:
            code (str): A file or directory to be uploaded to S3.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            str: The S3 URI of the uploaded file or directory.

        """
        from sagemaker.workflow.utilities import _pipeline_config

        if _pipeline_config and _pipeline_config.code_hash:
            desired_s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                _pipeline_config.pipeline_name,
                self._CODE_CONTAINER_INPUT_NAME,
                _pipeline_config.code_hash,
            )
        else:
            desired_s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                self._current_job_name,
                "input",
                self._CODE_CONTAINER_INPUT_NAME,
            )
        return s3.S3Uploader.upload(
            local_path=code,
            desired_s3_uri=desired_s3_uri,
            kms_key=kms_key,
            sagemaker_session=self.sagemaker_session,
        )

    def _convert_code_and_add_to_inputs(self, inputs, s3_uri):
        """Creates a ``ProcessingInput`` object from an S3 URI and adds it to the list of inputs.

        Args:
            inputs (list[sagemaker.processing.ProcessingInput]):
                List of ``ProcessingInput`` objects.
            s3_uri (str): S3 URI of the input to be added to inputs.

        Returns:
            list[sagemaker.processing.ProcessingInput]: A new list of ``ProcessingInput`` objects,
                with the ``ProcessingInput`` object created from ``s3_uri`` appended to the list.

        """

        code_file_input = ProcessingInput(
            source=s3_uri,
            destination=str(
                pathlib.PurePosixPath(
                    self._CODE_CONTAINER_BASE_PATH, self._CODE_CONTAINER_INPUT_NAME
                )
            ),
            input_name=self._CODE_CONTAINER_INPUT_NAME,
        )
        return (inputs or []) + [code_file_input]

    def _set_entrypoint(self, command, user_script_name):
        """Sets the entrypoint based on the user's script and corresponding executable.

        Args:
            user_script_name (str): A filename with an extension.
        """
        user_script_location = str(
            pathlib.PurePosixPath(
                self._CODE_CONTAINER_BASE_PATH,
                self._CODE_CONTAINER_INPUT_NAME,
                user_script_name,
            )
        )
        self.entrypoint = command + [user_script_location]


class ProcessingJob(_Job):
    """Provides functionality to start, describe, and stop processing jobs."""

    def __init__(self, sagemaker_session, job_name, inputs, outputs, output_kms_key=None):
        """Initializes a Processing job.

        Args:
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
            job_name (str): Name of the Processing job.
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): A list of
                :class:`~sagemaker.processing.ProcessingInput` objects.
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): A list of
                :class:`~sagemaker.processing.ProcessingOutput` objects.
            output_kms_key (str): The output KMS key associated with the job (default: None).
        """
        self.inputs = inputs
        self.outputs = outputs
        self.output_kms_key = output_kms_key
        super(ProcessingJob, self).__init__(sagemaker_session=sagemaker_session, job_name=job_name)

    @classmethod
    def start_new(cls, processor, inputs, outputs, experiment_config):
        """Starts a new processing job using the provided inputs and outputs.

        Args:
            processor (:class:`~sagemaker.processing.Processor`): The ``Processor`` instance
                that started the job.
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): A list of
                :class:`~sagemaker.processing.ProcessingInput` objects.
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): A list of
                :class:`~sagemaker.processing.ProcessingOutput` objects.
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

        Returns:
            :class:`~sagemaker.processing.ProcessingJob`: The instance of ``ProcessingJob`` created
                using the ``Processor``.
        """
        process_args = cls._get_process_args(processor, inputs, outputs, experiment_config)

        # Log the job name and the user's inputs and outputs as lists of dictionaries.
        logger.debug("Job Name: %s", process_args["job_name"])
        logger.debug("Inputs: %s", process_args["inputs"])
        logger.debug("Outputs: %s", process_args["output_config"]["Outputs"])

        # Call sagemaker_session.process using the arguments dictionary.
        processor.sagemaker_session.process(**process_args)

        return cls(
            processor.sagemaker_session,
            processor._current_job_name,
            inputs,
            outputs,
            processor.output_kms_key,
        )

    @classmethod
    def _get_process_args(cls, processor, inputs, outputs, experiment_config):
        """Gets a dict of arguments for a new Amazon SageMaker processing job from the processor

        Args:
            processor (:class:`~sagemaker.processing.Processor`): The ``Processor`` instance
                that started the job.
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): A list of
                :class:`~sagemaker.processing.ProcessingInput` objects.
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): A list of
                :class:`~sagemaker.processing.ProcessingOutput` objects.
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

        Returns:
            Dict: dict for `sagemaker.session.Session.process` method
        """
        # Initialize an empty dictionary for arguments to be passed to sagemaker_session.process.
        process_request_args = {}

        # Add arguments to the dictionary.
        process_request_args["inputs"] = [inp._to_request_dict() for inp in inputs]

        process_request_args["output_config"] = {
            "Outputs": [output._to_request_dict() for output in outputs]
        }
        if processor.output_kms_key is not None:
            process_request_args["output_config"]["KmsKeyId"] = processor.output_kms_key

        process_request_args["experiment_config"] = experiment_config
        process_request_args["job_name"] = processor._current_job_name

        process_request_args["resources"] = {
            "ClusterConfig": {
                "InstanceType": processor.instance_type,
                "InstanceCount": processor.instance_count,
                "VolumeSizeInGB": processor.volume_size_in_gb,
            }
        }

        if processor.volume_kms_key is not None:
            process_request_args["resources"]["ClusterConfig"][
                "VolumeKmsKeyId"
            ] = processor.volume_kms_key

        if processor.max_runtime_in_seconds is not None:
            process_request_args["stopping_condition"] = {
                "MaxRuntimeInSeconds": processor.max_runtime_in_seconds
            }
        else:
            process_request_args["stopping_condition"] = None

        process_request_args["app_specification"] = {"ImageUri": processor.image_uri}
        if processor.arguments is not None:
            process_request_args["app_specification"]["ContainerArguments"] = processor.arguments
        if processor.entrypoint is not None:
            process_request_args["app_specification"]["ContainerEntrypoint"] = processor.entrypoint

        process_request_args["environment"] = processor.env

        if processor.network_config is not None:
            process_request_args["network_config"] = processor.network_config._to_request_dict()
        else:
            process_request_args["network_config"] = None

        process_request_args["role_arn"] = (
            processor.role
            if is_pipeline_variable(processor.role)
            else processor.sagemaker_session.expand_role(processor.role)
        )

        process_request_args["tags"] = processor.tags

        return process_request_args

    @classmethod
    def from_processing_name(cls, sagemaker_session, processing_job_name):
        """Initializes a ``ProcessingJob`` from a processing job name.

        Args:
            processing_job_name (str): Name of the processing job.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.

        Returns:
            :class:`~sagemaker.processing.ProcessingJob`: The instance of ``ProcessingJob`` created
                from the job name.
        """
        job_desc = sagemaker_session.describe_processing_job(job_name=processing_job_name)

        inputs = None
        if job_desc.get("ProcessingInputs"):
            inputs = [
                ProcessingInput(
                    input_name=processing_input["InputName"],
                    s3_input=S3Input.from_boto(processing_input.get("S3Input")),
                    dataset_definition=DatasetDefinition.from_boto(
                        processing_input.get("DatasetDefinition")
                    ),
                    app_managed=processing_input.get("AppManaged", False),
                )
                for processing_input in job_desc["ProcessingInputs"]
            ]

        outputs = None
        if job_desc.get("ProcessingOutputConfig") and job_desc["ProcessingOutputConfig"].get(
            "Outputs"
        ):
            outputs = []
            for processing_output_dict in job_desc["ProcessingOutputConfig"]["Outputs"]:
                processing_output = ProcessingOutput(
                    output_name=processing_output_dict["OutputName"],
                    app_managed=processing_output_dict.get("AppManaged", False),
                    feature_store_output=FeatureStoreOutput.from_boto(
                        processing_output_dict.get("FeatureStoreOutput")
                    ),
                )

                if "S3Output" in processing_output_dict:
                    processing_output.source = processing_output_dict["S3Output"]["LocalPath"]
                    processing_output.destination = processing_output_dict["S3Output"]["S3Uri"]

                outputs.append(processing_output)
        output_kms_key = None
        if job_desc.get("ProcessingOutputConfig"):
            output_kms_key = job_desc["ProcessingOutputConfig"].get("KmsKeyId")

        return cls(
            sagemaker_session=sagemaker_session,
            job_name=processing_job_name,
            inputs=inputs,
            outputs=outputs,
            output_kms_key=output_kms_key,
        )

    @classmethod
    def from_processing_arn(cls, sagemaker_session, processing_job_arn):
        """Initializes a ``ProcessingJob`` from a Processing ARN.

        Args:
            processing_job_arn (str): ARN of the processing job.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.

        Returns:
            :class:`~sagemaker.processing.ProcessingJob`: The instance of ``ProcessingJob`` created
                from the processing job's ARN.
        """
        processing_job_name = processing_job_arn.split(":")[5][
            len("processing-job/") :
        ]  # This is necessary while the API only vends an arn.
        return cls.from_processing_name(
            sagemaker_session=sagemaker_session, processing_job_name=processing_job_name
        )

    def _is_local_channel(self, input_url):
        """Used for Local Mode. Not yet implemented.

        Args:
            input_url (str): input URL

        Raises:
            NotImplementedError: this method is not yet implemented.
        """
        raise NotImplementedError

    def wait(self, logs=True):
        """Waits for the processing job to complete.

        Args:
            logs (bool): Whether to show the logs produced by the job (default: True).

        """
        if logs:
            self.sagemaker_session.logs_for_processing_job(self.job_name, wait=True)
        else:
            self.sagemaker_session.wait_for_processing_job(self.job_name)

    def describe(self):
        """Prints out a response from the DescribeProcessingJob API call."""
        return self.sagemaker_session.describe_processing_job(self.job_name)

    def stop(self):
        """Stops the processing job."""
        self.sagemaker_session.stop_processing_job(self.name)

    @staticmethod
    def prepare_app_specification(container_arguments, container_entrypoint, image_uri):
        """Prepares a dict that represents a ProcessingJob's AppSpecification.

        Args:
            container_arguments (list[str]): The arguments for a container
                used to run a processing job.
            container_entrypoint (list[str]): The entrypoint for a container
                used to run a processing job.
            image_uri (str): The container image to be run by the processing job.

        Returns:
            dict: Represents AppSpecification which configures the
            processing job to run a specified Docker container image.
        """
        config = {"ImageUri": image_uri}
        if container_arguments is not None:
            config["ContainerArguments"] = container_arguments
        if container_entrypoint is not None:
            config["ContainerEntrypoint"] = container_entrypoint
        return config

    @staticmethod
    def prepare_output_config(kms_key_id, outputs):
        """Prepares a dict that represents a ProcessingOutputConfig.

        Args:
            kms_key_id (str): The AWS Key Management Service (AWS KMS) key that
                Amazon SageMaker uses to encrypt the processing job output.
                KmsKeyId can be an ID of a KMS key, ARN of a KMS key, alias of a KMS key,
                or alias of a KMS key. The KmsKeyId is applied to all outputs.
            outputs (list[dict]): Output configuration information for a processing job.

        Returns:
            dict: Represents output configuration for the processing job.
        """
        config = {"Outputs": outputs}
        if kms_key_id is not None:
            config["KmsKeyId"] = kms_key_id
        return config

    @staticmethod
    def prepare_processing_resources(
        instance_count, instance_type, volume_kms_key_id, volume_size_in_gb
    ):
        """Prepares a dict that represents the ProcessingResources.

        Args:
            instance_count (int): The number of ML compute instances
                to use in the processing job. For distributed processing jobs,
                specify a value greater than 1. The default value is 1.
            instance_type (str): The ML compute instance type for the processing job.
            volume_kms_key_id (str): The AWS Key Management Service (AWS KMS) key
                that Amazon SageMaker uses to encrypt data on the storage
                volume attached to the ML compute instance(s) that run the processing job.
            volume_size_in_gb (int): The size of the ML storage volume in gigabytes
                that you want to provision. You must specify sufficient
                ML storage for your scenario.

        Returns:
            dict: Represents ProcessingResources which identifies the resources,
                ML compute instances, and ML storage volumes to deploy
                for a processing job.
        """
        processing_resources = {}
        cluster_config = {
            "InstanceCount": instance_count,
            "InstanceType": instance_type,
            "VolumeSizeInGB": volume_size_in_gb,
        }
        if volume_kms_key_id is not None:
            cluster_config["VolumeKmsKeyId"] = volume_kms_key_id
        processing_resources["ClusterConfig"] = cluster_config
        return processing_resources

    @staticmethod
    def prepare_stopping_condition(max_runtime_in_seconds):
        """Prepares a dict that represents the job's StoppingCondition.

        Args:
            max_runtime_in_seconds (int): Specifies the maximum runtime in seconds.

        Returns:
            dict
        """
        return {"MaxRuntimeInSeconds": max_runtime_in_seconds}


class ProcessingInput(object):
    """Accepts parameters that specify an Amazon S3 input for a processing job.

    Also provides a method to turn those parameters into a dictionary.
    """

    def __init__(
        self,
        source: Optional[Union[str, PipelineVariable]] = None,
        destination: Optional[Union[str, PipelineVariable]] = None,
        input_name: Optional[Union[str, PipelineVariable]] = None,
        s3_data_type: Union[str, PipelineVariable] = "S3Prefix",
        s3_input_mode: Union[str, PipelineVariable] = "File",
        s3_data_distribution_type: Union[str, PipelineVariable] = "FullyReplicated",
        s3_compression_type: Union[str, PipelineVariable] = "None",
        s3_input: Optional[S3Input] = None,
        dataset_definition: Optional[DatasetDefinition] = None,
        app_managed: Union[bool, PipelineVariable] = False,
    ):
        """Initializes a ``ProcessingInput`` instance.

        ``ProcessingInput`` accepts parameters that specify an Amazon S3 input
        for a processing job and provides a method to turn those parameters into a dictionary.

        Args:
            source (str or PipelineVariable): The source for the input. If a local path
                is provided, it will automatically be uploaded to S3 under:
                "s3://<default-bucket-name>/<job-name>/input/<input-name>".
            destination (str or PipelineVariable): The destination of the input.
            input_name (str or PipelineVariable): The name for the input. If a name
                is not provided, one will be generated (eg. "input-1").
            s3_data_type (str or PipelineVariable): Valid options are "ManifestFile" or "S3Prefix".
            s3_input_mode (str or PipelineVariable): Valid options are "Pipe" or "File".
            s3_data_distribution_type (str or PipelineVariable): Valid options are "FullyReplicated"
                or "ShardedByS3Key".
            s3_compression_type (str or PipelineVariable): Valid options are "None" or "Gzip".
            s3_input (:class:`~sagemaker.dataset_definition.inputs.S3Input`)
                Metadata of data objects stored in S3
            dataset_definition (:class:`~sagemaker.dataset_definition.inputs.DatasetDefinition`)
                DatasetDefinition input
            app_managed (bool or PipelineVariable): Whether the input are managed by SageMaker
                or application
        """
        self.source = source
        self.destination = destination
        self.input_name = input_name
        self.s3_data_type = s3_data_type
        self.s3_input_mode = s3_input_mode
        self.s3_data_distribution_type = s3_data_distribution_type
        self.s3_compression_type = s3_compression_type
        self.s3_input = s3_input
        self.dataset_definition = dataset_definition
        self.app_managed = app_managed
        self._create_s3_input()

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""

        # Create the request dictionary.
        s3_input_request = {
            "InputName": self.input_name,
            "AppManaged": self.app_managed,
        }

        if self.s3_input:
            # Check the compression type, then add it to the dictionary.
            if (
                self.s3_input.s3_compression_type == "Gzip"
                and self.s3_input.s3_input_mode != "Pipe"
            ):
                raise ValueError("Data can only be gzipped when the input mode is Pipe.")

            s3_input_request["S3Input"] = S3Input.to_boto(self.s3_input)

        if self.dataset_definition is not None:
            s3_input_request["DatasetDefinition"] = DatasetDefinition.to_boto(
                self.dataset_definition
            )

        # Return the request dictionary.
        return s3_input_request

    def _create_s3_input(self):
        """Create and initialize S3Input.

        When client provides S3Input, backfill other class memebers because they are used
        in other places. When client provides other S3Input class memebers, create and
        init S3Input.
        """

        if self.s3_input is not None:
            # backfill other class members
            self.source = self.s3_input.s3_uri
            self.destination = self.s3_input.local_path
            self.s3_data_type = self.s3_input.s3_data_type
            self.s3_input_mode = self.s3_input.s3_input_mode
            self.s3_data_distribution_type = self.s3_input.s3_data_distribution_type
        elif self.source is not None and self.destination is not None:
            self.s3_input = S3Input(
                s3_uri=self.source,
                local_path=self.destination,
                s3_data_type=self.s3_data_type,
                s3_input_mode=self.s3_input_mode,
                s3_data_distribution_type=self.s3_data_distribution_type,
                s3_compression_type=self.s3_compression_type,
            )


class ProcessingOutput(object):
    """Accepts parameters that specify an Amazon S3 output for a processing job.

    It also provides a method to turn those parameters into a dictionary.
    """

    def __init__(
        self,
        source: Optional[Union[str, PipelineVariable]] = None,
        destination: Optional[Union[str, PipelineVariable]] = None,
        output_name: Optional[Union[str, PipelineVariable]] = None,
        s3_upload_mode: Union[str, PipelineVariable] = "EndOfJob",
        app_managed: Union[bool, PipelineVariable] = False,
        feature_store_output: Optional["FeatureStoreOutput"] = None,
    ):
        """Initializes a ``ProcessingOutput`` instance.

        ``ProcessingOutput`` accepts parameters that specify an Amazon S3 output for a
        processing job and provides a method to turn those parameters into a dictionary.

        Args:
            source (str or PipelineVariable): The source for the output.
            destination (str or PipelineVariable): The destination of the output. If a destination
                is not provided, one will be generated:
                "s3://<default-bucket-name>/<job-name>/output/<output-name>"
                (Note: this does not apply when used with
                :class:`~sagemaker.workflow.steps.ProcessingStep`).
            output_name (str or PipelineVariable): The name of the output. If a name
                is not provided, one will be generated (eg. "output-1").
            s3_upload_mode (str or PipelineVariable): Valid options are "EndOfJob"
                or "Continuous".
            app_managed (bool or PipelineVariable): Whether the input are managed by SageMaker
                or application
            feature_store_output (:class:`~sagemaker.processing.FeatureStoreOutput`)
                Configuration for processing job outputs of FeatureStore.
        """
        self.source = source
        self.destination = destination
        self.output_name = output_name
        self.s3_upload_mode = s3_upload_mode
        self.app_managed = app_managed
        self.feature_store_output = feature_store_output

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        # Create the request dictionary.
        s3_output_request = {
            "OutputName": self.output_name,
            "AppManaged": self.app_managed,
        }

        if self.source is not None:
            s3_output_request["S3Output"] = {
                "S3Uri": self.destination,
                "LocalPath": self.source,
                "S3UploadMode": self.s3_upload_mode,
            }

        if self.feature_store_output is not None:
            s3_output_request["FeatureStoreOutput"] = FeatureStoreOutput.to_boto(
                self.feature_store_output
            )

        # Return the request dictionary.
        return s3_output_request


@attr.s
class RunArgs(object):
    """Accepts parameters that correspond to ScriptProcessors.

    An instance of this class is returned from the ``get_run_args()`` method on processors,
    and is used for normalizing the arguments so that they can be passed to
    :class:`~sagemaker.workflow.steps.ProcessingStep`

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

    code = attr.ib()
    inputs = attr.ib(default=None)
    outputs = attr.ib(default=None)
    arguments = attr.ib(default=None)


class FeatureStoreOutput(ApiObject):
    """Configuration for processing job outputs in Amazon SageMaker Feature Store."""

    feature_group_name = None


class FrameworkProcessor(ScriptProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using a machine learning framework."""

    framework_entrypoint_command = ["/bin/bash"]

    # Added new (kw)args for estimator. The rest are from ScriptProcessor with same defaults.
    def __init__(
        self,
        estimator_cls: type,
        framework_version: str,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        py_version: str = "py3",
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        command: Optional[List[str]] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        code_location: Optional[str] = None,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """Initializes a ``FrameworkProcessor`` instance.

        The ``FrameworkProcessor`` handles Amazon SageMaker Processing tasks for jobs
        using a machine learning framework, which allows for a set of Python scripts
        to be run as part of the Processing Job.

        Args:
            estimator_cls (type): A subclass of the :class:`~sagemaker.estimator.Framework`
                estimator
            framework_version (str): The version of the framework. Value is ignored when
                ``image_uri`` is provided.
            role (str or PipelineVariable): An AWS IAM role name or ARN. Amazon SageMaker
                Processing uses this role to access AWS resources, such as data stored
                in Amazon S3.
            instance_count (int or PipelineVariable): The number of instances to run a
                processing job with.
            instance_type (str or PipelineVariable): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            py_version (str): Python version you want to use for executing your
                model training code. One of 'py2' or 'py3'. Defaults to 'py3'. Value
                is ignored when ``image_uri`` is provided.
            image_uri (str or PipelineVariable): The URI of the Docker image to use for the
                processing jobs (default: None).
            command ([str]): The command to run, along with any command-line flags
                to *precede* the ```code script```. Example: ["python3", "-v"]. If not
                provided, ["python"] will be chosen (default: None).
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing volume
                (default: None).
            output_kms_key (str or PipelineVariable): The KMS key ID for processing job outputs
                (default: None).
            code_location (str): The S3 prefix URI where custom code will be
                uploaded (default: None). The code file uploaded to S3 is
                'code_location/job-name/source/sourcedir.tar.gz'. If not specified, the
                default ``code location`` is 's3://{sagemaker-default-bucket}'
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                processing image name and current timestamp (default: None).
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain (default: None).
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables to
                be passed to the processing jobs (default: None).
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): List of tags to
                be passed to the processing job (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets (default: None).
        """
        if not command:
            command = ["python"]

        self.estimator_cls = estimator_cls
        self.framework_version = framework_version
        self.py_version = py_version

        # 1. To finalize/normalize the image_uri or base_job_name, we need to create an
        #    estimator_cls instance.
        # 2. We want to make it easy for children of FrameworkProcessor to override estimator
        #    creation via a function (to create FrameworkProcessors for Estimators that may have
        #    different signatures - like HuggingFace or others in future).
        # 3. Super-class __init__ doesn't (currently) do anything with these params besides
        #    storing them
        #
        # Therefore we'll init the superclass first and then customize the setup after:
        super().__init__(
            role=role,
            image_uri=image_uri,
            command=command,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )

        # This subclass uses the "code" input for actual payload and the ScriptProcessor parent's
        # functionality for uploading just a small entrypoint script to invoke it.
        self._CODE_CONTAINER_INPUT_NAME = "entrypoint"

        self.code_location = (
            code_location[:-1] if (code_location and code_location.endswith("/")) else code_location
        )

        if image_uri is None or base_job_name is None:
            # For these default configuration purposes, we don't need the optional args:
            est = self._create_estimator()
            if image_uri is None:
                self.image_uri = est.training_image_uri()
            if base_job_name is None:
                self.base_job_name = est.base_job_name or estimator_cls._framework_name
                if base_job_name is None:
                    base_job_name = "framework-processor"

    def _create_estimator(
        self,
        entry_point="",
        source_dir=None,
        dependencies=None,
        git_config=None,
    ):
        """Instantiate the Framework Estimator that backs this Processor"""
        return self.estimator_cls(
            framework_version=self.framework_version,
            py_version=self.py_version,
            entry_point=entry_point,
            source_dir=source_dir,
            dependencies=dependencies,
            git_config=git_config,
            code_location=self.code_location,
            enable_network_isolation=False,  # True -> uploads to input channel. Not what we want!
            image_uri=self.image_uri,
            role=self.role,
            # Estimator instance_count doesn't currently matter to FrameworkProcessor, and the
            # SKLearn Framework Estimator requires instance_type==1. So here we hard-wire it to 1,
            # but if it matters in future perhaps we could take self.instance_count here and have
            # SKLearnProcessor override this function instead:
            instance_count=1,
            instance_type=self.instance_type,
            sagemaker_session=self.sagemaker_session,
            debugger_hook_config=False,
            disable_profiler=True,
            output_kms_key=self.output_kms_key,
        )

    def get_run_args(
        self,
        code,
        source_dir=None,
        dependencies=None,
        git_config=None,
        inputs=None,
        outputs=None,
        arguments=None,
        job_name=None,
    ):
        """Returns a RunArgs object.

        This object contains the normalized inputs, outputs and arguments needed
        when using a ``FrameworkProcessor`` in a :class:`~sagemaker.workflow.steps.ProcessingStep`.

        Args:
            code (str): This can be an S3 URI or a local path to a file with the framework
                script to run. See the ``code`` argument in
                `sagemaker.processing.FrameworkProcessor.run()`.
            source_dir (str): Path (absolute, relative, or an S3 URI) to a directory wit
                any other processing source code dependencies aside from the entrypoint
                file (default: None). See the ``source_dir`` argument in
                `sagemaker.processing.FrameworkProcessor.run()`
            dependencies (list[str]): A list of paths to directories (absolute or relative)
                with any additional libraries that will be exported to the container
                (default: []). See the ``dependencies`` argument in
                `sagemaker.processing.FrameworkProcessor.run()`.
            git_config (dict[str, str]): Git configurations used for cloning files. See the
                `git_config` argument in `sagemaker.processing.FrameworkProcessor.run()`.
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
        """
        logger.warning(
            "This function has been deprecated and could break pipeline step caching. "
            "We recommend using the run() function directly with pipeline sessions"
            "to access step arguments."
        )

        # When job_name is None, the job_name to upload code (+payload) will
        # differ from job_name used by run().
        s3_runproc_sh, inputs, job_name = self._pack_and_upload_code(
            code, source_dir, dependencies, git_config, job_name, inputs
        )

        return RunArgs(
            s3_runproc_sh,
            inputs=inputs,
            outputs=outputs,
            arguments=arguments,
        )

    @runnable_by_pipeline
    def run(  # type: ignore[override]
        self,
        code: str,
        source_dir: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        inputs: Optional[List[ProcessingInput]] = None,
        outputs: Optional[List[ProcessingOutput]] = None,
        arguments: Optional[List[Union[str, PipelineVariable]]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
    ):
        """Runs a processing job.

        Args:
            code (str): This can be an S3 URI or a local path to a file with the
                framework script to run.Path (absolute or relative) to the local
                Python source file which should be executed as the entry point
                to training. When `code` is an S3 URI, ignore `source_dir`,
                `dependencies`, and `git_config`. If ``source_dir`` is specified,
                then ``code`` must point to a file located at the root of ``source_dir``.
            source_dir (str): Path (absolute, relative or an S3 URI) to a directory
                with any other processing source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a file named `sourcedir.tar.gz`. Structure within this directory
                are preserved when processing on Amazon SageMaker (default: None).
            dependencies (list[str]): A list of paths to directories (absolute
                or relative) with any additional libraries that will be exported
                to the container (default: []). The library folders will be
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo (default: None).
            git_config (dict[str, str]): Git configurations used for cloning
                files, including ``repo``, ``branch``, ``commit``,
                ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                ``repo`` field is required. All other fields are optional.
                ``repo`` specifies the Git repository where your training script
                is stored. If you don't provide ``branch``, the default value
                'master' is used. If you don't provide ``commit``, the latest
                commit in the specified branch is used. .. admonition:: Example

                    The following config:

                    >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                    >>>               'branch': 'test-branch-git-config',
                    >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

                    results in cloning the repo specified in 'repo', then
                    checkout the 'master' branch, and checkout the specified
                    commit.

                ``2FA_enabled``, ``username``, ``password`` and ``token`` are
                used for authentication. For GitHub (or other Git) accounts, set
                ``2FA_enabled`` to 'True' if two-factor authentication is
                enabled for the account, otherwise set it to 'False'. If you do
                not provide a value for ``2FA_enabled``, a default value of
                'False' is used. CodeCommit does not support two-factor
                authentication, so do not provide "2FA_enabled" with CodeCommit
                repositories.

                For GitHub and other Git repos, when SSH URLs are provided, it
                doesn't matter whether 2FA is enabled or disabled; you should
                either have no passphrase for the SSH key pairs, or have the
                ssh-agent configured so that you will not be prompted for SSH
                passphrase when you do 'git clone' command with SSH URLs. When
                HTTPS URLs are provided: if 2FA is disabled, then either token
                or username+password will be used for authentication if provided
                (token prioritized); if 2FA is enabled, only token will be used
                for authentication if provided. If required authentication info
                is not provided, python SDK will try to use local credentials
                storage to authenticate. If that fails either, an error message
                will be thrown.

                For CodeCommit repos, 2FA is not supported, so '2FA_enabled'
                should not be provided. There is no token in CodeCommit, so
                'token' should not be provided too. When 'repo' is an SSH URL,
                the requirements are the same as GitHub-like repos. When 'repo'
                is an HTTPS URL, username+password will be used for
                authentication if they are provided; otherwise, python SDK will
                try to use either CodeCommit credential helper or local
                credential storage for authentication.
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job. These can be specified as either path strings or
                :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
            arguments (list[str] or list[PipelineVariable]): A list of string arguments
                to be passed to a processing job (default: None).
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
                * Both `ExperimentName` and `TrialName` will be ignored if the Processor instance
                is built with :class:`~sagemaker.workflow.pipeline_context.PipelineSession`.
                However, the value of `TrialComponentDisplayName` is honored for display in Studio.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        Returns:
            None or pipeline step arguments in case the Processor instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        """
        s3_runproc_sh, inputs, job_name = self._pack_and_upload_code(
            code, source_dir, dependencies, git_config, job_name, inputs, kms_key
        )

        # Submit a processing job.
        return super().run(
            code=s3_runproc_sh,
            inputs=inputs,
            outputs=outputs,
            arguments=arguments,
            wait=wait,
            logs=logs,
            job_name=job_name,
            experiment_config=experiment_config,
            kms_key=kms_key,
        )

    def _pack_and_upload_code(
        self, code, source_dir, dependencies, git_config, job_name, inputs, kms_key=None
    ):
        """Pack local code bundle and upload to Amazon S3."""
        if code.startswith("s3://"):
            return code, inputs, job_name

        if job_name is None:
            job_name = self._generate_current_job_name(job_name)

        estimator = self._upload_payload(
            code,
            source_dir,
            dependencies,
            git_config,
            job_name,
        )
        inputs = self._patch_inputs_with_payload(
            inputs,
            estimator._hyperparameters["sagemaker_submit_directory"],
        )

        local_code = get_config_value("local.local_code", self.sagemaker_session.config)
        if self.sagemaker_session.local_mode and local_code:
            raise RuntimeError(
                "SageMaker Processing Local Mode does not currently support 'local code' mode. "
                "Please use a LocalSession created with disable_local_code=True, or leave "
                "sagemaker_session unspecified when creating your Processor to have one set up "
                "automatically."
            )
        if "/sourcedir.tar.gz" in estimator.uploaded_code.s3_prefix:
            # Upload the bootstrapping code as s3://.../jobname/source/runproc.sh.
            entrypoint_s3_uri = estimator.uploaded_code.s3_prefix.replace(
                "sourcedir.tar.gz",
                "runproc.sh",
            )
        else:
            raise RuntimeError("S3 source_dir file must be named `sourcedir.tar.gz.`")

        script = estimator.uploaded_code.script_name
        evaluated_kms_key = kms_key if kms_key else self.output_kms_key
        s3_runproc_sh = self._create_and_upload_runproc(
            script, evaluated_kms_key, entrypoint_s3_uri
        )

        return s3_runproc_sh, inputs, job_name

    def _generate_framework_script(self, user_script: str) -> str:
        """Generate the framework entrypoint file (as text) for a processing job.

        This script implements the "framework" functionality for setting up your code:
        Untar-ing the sourcedir bundle in the ```code``` input; installing extra
        runtime dependencies if specified; and then invoking the ```command``` and
        ```code``` configured for the job.

        Args:
            user_script (str): Relative path to ```code``` in the source bundle
                - e.g. 'process.py'.
        """
        return dedent(
            """\
            #!/bin/bash

            cd /opt/ml/processing/input/code/
            tar -xzf sourcedir.tar.gz

            # Exit on any error. SageMaker uses error code to mark failed job.
            set -e

            if [[ -f 'requirements.txt' ]]; then
                # Some py3 containers has typing, which may breaks pip install
                pip uninstall --yes typing

                pip install -r requirements.txt
            fi

            {entry_point_command} {entry_point} "$@"
        """
        ).format(
            entry_point_command=" ".join(self.command),
            entry_point=user_script,
        )

    def _upload_payload(
        self,
        entry_point: str,
        source_dir: Optional[str],
        dependencies: Optional[List[str]],
        git_config: Optional[Dict[str, str]],
        job_name: str,
    ) -> "sagemaker.estimator.Framework":  # type: ignore[name-defined]   # noqa: F821
        """Upload payload sourcedir.tar.gz to S3."""
        # A new estimator instance is required, because each call to ScriptProcessor.run() can
        # use different codes.
        estimator = self._create_estimator(
            entry_point=entry_point,
            source_dir=source_dir,
            dependencies=dependencies,
            git_config=git_config,
        )

        estimator._prepare_for_training(job_name=job_name)
        logger.info(
            "Uploaded %s to %s",
            estimator.source_dir,
            estimator._hyperparameters["sagemaker_submit_directory"],
        )

        return estimator

    def _patch_inputs_with_payload(self, inputs, s3_payload) -> List[ProcessingInput]:
        """Add payload sourcedir.tar.gz to processing input.

        This method follows the same mechanism in ScriptProcessor.
        """
        # Follow the exact same mechanism that ScriptProcessor does, which
        # is to inject the S3 code artifact as a processing input. Note that
        # framework processor take-over /opt/ml/processing/input/code for
        # sourcedir.tar.gz, and let ScriptProcessor to place runproc.sh under
        # /opt/ml/processing/input/{self._CODE_CONTAINER_INPUT_NAME}.
        #
        # See:
        # - ScriptProcessor._CODE_CONTAINER_BASE_PATH, ScriptProcessor._CODE_CONTAINER_INPUT_NAME.
        # - https://github.com/aws/sagemaker-python-sdk/blob/ \
        #   a7399455f5386d83ddc5cb15c0db00c04bd518ec/src/sagemaker/processing.py#L425-L426
        if inputs is None:
            inputs = []

        # make a shallow copy of user inputs
        patched_inputs = copy(inputs)
        patched_inputs.append(
            ProcessingInput(
                input_name="code",
                source=s3_payload,
                destination="/opt/ml/processing/input/code/",
            )
        )
        return patched_inputs

    def _set_entrypoint(self, command, user_script_name):
        """Framework processor override for setting processing job entrypoint.

        Args:
            command ([str]): Ignored in favor of self.framework_entrypoint_command
            user_script_name (str): A filename with an extension.
        """

        user_script_location = str(
            pathlib.PurePosixPath(
                self._CODE_CONTAINER_BASE_PATH, self._CODE_CONTAINER_INPUT_NAME, user_script_name
            )
        )
        self.entrypoint = self.framework_entrypoint_command + [user_script_location]

    def _create_and_upload_runproc(self, user_script, kms_key, entrypoint_s3_uri):
        """Create runproc shell script and upload to S3 bucket.

        If leveraging a pipeline session with optimized S3 artifact paths,
        the runproc.sh file is hashed and uploaded to a separate S3 location.


        Args:
            user_script (str): Relative path to ```code``` in the source bundle
                - e.g. 'process.py'.
            kms_key (str): THe kms key used for encryption.
            entrypoint_s3_uri (str): The S3 upload path for the runproc script.
        """
        from sagemaker.workflow.utilities import _pipeline_config, hash_object

        if _pipeline_config and _pipeline_config.pipeline_name:
            runproc_file_str = self._generate_framework_script(user_script)
            runproc_file_hash = hash_object(runproc_file_str)
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                _pipeline_config.pipeline_name,
                "code",
                runproc_file_hash,
                "runproc.sh",
            )
            s3_runproc_sh = S3Uploader.upload_string_as_file_body(
                runproc_file_str,
                desired_s3_uri=s3_uri,
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        else:
            s3_runproc_sh = S3Uploader.upload_string_as_file_body(
                self._generate_framework_script(user_script),
                desired_s3_uri=entrypoint_s3_uri,
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        logger.info("runproc.sh uploaded to %s", s3_runproc_sh)

        return s3_runproc_sh
