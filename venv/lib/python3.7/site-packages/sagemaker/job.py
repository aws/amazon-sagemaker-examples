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

from abc import abstractmethod
from six import string_types

from sagemaker.inputs import FileSystemInput, TrainingInput
from sagemaker.local import file_input
from sagemaker.workflow import is_pipeline_variable


class _Job(object):
    """Handle creating, starting and waiting for Amazon SageMaker jobs to finish.

    This class shouldn't be directly instantiated.

    Subclasses must define a way to create, start and wait for an Amazon
    SageMaker job.
    """

    def __init__(self, sagemaker_session, job_name):
        """Placeholder docstring"""
        self.sagemaker_session = sagemaker_session
        self.job_name = job_name

    @abstractmethod
    def start_new(self, estimator, inputs):
        """Create a new Amazon SageMaker job from the estimator.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): Estimator object
                created by the user.
            inputs (str): Parameters used when called
                :meth:`~sagemaker.estimator.EstimatorBase.fit`.

        Returns:
            sagemaker.job: Constructed object that captures all information
            about the started job.
        """

    @abstractmethod
    def wait(self):
        """Wait for the Amazon SageMaker job to finish."""

    @abstractmethod
    def describe(self):
        """Describe the job."""

    @abstractmethod
    def stop(self):
        """Stop the job."""

    @staticmethod
    def _load_config(inputs, estimator, expand_role=True, validate_uri=True):
        """Placeholder docstring"""
        input_config = _Job._format_inputs_to_input_config(inputs, validate_uri)
        role = (
            estimator.sagemaker_session.expand_role(estimator.role)
            if (expand_role and not is_pipeline_variable(estimator.role))
            else estimator.role
        )
        output_config = _Job._prepare_output_config(
            estimator.output_path,
            estimator.output_kms_key,
            disable_output_compression=estimator.disable_output_compression,
        )
        resource_config = _Job._prepare_resource_config(
            estimator.instance_count,
            estimator.instance_type,
            estimator.instance_groups,
            estimator.volume_size,
            estimator.volume_kms_key,
            estimator.keep_alive_period_in_seconds,
        )
        stop_condition = _Job._prepare_stop_condition(estimator.max_run, estimator.max_wait)
        vpc_config = estimator.get_vpc_config()

        model_channel = _Job._prepare_channel(
            input_config,
            estimator.model_uri,
            estimator.model_channel_name,
            validate_uri,
            content_type="application/x-sagemaker-model",
            input_mode="File",
        )
        if model_channel:
            input_config = [] if input_config is None else input_config
            input_config.append(model_channel)

        if estimator.enable_network_isolation():
            code_channel = _Job._prepare_channel(
                input_config, estimator.code_uri, estimator.code_channel_name, validate_uri
            )

            if code_channel:
                input_config = [] if input_config is None else input_config
                input_config.append(code_channel)

        return {
            "input_config": input_config,
            "role": role,
            "output_config": output_config,
            "resource_config": resource_config,
            "stop_condition": stop_condition,
            "vpc_config": vpc_config,
        }

    @staticmethod
    def _format_inputs_to_input_config(inputs, validate_uri=True):
        """Placeholder docstring"""
        if inputs is None:
            return None

        # Deferred import due to circular dependency
        from sagemaker.amazon.amazon_estimator import RecordSet
        from sagemaker.amazon.amazon_estimator import FileSystemRecordSet

        if isinstance(inputs, (RecordSet, FileSystemRecordSet)):
            inputs = inputs.data_channel()

        input_dict = {}
        if isinstance(inputs, string_types):
            input_dict["training"] = _Job._format_string_uri_input(inputs, validate_uri)
        elif isinstance(inputs, TrainingInput):
            input_dict["training"] = inputs
        elif isinstance(inputs, file_input):
            input_dict["training"] = inputs
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                input_dict[k] = _Job._format_string_uri_input(v, validate_uri)
        elif isinstance(inputs, list):
            input_dict = _Job._format_record_set_list_input(inputs)
        elif isinstance(inputs, FileSystemInput):
            input_dict["training"] = inputs
        else:
            msg = (
                "Cannot format input {}. Expecting one of str, dict, TrainingInput or "
                "FileSystemInput"
            )
            raise ValueError(msg.format(inputs))

        channels = [
            _Job._convert_input_to_channel(name, input) for name, input in input_dict.items()
        ]

        return channels

    @staticmethod
    def _convert_input_to_channel(channel_name, channel_s3_input):
        """Placeholder docstring"""
        channel_config = channel_s3_input.config.copy()
        channel_config["ChannelName"] = channel_name
        return channel_config

    @staticmethod
    def _format_string_uri_input(
        uri_input,
        validate_uri=True,
        content_type=None,
        input_mode=None,
        compression=None,
        target_attribute_name=None,
    ):
        """Placeholder docstring"""
        s3_input_result = TrainingInput(
            uri_input,
            content_type=content_type,
            input_mode=input_mode,
            compression=compression,
            target_attribute_name=target_attribute_name,
        )
        if isinstance(uri_input, str) and validate_uri and uri_input.startswith("s3://"):
            return s3_input_result
        if isinstance(uri_input, str) and validate_uri and uri_input.startswith("file://"):
            return file_input(uri_input)
        if isinstance(uri_input, str) and validate_uri:
            raise ValueError(
                'URI input {} must be a valid S3 or FILE URI: must start with "s3://" or '
                '"file://"'.format(uri_input)
            )
        if isinstance(uri_input, str):
            return s3_input_result
        if isinstance(uri_input, (TrainingInput, file_input, FileSystemInput)):
            return uri_input
        if is_pipeline_variable(uri_input):
            return s3_input_result

        raise ValueError(
            "Cannot format input {}. Expecting one of str, TrainingInput, file_input or "
            "FileSystemInput".format(uri_input)
        )

    @staticmethod
    def _prepare_channel(
        input_config,
        channel_uri=None,
        channel_name=None,
        validate_uri=True,
        content_type=None,
        input_mode=None,
    ):
        """Placeholder docstring"""
        if not channel_uri:
            return None
        if not channel_name:
            raise ValueError(
                "Expected a channel name if a channel URI {} is specified".format(channel_uri)
            )

        if input_config:
            for existing_channel in input_config:
                if existing_channel["ChannelName"] == channel_name:
                    raise ValueError("Duplicate channel {} not allowed.".format(channel_name))

        channel_input = _Job._format_string_uri_input(
            channel_uri, validate_uri, content_type, input_mode
        )
        channel = _Job._convert_input_to_channel(channel_name, channel_input)

        return channel

    @staticmethod
    def _format_model_uri_input(model_uri, validate_uri=True):
        """Placeholder docstring"""
        if isinstance(model_uri, string_types) and validate_uri and model_uri.startswith("s3://"):
            return TrainingInput(
                model_uri,
                input_mode="File",
                distribution="FullyReplicated",
                content_type="application/x-sagemaker-model",
            )
        if isinstance(model_uri, string_types) and validate_uri and model_uri.startswith("file://"):
            return file_input(model_uri)
        if isinstance(model_uri, string_types) and validate_uri:
            raise ValueError(
                'Model URI must be a valid S3 or FILE URI: must start with "s3://" or ' '"file://'
            )
        if isinstance(model_uri, string_types):
            return TrainingInput(
                model_uri,
                input_mode="File",
                distribution="FullyReplicated",
                content_type="application/x-sagemaker-model",
            )
        raise ValueError("Cannot format model URI {}. Expecting str".format(model_uri))

    @staticmethod
    def _format_record_set_list_input(inputs):
        """Placeholder docstring"""
        # Deferred import due to circular dependency
        from sagemaker.amazon.amazon_estimator import FileSystemRecordSet, RecordSet

        input_dict = {}
        for record in inputs:
            if not isinstance(record, (RecordSet, FileSystemRecordSet)):
                raise ValueError("List compatible only with RecordSets or FileSystemRecordSets.")

            if record.channel in input_dict:
                raise ValueError("Duplicate channels not allowed.")
            if isinstance(record, RecordSet):
                input_dict[record.channel] = record.records_s3_input()
            if isinstance(record, FileSystemRecordSet):
                input_dict[record.channel] = record.file_system_input

        return input_dict

    @staticmethod
    def _prepare_output_config(s3_path, kms_key_id, disable_output_compression=False):
        """Placeholder docstring"""
        config = {"S3OutputPath": s3_path}
        if kms_key_id is not None:
            config["KmsKeyId"] = kms_key_id
        if disable_output_compression:
            config["CompressionType"] = "NONE"
        return config

    @staticmethod
    def _prepare_resource_config(
        instance_count,
        instance_type,
        instance_groups,
        volume_size,
        volume_kms_key,
        keep_alive_period_in_seconds,
    ):
        """Placeholder docstring"""
        resource_config = {
            "VolumeSizeInGB": volume_size,
        }
        if volume_kms_key is not None:
            resource_config["VolumeKmsKeyId"] = volume_kms_key
        if keep_alive_period_in_seconds is not None:
            resource_config["KeepAlivePeriodInSeconds"] = keep_alive_period_in_seconds
        if instance_groups is not None:
            if instance_count is not None or instance_type is not None:
                raise ValueError(
                    "instance_count and instance_type cannot be set when instance_groups is set"
                )

            resource_config["InstanceGroups"] = [
                group._to_request_dict() for group in instance_groups
            ]
        else:
            if instance_count is None or instance_type is None:
                raise ValueError(
                    "instance_count and instance_type must be set if instance_groups is not set"
                )
            resource_config["InstanceCount"] = instance_count
            resource_config["InstanceType"] = instance_type

        return resource_config

    @staticmethod
    def _prepare_stop_condition(max_run, max_wait):
        """Placeholder docstring"""
        if max_wait:
            return {"MaxRuntimeInSeconds": max_run, "MaxWaitTimeInSeconds": max_wait}
        return {"MaxRuntimeInSeconds": max_run}

    @property
    def name(self):
        """Placeholder docstring"""
        return self.job_name
