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
"""This module contains code related to Amazon SageMaker Model Monitoring.

These classes assist with suggesting baselines and creating monitoring schedules for
data captured by SageMaker Endpoints.
"""
from __future__ import print_function, absolute_import

import copy
import json
import os
import pathlib
import logging
import uuid
from typing import Union, Optional, Dict, List
import attr

from six import string_types
from six.moves.urllib.parse import urlparse
from botocore.exceptions import ClientError

from sagemaker import image_uris, s3
from sagemaker.config.config_schema import (
    SAGEMAKER,
    MONITORING_SCHEDULE,
    TAGS,
    MONITORING_JOB_SUBNETS_PATH,
    MONITORING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    MONITORING_JOB_ENVIRONMENT_PATH,
    MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH,
    MONITORING_JOB_VOLUME_KMS_KEY_ID_PATH,
    MONITORING_JOB_SECURITY_GROUP_IDS_PATH,
    MONITORING_JOB_OUTPUT_KMS_KEY_ID_PATH,
    MONITORING_JOB_ROLE_ARN_PATH,
)
from sagemaker.exceptions import UnexpectedStatusException
from sagemaker.model_monitor.monitoring_files import Constraints, ConstraintViolations, Statistics
from sagemaker.model_monitor.monitoring_alert import (
    MonitoringAlertSummary,
    MonitoringAlertHistorySummary,
    MonitoringAlertActions,
    ModelDashboardIndicatorAction,
)
from sagemaker.model_monitor.data_quality_monitoring_config import DataQualityMonitoringConfig
from sagemaker.model_monitor.dataset_format import MonitoringDatasetFormat
from sagemaker.network import NetworkConfig
from sagemaker.processing import Processor, ProcessingInput, ProcessingJob, ProcessingOutput
from sagemaker.session import Session
from sagemaker.utils import (
    name_from_base,
    retries,
    resolve_value_from_config,
    resolve_class_attribute_from_config,
)
from sagemaker.lineage._utils import get_resource_name_from_arn

DEFAULT_REPOSITORY_NAME = "sagemaker-model-monitor-analyzer"

STATISTICS_JSON_DEFAULT_FILE_NAME = "statistics.json"
CONSTRAINTS_JSON_DEFAULT_FILE_NAME = "constraints.json"
CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME = "constraint_violations.json"

_CONTAINER_BASE_PATH = "/opt/ml/processing"
_CONTAINER_INPUT_PATH = "input"
_CONTAINER_ENDPOINT_INPUT_PATH = "endpoint"
_BASELINE_DATASET_INPUT_NAME = "baseline_dataset_input"
_RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME = "record_preprocessor_script_input"
_POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME = "post_analytics_processor_script_input"
_CONTAINER_OUTPUT_PATH = "output"
_DEFAULT_OUTPUT_NAME = "monitoring_output"
_MODEL_MONITOR_S3_PATH = "model-monitor"
_BASELINING_S3_PATH = "baselining"
_MONITORING_S3_PATH = "monitoring"
_RESULTS_S3_PATH = "results"
_INPUT_S3_PATH = "input"

_SUGGESTION_JOB_BASE_NAME = "baseline-suggestion-job"
_MONITORING_SCHEDULE_BASE_NAME = "monitoring-schedule"

_DATASET_SOURCE_PATH_ENV_NAME = "dataset_source"
_DATASET_FORMAT_ENV_NAME = "dataset_format"
_OUTPUT_PATH_ENV_NAME = "output_path"
_RECORD_PREPROCESSOR_SCRIPT_ENV_NAME = "record_preprocessor_script"
_POST_ANALYTICS_PROCESSOR_SCRIPT_ENV_NAME = "post_analytics_processor_script"
_PUBLISH_CLOUDWATCH_METRICS_ENV_NAME = "publish_cloudwatch_metrics"
_ANALYSIS_TYPE_ENV_NAME = "analysis_type"
_PROBLEM_TYPE_ENV_NAME = "problem_type"
_GROUND_TRUTH_ATTRIBUTE_ENV_NAME = "ground_truth_attribute"
_INFERENCE_ATTRIBUTE_ENV_NAME = "inference_attribute"
_PROBABILITY_ATTRIBUTE_ENV_NAME = "probability_attribute"
_PROBABILITY_THRESHOLD_ATTRIBUTE_ENV_NAME = "probability_threshold_attribute"
_CATEGORICAL_DRIFT_METHOD_ENV_NAME = "categorical_drift_method"

_LOGGER = logging.getLogger(__name__)

framework_name = "model-monitor"


class ModelMonitor(object):
    """Sets up Amazon SageMaker Monitoring Schedules and baseline suggestions.

    Use this class when you want to provide your own container image containing the code
    you'd like to run, in order to produce your own statistics and constraint validation files.
    For a more guided experience, consider using the DefaultModelMonitor class instead.
    """

    def __init__(
        self,
        role=None,
        image_uri=None,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        entrypoint=None,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initializes a ``Monitor`` instance.

        The Monitor handles baselining datasets and creating Amazon SageMaker Monitoring Schedules
        to monitor SageMaker endpoints.

        Args:
            role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
            image_uri (str): The uri of the image to use for the jobs started by
                the Monitor.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            entrypoint ([str]): The entrypoint for the job.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            base_job_name (str): Prefix for the job name. If not specified,
                a default name is generated based on the training image name and
                current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            env (dict): Environment variables to be passed to the job.
            tags ([dict]): List of tags to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
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
        self.sagemaker_session = sagemaker_session or Session()
        self.tags = tags

        self.baselining_jobs = []
        self.latest_baselining_job = None
        self.arguments = None
        self.latest_baselining_job_name = None
        self.monitoring_schedule_name = None
        self.job_definition_name = None
        self.role = resolve_value_from_config(
            role, MONITORING_JOB_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        if not self.role:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create a Monitoring Schedule.")
        self.volume_kms_key = resolve_value_from_config(
            volume_kms_key,
            MONITORING_JOB_VOLUME_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.output_kms_key = resolve_value_from_config(
            output_kms_key,
            MONITORING_JOB_OUTPUT_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            network_config,
            "subnets",
            MONITORING_JOB_SUBNETS_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "security_group_ids",
            MONITORING_JOB_SECURITY_GROUP_IDS_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "enable_network_isolation",
            MONITORING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.network_config = resolve_class_attribute_from_config(
            NetworkConfig,
            self.network_config,
            "encrypt_inter_container_traffic",
            MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.env = resolve_value_from_config(
            env,
            MONITORING_JOB_ENVIRONMENT_PATH,
            default_value=None,
            sagemaker_session=self.sagemaker_session,
        )

    def run_baseline(
        self, baseline_inputs, output, arguments=None, wait=True, logs=True, job_name=None
    ):
        """Run a processing job meant to baseline your dataset.

        Args:
            baseline_inputs ([sagemaker.processing.ProcessingInput]): Input files for the processing
                job. These must be provided as ProcessingInput objects.
            output (sagemaker.processing.ProcessingOutput): Destination of the constraint_violations
                and statistics json files.
            arguments ([str]): A list of string arguments to be passed to a processing job.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.

        """
        self.latest_baselining_job_name = self._generate_baselining_job_name(job_name=job_name)
        self.arguments = arguments
        normalized_baseline_inputs = self._normalize_baseline_inputs(
            baseline_inputs=baseline_inputs
        )
        normalized_output = self._normalize_processing_output(output=output)

        baselining_processor = Processor(
            role=self.role,
            image_uri=self.image_uri,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            entrypoint=self.entrypoint,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            output_kms_key=self.output_kms_key,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            base_job_name=self.base_job_name,
            sagemaker_session=self.sagemaker_session,
            env=self.env,
            tags=self.tags,
            network_config=self.network_config,
        )

        baselining_processor.run(
            inputs=normalized_baseline_inputs,
            outputs=[normalized_output],
            arguments=self.arguments,
            wait=wait,
            logs=logs,
            job_name=self.latest_baselining_job_name,
        )

        self.latest_baselining_job = BaseliningJob.from_processing_job(
            processing_job=baselining_processor.latest_job
        )
        self.baselining_jobs.append(self.latest_baselining_job)

    def create_monitoring_schedule(
        self,
        endpoint_input=None,
        output=None,
        statistics=None,
        constraints=None,
        monitor_schedule_name=None,
        schedule_cron_expression=None,
        batch_transform_input=None,
        arguments=None,
    ):
        """Creates a monitoring schedule to monitor an Amazon SageMaker Endpoint.

        If constraints and statistics are provided, or if they are able to be retrieved from a
        previous baselining job associated with this monitor, those will be used.
        If constraints and statistics cannot be automatically retrieved, baseline_inputs will be
        required in order to kick off a baselining job.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput. (default: None)
            output (sagemaker.model_monitor.MonitoringOutput): The output of the monitoring
                schedule. (default: None)
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Statistic object or an S3 uri pointing to a statistic
                JSON file. (default: None)
            constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
                statistics, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file. (default: None)
            monitor_schedule_name (str): Schedule name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp. (default: None)
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job runs at. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily. (default: None)
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
                run the monitoring schedule on the batch transform
                (default: None)
            arguments ([str]): A list of string arguments to be passed to a processing job.

        """
        if self.monitoring_schedule_name is not None:
            message = (
                "It seems that this object was already used to create an Amazon Model "
                "Monitoring Schedule. To create another, first delete the existing one "
                "using my_monitor.delete_monitoring_schedule()."
            )
            print(message)
            raise ValueError(message)

        if not output:
            raise ValueError("output can not be None.")

        if (batch_transform_input is not None) ^ (endpoint_input is None):
            message = (
                "Need to have either batch_transform_input or endpoint_input to create an "
                "Amazon Model Monitoring Schedule. "
                "Please provide only one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        self.monitoring_schedule_name = self._generate_monitoring_schedule_name(
            schedule_name=monitor_schedule_name
        )

        if batch_transform_input is not None:
            normalized_monitoring_input = batch_transform_input._to_request_dict()
        else:
            normalized_monitoring_input = self._normalize_endpoint_input(
                endpoint_input=endpoint_input
            )._to_request_dict()

        normalized_monitoring_output = self._normalize_monitoring_output_fields(output=output)

        statistics_object, constraints_object = self._get_baseline_files(
            statistics=statistics, constraints=constraints, sagemaker_session=self.sagemaker_session
        )

        statistics_s3_uri = None
        if statistics_object is not None:
            statistics_s3_uri = statistics_object.file_s3_uri

        constraints_s3_uri = None
        if constraints_object is not None:
            constraints_s3_uri = constraints_object.file_s3_uri

        monitoring_output_config = {
            "MonitoringOutputs": [normalized_monitoring_output._to_request_dict()]
        }

        if self.output_kms_key is not None:
            monitoring_output_config["KmsKeyId"] = self.output_kms_key

        self.monitoring_schedule_name = (
            monitor_schedule_name
            or self._generate_monitoring_schedule_name(schedule_name=monitor_schedule_name)
        )

        network_config_dict = None
        if self.network_config is not None:
            network_config_dict = self.network_config._to_request_dict()

        if arguments is not None:
            self.arguments = arguments

        self.sagemaker_session.create_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name,
            schedule_expression=schedule_cron_expression,
            statistics_s3_uri=statistics_s3_uri,
            constraints_s3_uri=constraints_s3_uri,
            monitoring_inputs=[normalized_monitoring_input],
            monitoring_output_config=monitoring_output_config,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            image_uri=self.image_uri,
            entrypoint=self.entrypoint,
            arguments=self.arguments,
            record_preprocessor_source_uri=None,
            post_analytics_processor_source_uri=None,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            environment=self.env,
            network_config=network_config_dict,
            role_arn=self.sagemaker_session.expand_role(self.role),
            tags=self.tags,
        )

    def update_monitoring_schedule(
        self,
        endpoint_input=None,
        output=None,
        statistics=None,
        constraints=None,
        schedule_cron_expression=None,
        instance_count=None,
        instance_type=None,
        entrypoint=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        output_kms_key=None,
        arguments=None,
        max_runtime_in_seconds=None,
        env=None,
        network_config=None,
        role=None,
        image_uri=None,
        batch_transform_input=None,
    ):
        """Updates the existing monitoring schedule.

        If more options than schedule_cron_expression are to be updated, a new job definition will
        be created to hold them. The old job definition will not be deleted.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            output (sagemaker.model_monitor.MonitoringOutput): The output of the monitoring
                schedule.
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Statistics object or an S3 uri pointing to a statistics
                JSON file.
            constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
                statistics, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job runs at.  See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            entrypoint (str): The entrypoint for the job.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            arguments ([str]): A list of string arguments to be passed to a processing job.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            env (dict): Environment variables to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.
            image_uri (str): The uri of the image to use for the jobs started by
                the Monitor.
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
                run the monitoring schedule on the batch transform (default: None)

        """
        monitoring_inputs = None

        if (batch_transform_input is not None) and (endpoint_input is not None):
            message = (
                "Cannot update both batch_transform_input and endpoint_input to update an "
                "Amazon Model Monitoring Schedule. "
                "Please provide atmost one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        if endpoint_input is not None:
            monitoring_inputs = [
                self._normalize_endpoint_input(endpoint_input=endpoint_input)._to_request_dict()
            ]

        elif batch_transform_input is not None:
            monitoring_inputs = [batch_transform_input._to_request_dict()]

        monitoring_output_config = None
        if output is not None:
            normalized_monitoring_output = self._normalize_monitoring_output_fields(output=output)
            monitoring_output_config = {
                "MonitoringOutputs": [normalized_monitoring_output._to_request_dict()]
            }

        statistics_object, constraints_object = self._get_baseline_files(
            statistics=statistics, constraints=constraints, sagemaker_session=self.sagemaker_session
        )

        statistics_s3_uri = None
        if statistics_object is not None:
            statistics_s3_uri = statistics_object.file_s3_uri

        constraints_s3_uri = None
        if constraints_object is not None:
            constraints_s3_uri = constraints_object.file_s3_uri

        if instance_type is not None:
            self.instance_type = instance_type

        if instance_count is not None:
            self.instance_count = instance_count

        if entrypoint is not None:
            self.entrypoint = entrypoint

        if volume_size_in_gb is not None:
            self.volume_size_in_gb = volume_size_in_gb

        if volume_kms_key is not None:
            self.volume_kms_key = volume_kms_key

        if output_kms_key is not None:
            self.output_kms_key = output_kms_key
            monitoring_output_config["KmsKeyId"] = self.output_kms_key

        if arguments is not None:
            self.arguments = arguments

        if max_runtime_in_seconds is not None:
            self.max_runtime_in_seconds = max_runtime_in_seconds

        if env is not None:
            self.env = env

        if network_config is not None:
            self.network_config = network_config

        if role is not None:
            self.role = role

        if image_uri is not None:
            self.image_uri = image_uri

        network_config_dict = None
        if self.network_config is not None:
            network_config_dict = self.network_config._to_request_dict()
        # Do not need to check config because that check is done inside
        # self.sagemaker_session.update_monitoring_schedule

        self.sagemaker_session.update_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name,
            schedule_expression=schedule_cron_expression,
            statistics_s3_uri=statistics_s3_uri,
            constraints_s3_uri=constraints_s3_uri,
            monitoring_inputs=monitoring_inputs,
            monitoring_output_config=monitoring_output_config,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            image_uri=image_uri,
            entrypoint=entrypoint,
            arguments=arguments,
            max_runtime_in_seconds=max_runtime_in_seconds,
            environment=env,
            network_config=network_config_dict,
            role_arn=self.sagemaker_session.expand_role(self.role),
        )

        self._wait_for_schedule_changes_to_apply()

    def start_monitoring_schedule(self):
        """Starts the monitoring schedule."""
        self.sagemaker_session.start_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

        self._wait_for_schedule_changes_to_apply()

    def stop_monitoring_schedule(self):
        """Stops the monitoring schedule."""
        self.sagemaker_session.stop_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

        self._wait_for_schedule_changes_to_apply()

    def delete_monitoring_schedule(self):
        """Deletes the monitoring schedule (subclass is responsible for deleting job definition)"""
        # DO NOT call super which erases schedule name and makes wait impossible.
        self.sagemaker_session.delete_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )
        if self.job_definition_name is not None:
            # Job definition is locked by schedule so need to wait for the schedule to be deleted
            try:
                self._wait_for_schedule_changes_to_apply()
            except self.sagemaker_session.sagemaker_client.exceptions.ResourceNotFound:
                # OK the schedule is gone
                pass
        self.monitoring_schedule_name = None

    def baseline_statistics(self, file_name=STATISTICS_JSON_DEFAULT_FILE_NAME):
        """Returns a Statistics object representing the statistics json file

        Object is generated by the latest baselining job.

        Args:
            file_name (str): The name of the .json statistics file

        Returns:
            sagemaker.model_monitor.Statistics: The Statistics object representing the file that
                was generated by the job.

        """
        return self.latest_baselining_job.baseline_statistics(
            file_name=file_name, kms_key=self.output_kms_key
        )

    def suggested_constraints(self, file_name=CONSTRAINTS_JSON_DEFAULT_FILE_NAME):
        """Returns a Statistics object representing the constraints json file.

        Object is generated by the latest baselining job

        Args:
            file_name (str): The name of the .json constraints file

        Returns:
            sagemaker.model_monitor.Constraints: The Constraints object representing the file that
                was generated by the job.

        """
        return self.latest_baselining_job.suggested_constraints(
            file_name=file_name, kms_key=self.output_kms_key
        )

    def latest_monitoring_statistics(self, file_name=STATISTICS_JSON_DEFAULT_FILE_NAME):
        """Returns the sagemaker.model_monitor.

        Statistics generated by the latest monitoring execution.

        Args:
            file_name (str): The name of the statistics file to be retrieved. Only override if
                generating a custom file name.

        Returns:
            sagemaker.model_monitoring.Statistics: The Statistics object representing the file
                generated by the latest monitoring execution.

        """
        executions = self.list_executions()
        if len(executions) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return None

        latest_monitoring_execution = executions[-1]
        return latest_monitoring_execution.statistics(file_name=file_name)

    def latest_monitoring_constraint_violations(
        self, file_name=CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME
    ):
        """Returns the sagemaker.model_monitor.

        ConstraintViolations generated by the latest monitoring execution.

        Args:
            file_name (str): The name of the constraint violdations file to be retrieved. Only
                override if generating a custom file name.

        Returns:
            sagemaker.model_monitoring.ConstraintViolations: The ConstraintViolations object
                representing the file generated by the latest monitoring execution.

        """
        executions = self.list_executions()
        if len(executions) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return None

        latest_monitoring_execution = executions[-1]
        return latest_monitoring_execution.constraint_violations(file_name=file_name)

    def describe_latest_baselining_job(self):
        """Describe the latest baselining job kicked off by the suggest workflow."""
        if self.latest_baselining_job is None:
            raise ValueError("No suggestion jobs were kicked off.")
        return self.latest_baselining_job.describe()

    def describe_schedule(self):
        """Describes the schedule that this object represents.

        Returns:
            dict: A dictionary response with the monitoring schedule description.

        """
        return self.sagemaker_session.describe_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

    def list_executions(self):
        """Get the list of the latest monitoring executions in descending order of "ScheduledTime".

        Statistics or violations can be called following this example:
        Example:
            >>> my_executions = my_monitor.list_executions()
            >>> second_to_last_execution_statistics = my_executions[-1].statistics()
            >>> second_to_last_execution_violations = my_executions[-1].constraint_violations()

        Returns:
            [sagemaker.model_monitor.MonitoringExecution]: List of MonitoringExecutions in
                descending order of "ScheduledTime".

        """
        monitoring_executions_dict = self.sagemaker_session.list_monitoring_executions(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

        if len(monitoring_executions_dict["MonitoringExecutionSummaries"]) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return []

        processing_job_arns = [
            execution_dict["ProcessingJobArn"]
            for execution_dict in monitoring_executions_dict["MonitoringExecutionSummaries"]
            if execution_dict.get("ProcessingJobArn") is not None
        ]
        monitoring_executions = [
            MonitoringExecution.from_processing_arn(
                sagemaker_session=self.sagemaker_session, processing_job_arn=processing_job_arn
            )
            for processing_job_arn in processing_job_arns
        ]
        monitoring_executions.reverse()

        return monitoring_executions

    def get_latest_execution_logs(self, wait=False):
        """Get the processing job logs for the most recent monitoring execution

        Args:
            wait (bool): Whether the call should wait until the job completes (default: False).

        Raises:
            ValueError: If no execution job or processing job for the last execution has run

        Returns: None
        """
        monitoring_executions = self.sagemaker_session.list_monitoring_executions(
            monitoring_schedule_name=self.monitoring_schedule_name
        )
        if len(monitoring_executions["MonitoringExecutionSummaries"]) == 0:
            raise ValueError("No execution jobs were kicked off.")
        if "ProcessingJobArn" not in monitoring_executions["MonitoringExecutionSummaries"][0]:
            raise ValueError("Processing Job did not run for the last execution")
        job_arn = monitoring_executions["MonitoringExecutionSummaries"][0]["ProcessingJobArn"]
        self.sagemaker_session.logs_for_processing_job(
            job_name=get_resource_name_from_arn(job_arn), wait=wait
        )

    def update_monitoring_alert(
        self,
        monitoring_alert_name: str,
        data_points_to_alert: Optional[int],
        evaluation_period: Optional[int],
    ):
        """Update the monitoring schedule alert.

         Args:
            monitoring_alert_name (str): The name of the monitoring alert to update.
            data_points_to_alert (int):  The data point to alert.
            evaluation_period (int): The period to evaluate the alert status.

        Returns: None
        """

        if self.monitoring_schedule_name is None:
            message = "Nothing to update, please create a schedule first."
            _LOGGER.error(message)
            raise ValueError(message)

        if not data_points_to_alert and not evaluation_period:
            raise ValueError("Got no alert property to update.")

        self.sagemaker_session.update_monitoring_alert(
            monitoring_schedule_name=self.monitoring_schedule_name,
            monitoring_alert_name=monitoring_alert_name,
            data_points_to_alert=data_points_to_alert,
            evaluation_period=evaluation_period,
        )

    def list_monitoring_alerts(
        self, next_token: Optional[str] = None, max_results: Optional[int] = 10
    ):
        """List the monitoring alerts.

        Args:
             next_token (Optional[str]):  The pagination token. Default: None
             max_results (Optional[int]): The maximum number of results to return.
             Must be between 1 and 100. Default: 10

        Returns:
             List[MonitoringAlertSummary]: list of monitoring alert history.
             str: Next token.
        """
        if self.monitoring_schedule_name is None:
            message = "No alert to list, please create a schedule first."
            _LOGGER.warning(message)
            return [], None

        monitoring_alert_dict: Dict = self.sagemaker_session.list_monitoring_alerts(
            monitoring_schedule_name=self.monitoring_schedule_name,
            next_token=next_token,
            max_results=max_results,
        )
        monitoring_alerts: List[MonitoringAlertSummary] = []
        for monitoring_alert in monitoring_alert_dict["MonitoringAlertSummaries"]:
            monitoring_alerts.append(
                MonitoringAlertSummary(
                    alert_name=monitoring_alert["MonitoringAlertName"],
                    creation_time=monitoring_alert["CreationTime"],
                    last_modified_time=monitoring_alert["LastModifiedTime"],
                    alert_status=monitoring_alert["AlertStatus"],
                    data_points_to_alert=monitoring_alert["DatapointsToAlert"],
                    evaluation_period=monitoring_alert["EvaluationPeriod"],
                    actions=MonitoringAlertActions(
                        model_dashboard_indicator=ModelDashboardIndicatorAction(
                            enabled=monitoring_alert["Actions"]["ModelDashboardIndicator"][
                                "Enabled"
                            ],
                        )
                    ),
                )
            )

        next_token = (
            monitoring_alert_dict["NextToken"] if "NextToken" in monitoring_alert_dict else None
        )
        return monitoring_alerts, next_token

    def list_monitoring_alert_history(
        self,
        monitoring_alert_name: Optional[str] = None,
        sort_by: Optional[str] = "CreationTime",
        sort_order: Optional[str] = "Descending",
        next_token: Optional[str] = None,
        max_results: Optional[int] = 10,
        creation_time_before: Optional[str] = None,
        creation_time_after: Optional[str] = None,
        status_equals: Optional[str] = None,
    ):
        """Lists the alert history associated with the given schedule_name and alert_name.

        Args:
            monitoring_alert_name (Optional[str]): The name of the alert_name to filter on.
                If not provided, does not filter on it. Default: None.
            sort_by (Optional[str]): sort_by (str): The field to sort by.
                Can be one of: "Name", "CreationTime"
                Default: "CreationTime".
            sort_order (Optional[str]): The sort order. Can be one of: "Ascending", "Descending".
                Default: "Descending".
            next_token (Optional[str]):  The pagination token. Default: None.
            max_results (Optional[int]): The maximum number of results to return.
                Must be between 1 and 100. Default: 10.
            creation_time_before (Optional[str]): A filter to filter alert history before a time
                Default: None.
            creation_time_after (Optional[str]): A filter to filter alert history after a time
                Default: None.
            status_equals (Optional[str]): A filter to filter alert history by status
                Default: None.
        Returns:
            List[MonitoringAlertHistorySummary]: list of monitoring alert history.
            str: Next token.
        """
        if self.monitoring_schedule_name is None:
            message = "No alert history to list, please create a schedule first."
            _LOGGER.warning(message)
            return [], None

        monitoring_alert_history_dict: Dict = self.sagemaker_session.list_monitoring_alert_history(
            monitoring_schedule_name=self.monitoring_schedule_name,
            monitoring_alert_name=monitoring_alert_name,
            sort_by=sort_by,
            sort_order=sort_order,
            next_token=next_token,
            max_results=max_results,
            status_equals=status_equals,
            creation_time_before=creation_time_before,
            creation_time_after=creation_time_after,
        )
        monitoring_alert_history: List[MonitoringAlertHistorySummary] = []
        for monitoring_alert_history_summary in monitoring_alert_history_dict[
            "MonitoringAlertHistory"
        ]:
            monitoring_alert_history.append(
                MonitoringAlertHistorySummary(
                    alert_name=monitoring_alert_history_summary["MonitoringAlertName"],
                    creation_time=monitoring_alert_history_summary["CreationTime"],
                    alert_status=monitoring_alert_history_summary["AlertStatus"],
                )
            )

        next_token = (
            monitoring_alert_history_dict["NextToken"]
            if "NextToken" in monitoring_alert_history_dict
            else None
        )
        return monitoring_alert_history, next_token

    @classmethod
    def attach(cls, monitor_schedule_name, sagemaker_session=None):
        """Set this object's schedule name point to the Amazon Sagemaker Monitoring Schedule name.

        This allows subsequent describe_schedule or list_executions calls to point
        to the given schedule.

        Args:
            monitor_schedule_name (str): The name of the schedule to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.

        """
        sagemaker_session = sagemaker_session or Session()
        schedule_desc = sagemaker_session.describe_monitoring_schedule(
            monitoring_schedule_name=monitor_schedule_name
        )

        monitoring_job_definition = schedule_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinition"
        ]
        role = monitoring_job_definition["RoleArn"]
        image_uri = monitoring_job_definition["MonitoringAppSpecification"].get("ImageUri")
        cluster_config = monitoring_job_definition["MonitoringResources"]["ClusterConfig"]
        instance_count = cluster_config.get("InstanceCount")
        instance_type = cluster_config["InstanceType"]
        volume_size_in_gb = cluster_config["VolumeSizeInGB"]
        volume_kms_key = cluster_config.get("VolumeKmsKeyId")
        entrypoint = monitoring_job_definition["MonitoringAppSpecification"].get(
            "ContainerEntrypoint"
        )
        output_kms_key = monitoring_job_definition["MonitoringOutputConfig"].get("KmsKeyId")
        network_config_dict = monitoring_job_definition.get("NetworkConfig")

        max_runtime_in_seconds = None
        stopping_condition = monitoring_job_definition.get("StoppingCondition")
        if stopping_condition:
            max_runtime_in_seconds = stopping_condition.get("MaxRuntimeInSeconds")

        env = monitoring_job_definition.get("Environment", None)

        vpc_config = None
        if network_config_dict:
            vpc_config = network_config_dict.get("VpcConfig")

        security_group_ids = None
        if vpc_config:
            security_group_ids = vpc_config["SecurityGroupIds"]

        subnets = None
        if vpc_config:
            subnets = vpc_config["Subnets"]

        network_config = None
        if network_config_dict:
            network_config = NetworkConfig(
                enable_network_isolation=network_config_dict["EnableNetworkIsolation"],
                encrypt_inter_container_traffic=network_config_dict[
                    "EnableInterContainerTrafficEncryption"
                ],
                security_group_ids=security_group_ids,
                subnets=subnets,
            )

        tags = sagemaker_session.list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])

        attached_monitor = cls(
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            entrypoint=entrypoint,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )
        attached_monitor.monitoring_schedule_name = monitor_schedule_name
        return attached_monitor

    @staticmethod
    def _attach(clazz, sagemaker_session, schedule_desc, job_desc, tags):
        """Attach a model monitor object to an existing monitoring schedule.

        Args:
            clazz: a subclass of this class
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            schedule_desc (dict): output of describe monitoring schedule API.
            job_desc (dict): output of describe job definition API.

        Returns:
            Object of a subclass of this class.
        """

        monitoring_schedule_name = schedule_desc["MonitoringScheduleName"]
        job_definition_name = schedule_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinitionName"
        ]
        monitoring_type = schedule_desc["MonitoringScheduleConfig"]["MonitoringType"]
        role = job_desc["RoleArn"]
        cluster_config = job_desc["JobResources"]["ClusterConfig"]
        instance_count = cluster_config.get("InstanceCount")
        instance_type = cluster_config["InstanceType"]
        volume_size_in_gb = cluster_config["VolumeSizeInGB"]
        volume_kms_key = cluster_config.get("VolumeKmsKeyId")
        output_kms_key = job_desc["{}JobOutputConfig".format(monitoring_type)].get("KmsKeyId")
        network_config_dict = job_desc.get("NetworkConfig", {})

        max_runtime_in_seconds = None
        stopping_condition = job_desc.get("StoppingCondition")
        if stopping_condition:
            max_runtime_in_seconds = stopping_condition.get("MaxRuntimeInSeconds")

        env = job_desc["{}AppSpecification".format(monitoring_type)].get("Environment", None)

        vpc_config = network_config_dict.get("VpcConfig")

        security_group_ids = None
        if vpc_config:
            security_group_ids = vpc_config["SecurityGroupIds"]

        subnets = None
        if vpc_config:
            subnets = vpc_config["Subnets"]

        network_config = None
        if network_config_dict:
            network_config = NetworkConfig(
                enable_network_isolation=network_config_dict["EnableNetworkIsolation"],
                encrypt_inter_container_traffic=network_config_dict[
                    "EnableInterContainerTrafficEncryption"
                ],
                security_group_ids=security_group_ids,
                subnets=subnets,
            )

        attached_monitor = clazz(
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )
        attached_monitor.monitoring_schedule_name = monitoring_schedule_name
        attached_monitor.job_definition_name = job_definition_name
        return attached_monitor

    def _generate_baselining_job_name(self, job_name=None):
        """Generate the job name before running a suggestion processing job.

        Args:
            job_name (str): Name of the suggestion processing job to be created. If not
                specified, one is generated using the base name given to the
                constructor, if applicable.

        Returns:
            str: The supplied or generated job name.

        """
        if job_name is not None:
            return job_name

        if self.base_job_name:
            base_name = self.base_job_name
        else:
            base_name = _SUGGESTION_JOB_BASE_NAME

        return name_from_base(base=base_name)

    def _generate_monitoring_schedule_name(self, schedule_name=None):
        """Generate the monitoring schedule name.

        Args:
            schedule_name (str): Name of the monitoring schedule to be created. If not
                specified, one is generated using the base name given to the
                constructor, if applicable.

        Returns:
            str: The supplied or generated job name.

        """
        if schedule_name is not None:
            return schedule_name

        if self.base_job_name:
            base_name = self.base_job_name
        else:
            base_name = _MONITORING_SCHEDULE_BASE_NAME

        return name_from_base(base=base_name)

    @staticmethod
    def _generate_env_map(
        env,
        output_path=None,
        enable_cloudwatch_metrics=None,
        record_preprocessor_script_container_path=None,
        post_processor_script_container_path=None,
        dataset_format=None,
        dataset_source_container_path=None,
        analysis_type=None,
        problem_type=None,
        inference_attribute=None,
        probability_attribute=None,
        ground_truth_attribute=None,
        probability_threshold_attribute=None,
        categorical_drift_method=None,
    ):
        """Generate a list of environment variables from first-class parameters.

        Args:
            output_path (str): Local path to the output.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            record_preprocessor_script_container_path (str): The path to the record preprocessor
                script.
            post_processor_script_container_path (str): The path to the post analytics processor
                script.
            dataset_format (dict): The format of the baseline_dataset.
            dataset_source_container_path (str): The path to the dataset source.
            inference_attribute (str): Index or JSONpath to locate predicted label(s).
                Only used for ModelQualityMonitor.
            probability_attribute (str or int): Index or JSONpath to locate probabilities.
                Only used for ModelQualityMonitor.
            ground_truth_attribute (str): Index to locate actual label(s).
                Only used for ModelQualityMonitor.
            probability_threshold_attribute (float): threshold to convert probabilities to binaries
                Only used for ModelQualityMonitor.
            categorical_drift_method (str): categorical_drift_method to override the
                categorical_drift_method of global monitoring_config in constraints
                suggested by Model Monitor container. Only used for DataQualityMonitor.

        Returns:
            dict: Dictionary of environment keys and values.

        """
        cloudwatch_env_map = {True: "Enabled", False: "Disabled"}

        if env is not None:
            env = copy.deepcopy(env)
        env = env or {}

        if output_path is not None:
            env[_OUTPUT_PATH_ENV_NAME] = output_path

        if enable_cloudwatch_metrics is not None:
            env[_PUBLISH_CLOUDWATCH_METRICS_ENV_NAME] = cloudwatch_env_map[
                enable_cloudwatch_metrics
            ]

        if dataset_format is not None:
            env[_DATASET_FORMAT_ENV_NAME] = json.dumps(dataset_format)

        if record_preprocessor_script_container_path is not None:
            env[_RECORD_PREPROCESSOR_SCRIPT_ENV_NAME] = record_preprocessor_script_container_path

        if post_processor_script_container_path is not None:
            env[_POST_ANALYTICS_PROCESSOR_SCRIPT_ENV_NAME] = post_processor_script_container_path

        if dataset_source_container_path is not None:
            env[_DATASET_SOURCE_PATH_ENV_NAME] = dataset_source_container_path

        if analysis_type is not None:
            env[_ANALYSIS_TYPE_ENV_NAME] = analysis_type

        if problem_type is not None:
            env[_PROBLEM_TYPE_ENV_NAME] = problem_type

        if inference_attribute is not None:
            env[_INFERENCE_ATTRIBUTE_ENV_NAME] = inference_attribute

        if probability_attribute is not None:
            env[_PROBABILITY_ATTRIBUTE_ENV_NAME] = probability_attribute

        if ground_truth_attribute is not None:
            env[_GROUND_TRUTH_ATTRIBUTE_ENV_NAME] = ground_truth_attribute

        if probability_threshold_attribute is not None:
            env[_PROBABILITY_THRESHOLD_ATTRIBUTE_ENV_NAME] = probability_threshold_attribute

        if categorical_drift_method is not None:
            env[_CATEGORICAL_DRIFT_METHOD_ENV_NAME] = categorical_drift_method

        return env

    @staticmethod
    def _get_baseline_files(statistics, constraints, sagemaker_session=None):
        """Populates baseline values if possible.

        Args:
            statistics (sagemaker.model_monitor.Statistics or str): The statistics object or str.
                If none, this method will attempt to retrieve a previously baselined constraints
                object.
            constraints (sagemaker.model_monitor.Constraints or str): The constraints object or str.
                If none, this method will attempt to retrieve a previously baselined constraints
                object.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, one
                is created using the default AWS configuration chain.

        Returns:
            sagemaker.model_monitor.Statistics, sagemaker.model_monitor.Constraints: The Statistics
                and Constraints objects that were provided or created by the latest
                baselining job. If none were found, returns None.

        """
        if statistics is not None and isinstance(statistics, string_types):
            statistics = Statistics.from_s3_uri(
                statistics_file_s3_uri=statistics, sagemaker_session=sagemaker_session
            )
        if constraints is not None and isinstance(constraints, string_types):
            constraints = Constraints.from_s3_uri(
                constraints_file_s3_uri=constraints, sagemaker_session=sagemaker_session
            )

        return statistics, constraints

    def _normalize_endpoint_input(self, endpoint_input):
        """Ensure that the input is an EndpointInput object.

        Args:
            endpoint_input ([str or sagemaker.model_monitor.EndpointInput]): An endpoint input
                to be normalized. Can be either a string or a EndpointInput object.

        Returns:
            sagemaker.model_monitor.EndpointInput: The normalized EndpointInput object.

        """
        # If the input is a string, turn it into an EndpointInput object.
        if isinstance(endpoint_input, string_types):
            endpoint_input = EndpointInput(
                endpoint_name=endpoint_input,
                destination=str(
                    pathlib.PurePosixPath(
                        _CONTAINER_BASE_PATH, _CONTAINER_INPUT_PATH, _CONTAINER_ENDPOINT_INPUT_PATH
                    )
                ),
            )

        return endpoint_input

    def _normalize_baseline_inputs(self, baseline_inputs=None):
        """Ensure that all the ProcessingInput objects have names and S3 uris.

        Args:
            baseline_inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput
                objects to be normalized.

        Returns:
            [sagemaker.processing.ProcessingInput]: The list of normalized
                ProcessingInput objects.

        """
        # Initialize a list of normalized ProcessingInput objects.
        normalized_inputs = []
        if baseline_inputs is not None:
            # Iterate through the provided list of inputs.
            for count, file_input in enumerate(baseline_inputs, 1):
                if not isinstance(file_input, ProcessingInput):
                    raise TypeError("Your inputs must be provided as ProcessingInput objects.")
                # Generate a name for the ProcessingInput if it doesn't have one.
                if file_input.input_name is None:
                    file_input.input_name = "input-{}".format(count)
                # If the source is a local path, upload it to S3
                # and save the S3 uri in the ProcessingInput source.
                parse_result = urlparse(file_input.source)
                if parse_result.scheme != "s3":
                    s3_uri = s3.s3_path_join(
                        "s3://",
                        self.sagemaker_session.default_bucket(),
                        self.sagemaker_session.default_bucket_prefix,
                        self.latest_baselining_job_name,
                        file_input.input_name,
                    )
                    s3.S3Uploader.upload(
                        local_path=file_input.source,
                        desired_s3_uri=s3_uri,
                        sagemaker_session=self.sagemaker_session,
                    )
                    file_input.source = s3_uri
                normalized_inputs.append(file_input)
        return normalized_inputs

    def _normalize_baseline_output(self, output_s3_uri=None):
        """Ensure that the output is a ProcessingOutput object.

        Args:
            output_s3_uri (str): The output S3 uri to deposit the baseline files in.

        Returns:
            sagemaker.processing.ProcessingOutput: The normalized ProcessingOutput object.

        """
        s3_uri = output_s3_uri or s3.s3_path_join(
            "s3://",
            self.sagemaker_session.default_bucket(),
            self.sagemaker_session.default_bucket_prefix,
            _MODEL_MONITOR_S3_PATH,
            _BASELINING_S3_PATH,
            self.latest_baselining_job_name,
            _RESULTS_S3_PATH,
        )
        return ProcessingOutput(
            source=str(pathlib.PurePosixPath(_CONTAINER_BASE_PATH, _CONTAINER_OUTPUT_PATH)),
            destination=s3_uri,
            output_name=_DEFAULT_OUTPUT_NAME,
        )

    def _normalize_processing_output(self, output=None):
        """Ensure that the output is a ProcessingOutput object.

        Args:
            output (str or sagemaker.processing.ProcessingOutput): An output to be normalized.

        Returns:
            sagemaker.processing.ProcessingOutput: The normalized ProcessingOutput object.

        """
        # If the output is a string, turn it into a ProcessingOutput object.
        if isinstance(output, string_types):
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                self.latest_baselining_job_name,
                "output",
            )
            output = ProcessingOutput(
                source=output, destination=s3_uri, output_name=_DEFAULT_OUTPUT_NAME
            )

        return output

    def _normalize_monitoring_output(self, monitoring_schedule_name, output_s3_uri=None):
        """Ensure that the output is a MonitoringOutput object.

        Args:
            monitoring_schedule_name (str): Monitoring schedule name
            output_s3_uri (str): The output S3 uri to deposit the monitoring evaluation files in.

        Returns:
            sagemaker.model_monitor.MonitoringOutput: The normalized MonitoringOutput object.

        """
        s3_uri = output_s3_uri or s3.s3_path_join(
            "s3://",
            self.sagemaker_session.default_bucket(),
            self.sagemaker_session.default_bucket_prefix,
            _MODEL_MONITOR_S3_PATH,
            _MONITORING_S3_PATH,
            monitoring_schedule_name,
            _RESULTS_S3_PATH,
        )
        output = MonitoringOutput(
            source=str(pathlib.PurePosixPath(_CONTAINER_BASE_PATH, _CONTAINER_OUTPUT_PATH)),
            destination=s3_uri,
        )
        return output

    def _normalize_monitoring_output_fields(self, output=None):
        """Ensure that output has the correct fields.

        Args:
            output (sagemaker.model_monitor.MonitoringOutput): An output to be normalized.

        Returns:
            sagemaker.model_monitor.MonitoringOutput: The normalized MonitoringOutput object.

        """
        # If the output destination is missing, assign a default destination to it.
        if output.destination is None:
            output.destination = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                self.monitoring_schedule_name,
                "output",
            )

        return output

    def _s3_uri_from_local_path(self, path):
        """If path is local, uploads to S3 and returns S3 uri. Otherwise returns S3 uri as-is.

        Args:
            path (str): Path to file. This can be a local path or an S3 path.

        Returns:
            str: S3 uri to file.

        """
        parse_result = urlparse(path)
        if parse_result.scheme != "s3":
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                _MODEL_MONITOR_S3_PATH,
                _MONITORING_S3_PATH,
                self.monitoring_schedule_name,
                _INPUT_S3_PATH,
                str(uuid.uuid4()),
            )
            s3.S3Uploader.upload(
                local_path=path, desired_s3_uri=s3_uri, sagemaker_session=self.sagemaker_session
            )
            path = s3.s3_path_join(s3_uri, os.path.basename(path))
        return path

    def _wait_for_schedule_changes_to_apply(self):
        """Waits for the schedule to no longer be in the 'Pending' state."""
        for _ in retries(
            max_retry_count=36,  # 36*5 = 3min
            exception_message_prefix="Waiting for schedule to leave 'Pending' status",
            seconds_to_sleep=5,
        ):
            schedule_desc = self.describe_schedule()
            if schedule_desc["MonitoringScheduleStatus"] != "Pending":
                break

    @classmethod
    def monitoring_type(cls):
        """Type of the monitoring job."""
        raise TypeError("Subclass of {} shall define this property".format(__class__.__name__))

    def _create_monitoring_schedule_from_job_definition(
        self, monitor_schedule_name, job_definition_name, schedule_cron_expression=None
    ):
        """Creates a monitoring schedule.

        Args:
            monitor_schedule_name (str): Monitoring schedule name.
            job_definition_name (str): Job definition name.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily.
        """
        message = "Creating Monitoring Schedule with name: {}".format(monitor_schedule_name)
        _LOGGER.info(message)

        monitoring_schedule_config = {
            "MonitoringJobDefinitionName": job_definition_name,
            "MonitoringType": self.monitoring_type(),
        }
        if schedule_cron_expression is not None:
            monitoring_schedule_config["ScheduleConfig"] = {
                "ScheduleExpression": schedule_cron_expression
            }
        all_tags = self.sagemaker_session._append_sagemaker_config_tags(
            self.tags, "{}.{}.{}".format(SAGEMAKER, MONITORING_SCHEDULE, TAGS)
        )

        # Not using value from sagemaker
        # config key MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH here
        # because no MonitoringJobDefinition is set for this call

        self.sagemaker_session.sagemaker_client.create_monitoring_schedule(
            MonitoringScheduleName=monitor_schedule_name,
            MonitoringScheduleConfig=monitoring_schedule_config,
            Tags=all_tags or [],
        )

    def _upload_and_convert_to_processing_input(self, source, destination, name):
        """Generates a ProcessingInput object from a source.

        Source can be a local path or an S3 uri.

        Args:
            source (str): The source of the data. This can be a local path or an S3 uri.
            destination (str): The desired container path for the data to be downloaded to.
            name (str): The name of the ProcessingInput.

        Returns:
            sagemaker.processing.ProcessingInput: The ProcessingInput object.

        """
        if source is None:
            return None

        parse_result = urlparse(url=source)

        if parse_result.scheme != "s3":
            s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                _MODEL_MONITOR_S3_PATH,
                _BASELINING_S3_PATH,
                self.latest_baselining_job_name,
                _INPUT_S3_PATH,
                name,
            )
            s3.S3Uploader.upload(
                local_path=source, desired_s3_uri=s3_uri, sagemaker_session=self.sagemaker_session
            )
            source = s3_uri

        return ProcessingInput(source=source, destination=destination, input_name=name)

    # noinspection PyMethodOverriding
    def _update_monitoring_schedule(self, job_definition_name, schedule_cron_expression=None):
        """Updates existing monitoring schedule with new job definition and/or schedule expression.

        Args:
            job_definition_name (str): Job definition name.
            schedule_cron_expression (str or None): The cron expression that dictates the frequency
                that this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions.
        """
        if self.job_definition_name is None or self.monitoring_schedule_name is None:
            message = "Nothing to update, please create a schedule first."
            _LOGGER.error(message)
            raise ValueError(message)

        monitoring_schedule_config = {
            "MonitoringJobDefinitionName": job_definition_name,
            "MonitoringType": self.monitoring_type(),
        }
        if schedule_cron_expression is not None:
            monitoring_schedule_config["ScheduleConfig"] = {
                "ScheduleExpression": schedule_cron_expression
            }

        # Not using value from sagemaker
        # config key MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH here
        # because no MonitoringJobDefinition is set for this call

        self.sagemaker_session.sagemaker_client.update_monitoring_schedule(
            MonitoringScheduleName=self.monitoring_schedule_name,
            MonitoringScheduleConfig=monitoring_schedule_config,
        )
        self._wait_for_schedule_changes_to_apply()


class DefaultModelMonitor(ModelMonitor):
    """Sets up Amazon SageMaker Monitoring Schedules and baseline suggestions.

    Use this class when you want to utilize Amazon SageMaker Monitoring's plug-and-play
    solution that only requires your dataset and optional pre/postprocessing scripts.
    For a more customized experience, consider using the ModelMonitor class instead.
    """

    JOB_DEFINITION_BASE_NAME = "data-quality-job-definition"

    def __init__(
        self,
        role=None,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initializes a ``Monitor`` instance.

        The Monitor handles baselining datasets and creating Amazon SageMaker Monitoring
        Schedules to monitor SageMaker endpoints.

        Args:
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.
            instance_count (int): The number of instances to run the jobs with.
            instance_type (str): Type of EC2 instance to use for the job, for example,
                'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            base_job_name (str): Prefix for the job name. If not specified,
                a default name is generated based on the training image name and
                current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            env (dict): Environment variables to be passed to the job.
            tags ([dict]): List of tags to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.

        """
        session = sagemaker_session or Session()
        super(DefaultModelMonitor, self).__init__(
            role=role,
            image_uri=DefaultModelMonitor._get_default_image_uri(session.boto_session.region_name),
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

    @classmethod
    def monitoring_type(cls):
        """Type of the monitoring job."""
        return "DataQuality"

    # ToDo: either support record_preprocessor_script or remove it from here. It has
    #  not been removed due to backward compatibility issues
    def suggest_baseline(
        self,
        baseline_dataset,
        dataset_format,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        wait=True,
        logs=True,
        job_name=None,
        monitoring_config_override=None,
    ):
        """Suggest baselines for use with Amazon SageMaker Model Monitoring Schedules.

        Args:
            baseline_dataset (str): The path to the baseline_dataset file. This can be a local path
                or an S3 uri.
            dataset_format (dict): The format of the baseline_dataset.
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): Desired S3 destination Destination of the constraint_violations
                and statistics json files.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            monitoring_config_override (DataQualityMonitoringConfig): monitoring_config object to
                override the global monitoring_config parameter of constraints suggested by
                Model Monitor Container. If not specified, the values suggested by container is
                set.
        Returns:
            sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
                baselining job.

        """
        if not DataQualityMonitoringConfig.valid_monitoring_config(monitoring_config_override):
            raise RuntimeError("Invalid value for monitoring_config_override.")

        self.latest_baselining_job_name = self._generate_baselining_job_name(job_name=job_name)

        normalized_baseline_dataset_input = self._upload_and_convert_to_processing_input(
            source=baseline_dataset,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH, _CONTAINER_INPUT_PATH, _BASELINE_DATASET_INPUT_NAME
                )
            ),
            name=_BASELINE_DATASET_INPUT_NAME,
        )

        # Unlike other input, dataset must be a directory for the Monitoring image.
        baseline_dataset_container_path = normalized_baseline_dataset_input.destination

        normalized_record_preprocessor_script_input = self._upload_and_convert_to_processing_input(
            source=record_preprocessor_script,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH,
                    _CONTAINER_INPUT_PATH,
                    _RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME,
                )
            ),
            name=_RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME,
        )

        record_preprocessor_script_container_path = None
        if normalized_record_preprocessor_script_input is not None:
            record_preprocessor_script_container_path = str(
                pathlib.PurePosixPath(
                    normalized_record_preprocessor_script_input.destination,
                    os.path.basename(record_preprocessor_script),
                )
            )

        normalized_post_processor_script_input = self._upload_and_convert_to_processing_input(
            source=post_analytics_processor_script,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH,
                    _CONTAINER_INPUT_PATH,
                    _POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME,
                )
            ),
            name=_POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME,
        )

        post_processor_script_container_path = None
        if normalized_post_processor_script_input is not None:
            post_processor_script_container_path = str(
                pathlib.PurePosixPath(
                    normalized_post_processor_script_input.destination,
                    os.path.basename(post_analytics_processor_script),
                )
            )

        normalized_baseline_output = self._normalize_baseline_output(output_s3_uri=output_s3_uri)

        categorical_drift_method = None
        if monitoring_config_override and monitoring_config_override.distribution_constraints:
            distribution_constraints = monitoring_config_override.distribution_constraints
            categorical_drift_method = distribution_constraints.categorical_drift_method

        normalized_env = self._generate_env_map(
            env=self.env,
            dataset_format=dataset_format,
            output_path=normalized_baseline_output.source,
            enable_cloudwatch_metrics=False,  # Only supported for monitoring schedules
            dataset_source_container_path=baseline_dataset_container_path,
            record_preprocessor_script_container_path=record_preprocessor_script_container_path,
            post_processor_script_container_path=post_processor_script_container_path,
            categorical_drift_method=categorical_drift_method,
        )

        baselining_processor = Processor(
            role=self.role,
            image_uri=self.image_uri,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            entrypoint=self.entrypoint,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            output_kms_key=self.output_kms_key,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            base_job_name=self.base_job_name,
            sagemaker_session=self.sagemaker_session,
            env=normalized_env,
            tags=self.tags,
            network_config=self.network_config,
        )

        baseline_job_inputs_with_nones = [
            normalized_baseline_dataset_input,
            normalized_record_preprocessor_script_input,
            normalized_post_processor_script_input,
        ]

        baseline_job_inputs = [
            baseline_job_input
            for baseline_job_input in baseline_job_inputs_with_nones
            if baseline_job_input is not None
        ]

        baselining_processor.run(
            inputs=baseline_job_inputs,
            outputs=[normalized_baseline_output],
            arguments=self.arguments,
            wait=wait,
            logs=logs,
            job_name=self.latest_baselining_job_name,
        )

        self.latest_baselining_job = BaseliningJob.from_processing_job(
            processing_job=baselining_processor.latest_job
        )
        self.baselining_jobs.append(self.latest_baselining_job)
        return baselining_processor.latest_job

    def create_monitoring_schedule(
        self,
        endpoint_input=None,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        constraints=None,
        statistics=None,
        monitor_schedule_name=None,
        schedule_cron_expression=None,
        enable_cloudwatch_metrics=True,
        batch_transform_input=None,
    ):
        """Creates a monitoring schedule to monitor an Amazon SageMaker Endpoint.

        If constraints and statistics are provided, or if they are able to be retrieved from a
        previous baselining job associated with this monitor, those will be used.
        If constraints and statistics cannot be automatically retrieved, baseline_inputs will be
        required in order to kick off a baselining job.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput. (default: None)
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): Desired S3 destination of the constraint_violations and
                statistics json files.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
                statistics, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an s3_uri pointing to a constraints
                JSON file.
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Statistics object or an s3_uri pointing to a statistics
                JSON file.
            monitor_schedule_name (str): Schedule name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
                run the monitoring schedule on the batch transform (default: None)
        """
        if self.job_definition_name is not None or self.monitoring_schedule_name is not None:
            message = (
                "It seems that this object was already used to create an Amazon Model "
                "Monitoring Schedule. To create another, first delete the existing one "
                "using my_monitor.delete_monitoring_schedule()."
            )
            _LOGGER.error(message)
            raise ValueError(message)

        if (batch_transform_input is not None) ^ (endpoint_input is None):
            message = (
                "Need to have either batch_transform_input or endpoint_input to create an "
                "Amazon Model Monitoring Schedule. "
                "Please provide only one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        # create job definition
        monitor_schedule_name = self._generate_monitoring_schedule_name(
            schedule_name=monitor_schedule_name
        )
        new_job_definition_name = name_from_base(self.JOB_DEFINITION_BASE_NAME)
        request_dict = self._build_create_data_quality_job_definition_request(
            monitoring_schedule_name=monitor_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self.image_uri,
            latest_baselining_job_name=self.latest_baselining_job_name,
            endpoint_input=endpoint_input,
            record_preprocessor_script=record_preprocessor_script,
            post_analytics_processor_script=post_analytics_processor_script,
            output_s3_uri=self._normalize_monitoring_output(
                monitor_schedule_name, output_s3_uri
            ).destination,
            constraints=constraints,
            statistics=statistics,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics,
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            output_kms_key=self.output_kms_key,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            env=self.env,
            tags=self.tags,
            network_config=self.network_config,
            batch_transform_input=batch_transform_input,
        )
        self.sagemaker_session.sagemaker_client.create_data_quality_job_definition(**request_dict)

        # create schedule
        try:
            self._create_monitoring_schedule_from_job_definition(
                monitor_schedule_name=monitor_schedule_name,
                job_definition_name=new_job_definition_name,
                schedule_cron_expression=schedule_cron_expression,
            )
            self.job_definition_name = new_job_definition_name
            self.monitoring_schedule_name = monitor_schedule_name
        except Exception:
            _LOGGER.exception("Failed to create monitoring schedule.")
            # noinspection PyBroadException
            try:
                self.sagemaker_session.sagemaker_client.delete_data_quality_job_definition(
                    JobDefinitionName=new_job_definition_name
                )
            except Exception:  # pylint: disable=W0703
                message = "Failed to delete job definition {}.".format(new_job_definition_name)
                _LOGGER.exception(message)
            raise

    def update_monitoring_schedule(
        self,
        endpoint_input=None,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        statistics=None,
        constraints=None,
        schedule_cron_expression=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        env=None,
        network_config=None,
        enable_cloudwatch_metrics=None,
        role=None,
        batch_transform_input=None,
    ):
        """Updates the existing monitoring schedule.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): Desired S3 destination of the constraint_violations and
                statistics json files.
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Statistics object or an S3 uri pointing to a statistics
                JSON file.
            constraints (sagemaker.model_monitor.Constraints or str): If provided alongside
                statistics, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Constraints object or an S3 uri pointing to a constraints
                JSON file.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job runs at. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            env (dict): Environment variables to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker jobs use this role.
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
                run the monitoring schedule on the batch transform (default: None)

        """

        if (batch_transform_input is not None) and (endpoint_input is not None):
            message = (
                "Cannot update both batch_transform_input and endpoint_input to update an "
                "Amazon Model Monitoring Schedule. "
                "Please provide atmost one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        # check if this schedule is in v2 format and update as per v2 format if it is
        if self.job_definition_name is not None:
            self._update_data_quality_monitoring_schedule(
                endpoint_input=endpoint_input,
                record_preprocessor_script=record_preprocessor_script,
                post_analytics_processor_script=post_analytics_processor_script,
                output_s3_uri=output_s3_uri,
                statistics=statistics,
                constraints=constraints,
                schedule_cron_expression=schedule_cron_expression,
                instance_count=instance_count,
                instance_type=instance_type,
                volume_size_in_gb=volume_size_in_gb,
                volume_kms_key=volume_kms_key,
                output_kms_key=output_kms_key,
                max_runtime_in_seconds=max_runtime_in_seconds,
                env=env,
                network_config=network_config,
                enable_cloudwatch_metrics=enable_cloudwatch_metrics,
                role=role,
                batch_transform_input=batch_transform_input,
            )
            return

        monitoring_inputs = None
        if endpoint_input is not None:
            monitoring_inputs = [self._normalize_endpoint_input(endpoint_input)._to_request_dict()]

        elif batch_transform_input is not None:
            monitoring_inputs = [batch_transform_input._to_request_dict()]

        record_preprocessor_script_s3_uri = None
        if record_preprocessor_script is not None:
            record_preprocessor_script_s3_uri = self._s3_uri_from_local_path(
                path=record_preprocessor_script
            )

        post_analytics_processor_script_s3_uri = None
        if post_analytics_processor_script is not None:
            post_analytics_processor_script_s3_uri = self._s3_uri_from_local_path(
                path=post_analytics_processor_script
            )

        monitoring_output_config = None
        output_path = None
        if output_s3_uri is not None:
            normalized_monitoring_output = self._normalize_monitoring_output(
                monitoring_schedule_name=self.monitoring_schedule_name,
                output_s3_uri=output_s3_uri,
            )
            monitoring_output_config = {
                "MonitoringOutputs": [normalized_monitoring_output._to_request_dict()]
            }
            output_path = normalized_monitoring_output.source

        if env is not None:
            self.env = env

        normalized_env = self._generate_env_map(
            env=env, output_path=output_path, enable_cloudwatch_metrics=enable_cloudwatch_metrics
        )

        statistics_object, constraints_object = self._get_baseline_files(
            statistics=statistics, constraints=constraints, sagemaker_session=self.sagemaker_session
        )

        statistics_s3_uri = None
        if statistics_object is not None:
            statistics_s3_uri = statistics_object.file_s3_uri

        constraints_s3_uri = None
        if constraints_object is not None:
            constraints_s3_uri = constraints_object.file_s3_uri

        if instance_type is not None:
            self.instance_type = instance_type

        if instance_count is not None:
            self.instance_count = instance_count

        if volume_size_in_gb is not None:
            self.volume_size_in_gb = volume_size_in_gb

        if volume_kms_key is not None:
            self.volume_kms_key = volume_kms_key

        if output_kms_key is not None:
            self.output_kms_key = output_kms_key
            monitoring_output_config["KmsKeyId"] = self.output_kms_key

        if max_runtime_in_seconds is not None:
            self.max_runtime_in_seconds = max_runtime_in_seconds

        if network_config is not None:
            self.network_config = network_config

        network_config_dict = None
        if self.network_config is not None:
            network_config_dict = self.network_config._to_request_dict()
        # Do not need to check config because that check is done inside
        # self.sagemaker_session.update_monitoring_schedule

        if role is not None:
            self.role = role

        self.sagemaker_session.update_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name,
            schedule_expression=schedule_cron_expression,
            constraints_s3_uri=constraints_s3_uri,
            statistics_s3_uri=statistics_s3_uri,
            monitoring_inputs=monitoring_inputs,
            monitoring_output_config=monitoring_output_config,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            record_preprocessor_source_uri=record_preprocessor_script_s3_uri,
            post_analytics_processor_source_uri=post_analytics_processor_script_s3_uri,
            max_runtime_in_seconds=max_runtime_in_seconds,
            environment=normalized_env,
            network_config=network_config_dict,
            role_arn=self.sagemaker_session.expand_role(self.role),
        )

        self._wait_for_schedule_changes_to_apply()

    def _update_data_quality_monitoring_schedule(
        self,
        endpoint_input=None,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        constraints=None,
        statistics=None,
        schedule_cron_expression=None,
        enable_cloudwatch_metrics=None,
        role=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        env=None,
        network_config=None,
        batch_transform_input=None,
    ):
        """Updates the existing monitoring schedule.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
                for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
                to a constraints JSON file.
            statistics (sagemaker.model_monitor.Statistic or str): If provided alongside
                constraints, these will be used for monitoring the endpoint. This can be a
                sagemaker.model_monitor.Statistics object or an S3 uri pointing to a statistics
                JSON file.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            env (dict): Environment variables to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
                run the monitoring schedule on the batch transform (default: None)
        """
        valid_args = {
            arg: value for arg, value in locals().items() if arg != "self" and value is not None
        }

        # Nothing to update
        if len(valid_args) <= 0:
            return

        # Only need to update schedule expression
        if len(valid_args) == 1 and schedule_cron_expression is not None:
            self._update_monitoring_schedule(self.job_definition_name, schedule_cron_expression)
            return

        existing_desc = self.sagemaker_session.describe_monitoring_schedule(
            monitoring_schedule_name=self.monitoring_schedule_name
        )

        if (
            existing_desc.get("MonitoringScheduleConfig") is not None
            and existing_desc["MonitoringScheduleConfig"].get("ScheduleConfig") is not None
            and existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"]
            is not None
            and schedule_cron_expression is None
        ):
            schedule_cron_expression = existing_desc["MonitoringScheduleConfig"]["ScheduleConfig"][
                "ScheduleExpression"
            ]

        # Need to update schedule with a new job definition
        job_desc = self.sagemaker_session.sagemaker_client.describe_data_quality_job_definition(
            JobDefinitionName=self.job_definition_name
        )
        new_job_definition_name = name_from_base(self.JOB_DEFINITION_BASE_NAME)
        request_dict = self._build_create_data_quality_job_definition_request(
            monitoring_schedule_name=self.monitoring_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self.image_uri,
            existing_job_desc=job_desc,
            endpoint_input=endpoint_input,
            record_preprocessor_script=record_preprocessor_script,
            post_analytics_processor_script=post_analytics_processor_script,
            output_s3_uri=output_s3_uri,
            statistics=statistics,
            constraints=constraints,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics,
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            env=env,
            tags=self.tags,
            network_config=network_config,
            batch_transform_input=batch_transform_input,
        )
        self.sagemaker_session.sagemaker_client.create_data_quality_job_definition(**request_dict)
        try:
            self._update_monitoring_schedule(new_job_definition_name, schedule_cron_expression)
            self.job_definition_name = new_job_definition_name
            if role is not None:
                self.role = role
            if instance_count is not None:
                self.instance_count = instance_count
            if instance_type is not None:
                self.instance_type = instance_type
            if volume_size_in_gb is not None:
                self.volume_size_in_gb = volume_size_in_gb
            if volume_kms_key is not None:
                self.volume_kms_key = volume_kms_key
            if output_kms_key is not None:
                self.output_kms_key = output_kms_key
            if max_runtime_in_seconds is not None:
                self.max_runtime_in_seconds = max_runtime_in_seconds
            if env is not None:
                self.env = env
            if network_config is not None:
                self.network_config = network_config
        except Exception:
            _LOGGER.exception("Failed to update monitoring schedule.")
            # noinspection PyBroadException
            try:
                self.sagemaker_session.sagemaker_client.delete_data_quality_job_definition(
                    JobDefinitionName=new_job_definition_name
                )
            except Exception:  # pylint: disable=W0703
                message = "Failed to delete job definition {}.".format(new_job_definition_name)
                _LOGGER.exception(message)
            raise

    def delete_monitoring_schedule(self):
        """Deletes the monitoring schedule and its job definition."""
        super(DefaultModelMonitor, self).delete_monitoring_schedule()
        if self.job_definition_name is not None:
            # Delete job definition.
            message = "Deleting Data Quality Job Definition with name: {}".format(
                self.job_definition_name
            )
            _LOGGER.info(message)
            self.sagemaker_session.sagemaker_client.delete_data_quality_job_definition(
                JobDefinitionName=self.job_definition_name
            )
            self.job_definition_name = None

    def run_baseline(self):
        """Not implemented.

        '.run_baseline()' is only allowed for ModelMonitor objects. Please use
        `suggest_baseline` for DefaultModelMonitor objects, instead.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "'.run_baseline()' is only allowed for ModelMonitor objects. "
            "Please use suggest_baseline for DefaultModelMonitor objects, instead."
        )

    @classmethod
    def attach(cls, monitor_schedule_name, sagemaker_session=None):
        """Sets this object's schedule name to the name provided.

        This allows subsequent describe_schedule or list_executions calls to point
        to the given schedule.

        Args:
            monitor_schedule_name (str): The name of the schedule to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or Session()
        schedule_desc = sagemaker_session.describe_monitoring_schedule(
            monitoring_schedule_name=monitor_schedule_name
        )

        job_definition_name = schedule_desc["MonitoringScheduleConfig"].get(
            "MonitoringJobDefinitionName"
        )
        if job_definition_name:
            monitoring_type = schedule_desc["MonitoringScheduleConfig"].get("MonitoringType")
            if monitoring_type != cls.monitoring_type():
                raise TypeError(
                    "{} can only attach to Data quality monitoring schedule.".format(
                        __class__.__name__
                    )
                )
            job_desc = sagemaker_session.sagemaker_client.describe_data_quality_job_definition(
                JobDefinitionName=job_definition_name
            )
            tags = sagemaker_session.list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])

            return ModelMonitor._attach(
                clazz=cls,
                sagemaker_session=sagemaker_session,
                schedule_desc=schedule_desc,
                job_desc=job_desc,
                tags=tags,
            )

        job_definition = schedule_desc["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        role = job_definition["RoleArn"]
        cluster_config = job_definition["MonitoringResources"]["ClusterConfig"]
        instance_count = cluster_config["InstanceCount"]
        instance_type = cluster_config["InstanceType"]
        volume_size_in_gb = cluster_config["VolumeSizeInGB"]
        volume_kms_key = cluster_config.get("VolumeKmsKeyId")
        output_kms_key = job_definition["MonitoringOutputConfig"].get("KmsKeyId")
        max_runtime_in_seconds = job_definition.get("StoppingCondition", {}).get(
            "MaxRuntimeInSeconds"
        )
        env = job_definition["Environment"]

        network_config_dict = job_definition.get("NetworkConfig", {})
        network_config = None
        if network_config_dict:
            vpc_config = network_config_dict.get("VpcConfig", {})
            security_group_ids = vpc_config.get("SecurityGroupIds")
            subnets = vpc_config.get("Subnets")
            network_config = NetworkConfig(
                enable_network_isolation=network_config_dict["EnableNetworkIsolation"],
                encrypt_inter_container_traffic=network_config_dict[
                    "EnableInterContainerTrafficEncryption"
                ],
                security_group_ids=security_group_ids,
                subnets=subnets,
            )

        tags = sagemaker_session.list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])

        attached_monitor = cls(
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )
        attached_monitor.monitoring_schedule_name = monitor_schedule_name
        return attached_monitor

    def latest_monitoring_statistics(self):
        """Returns the sagemaker.model_monitor.Statistics.

        These are the statistics generated by the latest monitoring execution.

        Returns:
            sagemaker.model_monitoring.Statistics: The Statistics object representing the file
                generated by the latest monitoring execution.

        """
        executions = self.list_executions()
        if len(executions) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return None

        latest_monitoring_execution = executions[-1]

        try:
            return latest_monitoring_execution.statistics()
        except ClientError:
            status = latest_monitoring_execution.describe()["ProcessingJobStatus"]
            print(
                "Unable to retrieve statistics as job is in status '{}'. Latest statistics only "
                "available for completed executions.".format(status)
            )

    def latest_monitoring_constraint_violations(self):
        """Returns the sagemaker.model_monitor.

        ConstraintViolations generated by the latest monitoring execution.

        Returns:
            sagemaker.model_monitoring.ConstraintViolations: The ConstraintViolations object
                representing the file generated by the latest monitoring execution.

        """
        executions = self.list_executions()
        if len(executions) == 0:
            print(
                "No executions found for schedule. monitoring_schedule_name: {}".format(
                    self.monitoring_schedule_name
                )
            )
            return None

        latest_monitoring_execution = executions[-1]
        try:
            return latest_monitoring_execution.constraint_violations()
        except ClientError:
            status = latest_monitoring_execution.describe()["ProcessingJobStatus"]
            print(
                "Unable to retrieve constraint violations as job is in status '{}'. Latest "
                "violations only available for completed executions.".format(status)
            )

    @staticmethod
    def _get_default_image_uri(region):
        """Returns the Default Model Monitoring image uri based on the region.

        Args:
            region (str): The AWS region.

        Returns:
            str: The Default Model Monitoring image uri based on the region.
        """
        return image_uris.retrieve(framework=framework_name, region=region)

    def _build_create_data_quality_job_definition_request(
        self,
        monitoring_schedule_name,
        job_definition_name,
        image_uri,
        latest_baselining_job_name=None,
        existing_job_desc=None,
        endpoint_input=None,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        statistics=None,
        constraints=None,
        enable_cloudwatch_metrics=None,
        role=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        env=None,
        tags=None,
        network_config=None,
        batch_transform_input=None,
    ):
        """Build the request for job definition creation API

        Args:
            monitoring_schedule_name (str): Monitoring schedule name.
            job_definition_name (str): Job definition name.
                If not specified then a default one will be generated.
            image_uri (str): The uri of the image to use for the jobs started by the Monitor.
            latest_baselining_job_name (str): name of the last baselining job.
            existing_job_desc (dict): description of existing job definition. It will be updated by
                 values that were passed in, and then used to create the new job definition.
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
                for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
                to a constraints JSON file.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): KMS key id for output.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            env (dict): Environment variables to be passed to the job.
            tags ([dict]): List of tags to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
            run the monitoring schedule on the batch transform

        Returns:
            dict: request parameters to create job definition.
        """
        if existing_job_desc is not None:
            app_specification = existing_job_desc[
                "{}AppSpecification".format(self.monitoring_type())
            ]
            baseline_config = existing_job_desc.get(
                "{}BaselineConfig".format(self.monitoring_type()), {}
            )
            job_input = existing_job_desc["{}JobInput".format(self.monitoring_type())]
            job_output = existing_job_desc["{}JobOutputConfig".format(self.monitoring_type())]
            cluster_config = existing_job_desc["JobResources"]["ClusterConfig"]
            if role is None:
                role = existing_job_desc["RoleArn"]
            existing_network_config = existing_job_desc.get("NetworkConfig")
            stop_condition = existing_job_desc.get("StoppingCondition", {})
        else:
            app_specification = {}
            baseline_config = {}
            job_input = {}
            job_output = {}
            cluster_config = {}
            existing_network_config = None
            stop_condition = {}

        # app specification
        record_preprocessor_script_s3_uri = None
        if record_preprocessor_script is not None:
            record_preprocessor_script_s3_uri = self._s3_uri_from_local_path(
                path=record_preprocessor_script
            )

        post_analytics_processor_script_s3_uri = None
        if post_analytics_processor_script is not None:
            post_analytics_processor_script_s3_uri = self._s3_uri_from_local_path(
                path=post_analytics_processor_script
            )

        app_specification["ImageUri"] = image_uri
        if post_analytics_processor_script_s3_uri:
            app_specification[
                "PostAnalyticsProcessorSourceUri"
            ] = post_analytics_processor_script_s3_uri
        if record_preprocessor_script_s3_uri:
            app_specification["RecordPreprocessorSourceUri"] = record_preprocessor_script_s3_uri

        normalized_env = self._generate_env_map(
            env=env,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics,
        )
        if normalized_env:
            app_specification["Environment"] = normalized_env

        # baseline config
        # noinspection PyTypeChecker
        statistics_object, constraints_object = self._get_baseline_files(
            statistics=statistics,
            constraints=constraints,
            sagemaker_session=self.sagemaker_session,
        )
        if constraints_object is not None:
            constraints_s3_uri = constraints_object.file_s3_uri
            baseline_config["ConstraintsResource"] = dict(S3Uri=constraints_s3_uri)
        if statistics_object is not None:
            statistics_s3_uri = statistics_object.file_s3_uri
            baseline_config["StatisticsResource"] = dict(S3Uri=statistics_s3_uri)
        # ConstraintsResource and BaseliningJobName can co-exist in BYOC case
        if latest_baselining_job_name:
            baseline_config["BaseliningJobName"] = latest_baselining_job_name

        # job input
        if endpoint_input is not None:
            normalized_endpoint_input = self._normalize_endpoint_input(
                endpoint_input=endpoint_input
            )
            job_input = normalized_endpoint_input._to_request_dict()
        elif batch_transform_input is not None:
            job_input = batch_transform_input._to_request_dict()

        # job output
        if output_s3_uri is not None:
            normalized_monitoring_output = self._normalize_monitoring_output(
                monitoring_schedule_name, output_s3_uri
            )
            job_output["MonitoringOutputs"] = [normalized_monitoring_output._to_request_dict()]
        if output_kms_key is not None:
            job_output["KmsKeyId"] = output_kms_key

        # cluster config
        if instance_count is not None:
            cluster_config["InstanceCount"] = instance_count
        if instance_type is not None:
            cluster_config["InstanceType"] = instance_type
        if volume_size_in_gb is not None:
            cluster_config["VolumeSizeInGB"] = volume_size_in_gb
        if volume_kms_key is not None:
            cluster_config["VolumeKmsKeyId"] = volume_kms_key

        # stop condition
        if max_runtime_in_seconds is not None:
            stop_condition["MaxRuntimeInSeconds"] = max_runtime_in_seconds

        request_dict = {
            "JobDefinitionName": job_definition_name,
            "{}AppSpecification".format(self.monitoring_type()): app_specification,
            "{}JobInput".format(self.monitoring_type()): job_input,
            "{}JobOutputConfig".format(self.monitoring_type()): job_output,
            "JobResources": dict(ClusterConfig=cluster_config),
            "RoleArn": self.sagemaker_session.expand_role(role),
        }

        if baseline_config:
            request_dict["{}BaselineConfig".format(self.monitoring_type())] = baseline_config

        if network_config is not None:
            network_config_dict = network_config._to_request_dict()
            request_dict["NetworkConfig"] = network_config_dict
        elif existing_network_config is not None:
            request_dict["NetworkConfig"] = existing_network_config

        if stop_condition:
            request_dict["StoppingCondition"] = stop_condition

        if tags is not None:
            request_dict["Tags"] = tags

        return request_dict


class ModelQualityMonitor(ModelMonitor):
    """Amazon SageMaker model monitor to monitor quality metrics for an endpoint.

    Please see the __init__ method of its base class for how to instantiate it.
    """

    JOB_DEFINITION_BASE_NAME = "model-quality-job-definition"

    def __init__(
        self,
        role=None,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initializes a monitor instance.

        The monitor handles baselining datasets and creating Amazon SageMaker
        Monitoring Schedules to monitor SageMaker endpoints.

        Args:
            role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            base_job_name (str): Prefix for the job name. If not specified,
                a default name is generated based on the training image name and
                current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            env (dict): Environment variables to be passed to the job.
            tags ([dict]): List of tags to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """

        session = sagemaker_session or Session()
        super(ModelQualityMonitor, self).__init__(
            role=role,
            image_uri=ModelQualityMonitor._get_default_image_uri(session.boto_session.region_name),
            instance_count=instance_count,
            instance_type=instance_type,
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

    @classmethod
    def monitoring_type(cls):
        """Type of the monitoring job."""
        return "ModelQuality"

    def suggest_baseline(
        self,
        baseline_dataset,
        dataset_format,
        problem_type,
        inference_attribute=None,
        probability_attribute=None,
        ground_truth_attribute=None,
        probability_threshold_attribute=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        wait=False,
        logs=False,
        job_name=None,
    ):
        """Suggest baselines for use with Amazon SageMaker Model Monitoring Schedules.

        Args:
            baseline_dataset (str): The path to the baseline_dataset file. This can be a local
                path or an S3 uri.
            dataset_format (dict): The format of the baseline_dataset.
            problem_type (str): The type of problem of this model quality monitoring. Valid
                values are "Regression", "BinaryClassification", "MulticlassClassification".
            inference_attribute (str): Index or JSONpath to locate predicted label(s).
                Only used for ModelQualityMonitor.
            probability_attribute (str or int): Index or JSONpath to locate probabilities.
                Only used for ModelQualityMonitor.
            ground_truth_attribute (str): Index to locate actual label(s).
                Only used for ModelQualityMonitor.
            probability_threshold_attribute (float): threshold to convert probabilities to binaries
                Only used for ModelQualityMonitor.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): Desired S3 destination Destination of the constraint_violations
                and statistics json files.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            wait (bool): Whether the call should wait until the job completes (default: False).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: False).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.

        Returns:
            sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
                baselining job.

        """
        self.latest_baselining_job_name = self._generate_baselining_job_name(job_name=job_name)

        normalized_baseline_dataset_input = self._upload_and_convert_to_processing_input(
            source=baseline_dataset,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH, _CONTAINER_INPUT_PATH, _BASELINE_DATASET_INPUT_NAME
                )
            ),
            name=_BASELINE_DATASET_INPUT_NAME,
        )

        # Unlike other input, dataset must be a directory for the Monitoring image.
        baseline_dataset_container_path = normalized_baseline_dataset_input.destination

        normalized_post_processor_script_input = self._upload_and_convert_to_processing_input(
            source=post_analytics_processor_script,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH,
                    _CONTAINER_INPUT_PATH,
                    _POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME,
                )
            ),
            name=_POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME,
        )

        post_processor_script_container_path = None
        if normalized_post_processor_script_input is not None:
            post_processor_script_container_path = str(
                pathlib.PurePosixPath(
                    normalized_post_processor_script_input.destination,
                    os.path.basename(post_analytics_processor_script),
                )
            )

        normalized_baseline_output = self._normalize_baseline_output(output_s3_uri=output_s3_uri)

        normalized_env = self._generate_env_map(
            env=self.env,
            dataset_format=dataset_format,
            output_path=normalized_baseline_output.source,
            enable_cloudwatch_metrics=False,  # Only supported for monitoring schedules
            dataset_source_container_path=baseline_dataset_container_path,
            post_processor_script_container_path=post_processor_script_container_path,
            analysis_type="MODEL_QUALITY",
            problem_type=problem_type,
            inference_attribute=inference_attribute,
            probability_attribute=probability_attribute,
            ground_truth_attribute=ground_truth_attribute,
            probability_threshold_attribute=probability_threshold_attribute,
        )

        baselining_processor = Processor(
            role=self.role,
            image_uri=self.image_uri,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            entrypoint=self.entrypoint,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            output_kms_key=self.output_kms_key,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            base_job_name=self.base_job_name,
            sagemaker_session=self.sagemaker_session,
            env=normalized_env,
            tags=self.tags,
            network_config=self.network_config,
        )

        baseline_job_inputs_with_nones = [
            normalized_baseline_dataset_input,
            normalized_post_processor_script_input,
        ]

        baseline_job_inputs = [
            baseline_job_input
            for baseline_job_input in baseline_job_inputs_with_nones
            if baseline_job_input is not None
        ]

        baselining_processor.run(
            inputs=baseline_job_inputs,
            outputs=[normalized_baseline_output],
            arguments=self.arguments,
            wait=wait,
            logs=logs,
            job_name=self.latest_baselining_job_name,
        )

        self.latest_baselining_job = BaseliningJob.from_processing_job(
            processing_job=baselining_processor.latest_job
        )
        self.baselining_jobs.append(self.latest_baselining_job)
        return baselining_processor.latest_job

    # noinspection PyMethodOverriding
    def create_monitoring_schedule(
        self,
        endpoint_input=None,
        ground_truth_input=None,
        problem_type=None,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        constraints=None,
        monitor_schedule_name=None,
        schedule_cron_expression=None,
        enable_cloudwatch_metrics=True,
        batch_transform_input=None,
    ):
        """Creates a monitoring schedule.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to
                monitor. This can either be the endpoint name or an EndpointInput.
                (default: None)
            ground_truth_input (str): S3 URI to ground truth dataset.
                (default: None)
            problem_type (str): The type of problem of this model quality monitoring. Valid
                values are "Regression", "BinaryClassification", "MulticlassClassification".
                (default: None)
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
                for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
                to a constraints JSON file.
            monitor_schedule_name (str): Schedule name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
                run the monitoring schedule on the batch transform
        """
        # we default below two parameters to None in the function signature
        # but verify they are giving here for positional argument
        # backward compatibility reason.
        if not ground_truth_input:
            raise ValueError("ground_truth_input can not be None.")
        if not problem_type:
            raise ValueError("problem_type can not be None.")

        if self.job_definition_name is not None or self.monitoring_schedule_name is not None:
            message = (
                "It seems that this object was already used to create an Amazon Model "
                "Monitoring Schedule. To create another, first delete the existing one "
                "using my_monitor.delete_monitoring_schedule()."
            )
            _LOGGER.error(message)
            raise ValueError(message)

        if (batch_transform_input is not None) ^ (endpoint_input is None):
            message = (
                "Need to have either batch_transform_input or endpoint_input to create an "
                "Amazon Model Monitoring Schedule. "
                "Please provide only one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        # create job definition
        monitor_schedule_name = self._generate_monitoring_schedule_name(
            schedule_name=monitor_schedule_name
        )
        new_job_definition_name = name_from_base(self.JOB_DEFINITION_BASE_NAME)
        request_dict = self._build_create_model_quality_job_definition_request(
            monitoring_schedule_name=monitor_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self.image_uri,
            latest_baselining_job_name=self.latest_baselining_job_name,
            endpoint_input=endpoint_input,
            ground_truth_input=ground_truth_input,
            problem_type=problem_type,
            record_preprocessor_script=record_preprocessor_script,
            post_analytics_processor_script=post_analytics_processor_script,
            output_s3_uri=self._normalize_monitoring_output(
                monitor_schedule_name, output_s3_uri
            ).destination,
            constraints=constraints,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics,
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            output_kms_key=self.output_kms_key,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            env=self.env,
            tags=self.tags,
            network_config=self.network_config,
            batch_transform_input=batch_transform_input,
        )
        self.sagemaker_session.sagemaker_client.create_model_quality_job_definition(**request_dict)

        # create schedule
        try:
            self._create_monitoring_schedule_from_job_definition(
                monitor_schedule_name=monitor_schedule_name,
                job_definition_name=new_job_definition_name,
                schedule_cron_expression=schedule_cron_expression,
            )
            self.job_definition_name = new_job_definition_name
            self.monitoring_schedule_name = monitor_schedule_name
        except Exception:
            _LOGGER.exception("Failed to create monitoring schedule.")
            # noinspection PyBroadException
            try:
                self.sagemaker_session.sagemaker_client.delete_model_quality_job_definition(
                    JobDefinitionName=new_job_definition_name
                )
            except Exception:  # pylint: disable=W0703
                message = "Failed to delete job definition {}.".format(new_job_definition_name)
                _LOGGER.exception(message)
            raise

    def update_monitoring_schedule(
        self,
        endpoint_input=None,
        ground_truth_input=None,
        problem_type=None,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        constraints=None,
        schedule_cron_expression=None,
        enable_cloudwatch_metrics=None,
        role=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        env=None,
        network_config=None,
        batch_transform_input=None,
    ):
        """Updates the existing monitoring schedule.

        If more options than schedule_cron_expression are to be updated, a new job definition will
        be created to hold them. The old job definition will not be deleted.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint
                to monitor. This can either be the endpoint name or an EndpointInput.
            ground_truth_input (str): S3 URI to ground truth dataset.
            problem_type (str): The type of problem of this model quality monitoring. Valid values
                are "Regression", "BinaryClassification", "MulticlassClassification".
            record_preprocessor_script (str): The path to the record preprocessor script. This can
                be a local path or an S3 uri.
            post_analytics_processor_script (str): The path to the record post-analytics processor
                script. This can be a local path or an S3 uri.
            output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
                for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
                to a constraints JSON file.
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): The KMS key id for the job's outputs.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            env (dict): Environment variables to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
                run the monitoring schedule on the batch transform
        """
        valid_args = {
            arg: value for arg, value in locals().items() if arg != "self" and value is not None
        }

        # Nothing to update
        if len(valid_args) <= 0:
            return

        # Only need to update schedule expression
        if len(valid_args) == 1 and schedule_cron_expression is not None:
            self._update_monitoring_schedule(self.job_definition_name, schedule_cron_expression)
            return

        if (batch_transform_input is not None) and (endpoint_input is not None):
            message = (
                "Cannot update both batch_transform_input and endpoint_input to update an "
                "Amazon Model Monitoring Schedule. "
                "Please provide atmost one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        # Need to update schedule with a new job definition
        job_desc = self.sagemaker_session.sagemaker_client.describe_model_quality_job_definition(
            JobDefinitionName=self.job_definition_name
        )
        new_job_definition_name = name_from_base(self.JOB_DEFINITION_BASE_NAME)
        request_dict = self._build_create_model_quality_job_definition_request(
            monitoring_schedule_name=self.monitoring_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self.image_uri,
            existing_job_desc=job_desc,
            endpoint_input=endpoint_input,
            ground_truth_input=ground_truth_input,
            problem_type=problem_type,
            record_preprocessor_script=record_preprocessor_script,
            post_analytics_processor_script=post_analytics_processor_script,
            output_s3_uri=output_s3_uri,
            constraints=constraints,
            enable_cloudwatch_metrics=enable_cloudwatch_metrics,
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            env=env,
            tags=self.tags,
            network_config=network_config,
            batch_transform_input=batch_transform_input,
        )
        self.sagemaker_session.sagemaker_client.create_model_quality_job_definition(**request_dict)
        try:
            self._update_monitoring_schedule(new_job_definition_name, schedule_cron_expression)
            self.job_definition_name = new_job_definition_name
            if role is not None:
                self.role = role
            if instance_count is not None:
                self.instance_count = instance_count
            if instance_type is not None:
                self.instance_type = instance_type
            if volume_size_in_gb is not None:
                self.volume_size_in_gb = volume_size_in_gb
            if volume_kms_key is not None:
                self.volume_kms_key = volume_kms_key
            if output_kms_key is not None:
                self.output_kms_key = output_kms_key
            if max_runtime_in_seconds is not None:
                self.max_runtime_in_seconds = max_runtime_in_seconds
            if env is not None:
                self.env = env
            if network_config is not None:
                self.network_config = network_config
        except Exception:
            _LOGGER.exception("Failed to update monitoring schedule.")
            # noinspection PyBroadException
            try:
                self.sagemaker_session.sagemaker_client.delete_model_quality_job_definition(
                    JobDefinitionName=new_job_definition_name
                )
            except Exception:  # pylint: disable=W0703
                message = "Failed to delete job definition {}.".format(new_job_definition_name)
                _LOGGER.exception(message)
            raise

    def delete_monitoring_schedule(self):
        """Deletes the monitoring schedule and its job definition."""
        super(ModelQualityMonitor, self).delete_monitoring_schedule()
        # Delete job definition.
        message = "Deleting Model Quality Job Definition with name: {}".format(
            self.job_definition_name
        )
        _LOGGER.info(message)
        self.sagemaker_session.sagemaker_client.delete_model_quality_job_definition(
            JobDefinitionName=self.job_definition_name
        )
        self.job_definition_name = None

    @classmethod
    def attach(cls, monitor_schedule_name, sagemaker_session=None):
        """Sets this object's schedule name to the name provided.

        This allows subsequent describe_schedule or list_executions calls to point
        to the given schedule.

        Args:
            monitor_schedule_name (str): The name of the schedule to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or Session()
        schedule_desc = sagemaker_session.describe_monitoring_schedule(
            monitoring_schedule_name=monitor_schedule_name
        )
        monitoring_type = schedule_desc["MonitoringScheduleConfig"].get("MonitoringType")
        if monitoring_type != cls.monitoring_type():
            raise TypeError(
                "{} can only attach to ModelQuality monitoring schedule.".format(__class__.__name__)
            )
        job_definition_name = schedule_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinitionName"
        ]
        job_desc = sagemaker_session.sagemaker_client.describe_model_quality_job_definition(
            JobDefinitionName=job_definition_name
        )
        tags = sagemaker_session.list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])
        return ModelMonitor._attach(
            clazz=cls,
            sagemaker_session=sagemaker_session,
            schedule_desc=schedule_desc,
            job_desc=job_desc,
            tags=tags,
        )

    def _build_create_model_quality_job_definition_request(
        self,
        monitoring_schedule_name,
        job_definition_name,
        image_uri,
        latest_baselining_job_name=None,
        existing_job_desc=None,
        endpoint_input=None,
        ground_truth_input=None,
        problem_type=None,
        record_preprocessor_script=None,
        post_analytics_processor_script=None,
        output_s3_uri=None,
        constraints=None,
        enable_cloudwatch_metrics=None,
        role=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=None,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        env=None,
        tags=None,
        network_config=None,
        batch_transform_input=None,
    ):
        """Build the request for job definition creation API

        Args:
            monitoring_schedule_name (str): Monitoring schedule name.
            job_definition_name (str): Job definition name.
                If not specified then a default one will be generated.
            image_uri (str): The uri of the image to use for the jobs started by the Monitor.
            latest_baselining_job_name (str): name of the last baselining job.
            existing_job_desc (dict): description of existing job definition. It will be updated by
                 values that were passed in, and then used to create the new job definition.
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            ground_truth_input (str): S3 URI to ground truth dataset.
            problem_type (str): The type of problem of this model quality monitoring. Valid
                values are "Regression", "BinaryClassification", "MulticlassClassification".
            output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
                for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
                to a constraints JSON file.
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs.
            role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
            instance_count (int): The number of instances to run
                the jobs with.
            instance_type (str): Type of EC2 instance to use for
                the job, for example, 'ml.m5.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the job's volume.
            output_kms_key (str): KMS key id for output.
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600
            env (dict): Environment variables to be passed to the job.
            tags ([dict]): List of tags to be passed to the job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to
                run the monitoring schedule on the batch transform

        Returns:
            dict: request parameters to create job definition.
        """
        if existing_job_desc is not None:
            app_specification = existing_job_desc[
                "{}AppSpecification".format(self.monitoring_type())
            ]
            baseline_config = existing_job_desc.get(
                "{}BaselineConfig".format(self.monitoring_type()), {}
            )
            job_input = existing_job_desc["{}JobInput".format(self.monitoring_type())]
            job_output = existing_job_desc["{}JobOutputConfig".format(self.monitoring_type())]
            cluster_config = existing_job_desc["JobResources"]["ClusterConfig"]
            if role is None:
                role = existing_job_desc["RoleArn"]
            existing_network_config = existing_job_desc.get("NetworkConfig")
            stop_condition = existing_job_desc.get("StoppingCondition", {})
        else:
            app_specification = {}
            baseline_config = {}
            job_input = {}
            job_output = {}
            cluster_config = {}
            existing_network_config = None
            stop_condition = {}

        # app specification
        app_specification["ImageUri"] = image_uri
        if problem_type is not None:
            app_specification["ProblemType"] = problem_type
        record_preprocessor_script_s3_uri = None
        if record_preprocessor_script is not None:
            record_preprocessor_script_s3_uri = self._s3_uri_from_local_path(
                path=record_preprocessor_script
            )

        post_analytics_processor_script_s3_uri = None
        if post_analytics_processor_script is not None:
            post_analytics_processor_script_s3_uri = self._s3_uri_from_local_path(
                path=post_analytics_processor_script
            )

        if post_analytics_processor_script_s3_uri:
            app_specification[
                "PostAnalyticsProcessorSourceUri"
            ] = post_analytics_processor_script_s3_uri
        if record_preprocessor_script_s3_uri:
            app_specification["RecordPreprocessorSourceUri"] = record_preprocessor_script_s3_uri

        normalized_env = self._generate_env_map(
            env=env, enable_cloudwatch_metrics=enable_cloudwatch_metrics
        )
        if normalized_env:
            app_specification["Environment"] = normalized_env

        # baseline config
        if constraints:
            # noinspection PyTypeChecker
            _, constraints_object = self._get_baseline_files(
                statistics=None, constraints=constraints, sagemaker_session=self.sagemaker_session
            )
            constraints_s3_uri = None
            if constraints_object is not None:
                constraints_s3_uri = constraints_object.file_s3_uri
            baseline_config["ConstraintsResource"] = dict(S3Uri=constraints_s3_uri)
        if latest_baselining_job_name:
            baseline_config["BaseliningJobName"] = latest_baselining_job_name

        # job input
        if endpoint_input is not None:
            normalized_endpoint_input = self._normalize_endpoint_input(
                endpoint_input=endpoint_input
            )
            job_input = normalized_endpoint_input._to_request_dict()
        elif batch_transform_input is not None:
            job_input = batch_transform_input._to_request_dict()

        if ground_truth_input is not None:
            job_input["GroundTruthS3Input"] = dict(S3Uri=ground_truth_input)

        # job output
        if output_s3_uri is not None:
            normalized_monitoring_output = self._normalize_monitoring_output(
                monitoring_schedule_name, output_s3_uri
            )
            job_output["MonitoringOutputs"] = [normalized_monitoring_output._to_request_dict()]
        if output_kms_key is not None:
            job_output["KmsKeyId"] = output_kms_key

        # cluster config
        if instance_count is not None:
            cluster_config["InstanceCount"] = instance_count
        if instance_type is not None:
            cluster_config["InstanceType"] = instance_type
        if volume_size_in_gb is not None:
            cluster_config["VolumeSizeInGB"] = volume_size_in_gb
        if volume_kms_key is not None:
            cluster_config["VolumeKmsKeyId"] = volume_kms_key

        # stop condition
        if max_runtime_in_seconds is not None:
            stop_condition["MaxRuntimeInSeconds"] = max_runtime_in_seconds

        request_dict = {
            "JobDefinitionName": job_definition_name,
            "{}AppSpecification".format(self.monitoring_type()): app_specification,
            "{}JobInput".format(self.monitoring_type()): job_input,
            "{}JobOutputConfig".format(self.monitoring_type()): job_output,
            "JobResources": dict(ClusterConfig=cluster_config),
            "RoleArn": self.sagemaker_session.expand_role(role),
        }

        if baseline_config:
            request_dict["{}BaselineConfig".format(self.monitoring_type())] = baseline_config

        if network_config is not None:
            network_config_dict = network_config._to_request_dict()
            request_dict["NetworkConfig"] = network_config_dict
        elif existing_network_config is not None:
            request_dict["NetworkConfig"] = existing_network_config

        if stop_condition:
            request_dict["StoppingCondition"] = stop_condition

        if tags is not None:
            request_dict["Tags"] = tags

        return request_dict

    @staticmethod
    def _get_default_image_uri(region):
        """Returns the Default Model Monitoring image uri based on the region.

        Args:
            region (str): The AWS region.

        Returns:
            str: The Default Model Monitoring image uri based on the region.
        """
        return image_uris.retrieve(framework=framework_name, region=region)


class BaseliningJob(ProcessingJob):
    """Provides functionality to retrieve baseline-specific files output from baselining job."""

    def __init__(self, sagemaker_session, job_name, inputs, outputs, output_kms_key=None):
        """Initializes a Baselining job.

        It tracks a baselining job kicked off by the suggest workflow.

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            job_name (str): Name of the Amazon SageMaker Model Monitoring Baselining Job.
            inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput objects.
            outputs ([sagemaker.processing.ProcessingOutput]): A list of ProcessingOutput objects.
            output_kms_key (str): The output kms key associated with the job. Defaults to None
                if not provided.

        """
        self.inputs = inputs
        self.outputs = outputs
        super(BaseliningJob, self).__init__(
            sagemaker_session=sagemaker_session,
            job_name=job_name,
            inputs=inputs,
            outputs=outputs,
            output_kms_key=output_kms_key,
        )

    @classmethod
    def from_processing_job(cls, processing_job):
        """Initializes a Baselining job from a processing job.

        Args:
            processing_job (sagemaker.processing.ProcessingJob): The ProcessingJob used for
                baselining instance.

        Returns:
            sagemaker.processing.BaseliningJob: The instance of ProcessingJob created
                using the current job name.

        """
        return cls(
            processing_job.sagemaker_session,
            processing_job.job_name,
            processing_job.inputs,
            processing_job.outputs,
            processing_job.output_kms_key,
        )

    def baseline_statistics(self, file_name=STATISTICS_JSON_DEFAULT_FILE_NAME, kms_key=None):
        """Returns a sagemaker.model_monitor.

        Statistics object representing the statistics JSON file generated by this baselining job.

        Args:
            file_name (str): The name of the json-formatted statistics file
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.Statistics: The Statistics object representing the file that
                was generated by the job.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.

        """
        try:
            baselining_job_output_s3_path = self.outputs[0].destination
            return Statistics.from_s3_uri(
                statistics_file_s3_uri=s3.s3_path_join(baselining_job_output_s3_path, file_name),
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        except ClientError as client_error:
            if client_error.response["Error"]["Code"] == "NoSuchKey":
                status = self.sagemaker_session.describe_processing_job(job_name=self.job_name)[
                    "ProcessingJobStatus"
                ]
                if status != "Completed":
                    raise UnexpectedStatusException(
                        message="The underlying job is not in 'Completed' state. You may only "
                        "retrieve files for a job that has completed successfully.",
                        allowed_statuses="Completed",
                        actual_status=status,
                    )
            else:
                raise client_error

    def suggested_constraints(self, file_name=CONSTRAINTS_JSON_DEFAULT_FILE_NAME, kms_key=None):
        """Returns a sagemaker.model_monitor.

        Constraints object representing the constraints JSON file generated by this baselining job.

        Args:
            file_name (str): The name of the json-formatted constraints file
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.Constraints: The Constraints object representing the file that
                was generated by the job.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.

        """
        try:
            baselining_job_output_s3_path = self.outputs[0].destination
            return Constraints.from_s3_uri(
                constraints_file_s3_uri=s3.s3_path_join(baselining_job_output_s3_path, file_name),
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        except ClientError as client_error:
            if client_error.response["Error"]["Code"] == "NoSuchKey":
                status = self.sagemaker_session.describe_processing_job(job_name=self.job_name)[
                    "ProcessingJobStatus"
                ]
                if status != "Completed":
                    raise UnexpectedStatusException(
                        message="The underlying job is not in 'Completed' state. You may only "
                        "retrieve files for a job that has completed successfully.",
                        allowed_statuses="Completed",
                        actual_status=status,
                    )
            else:
                raise client_error


class MonitoringExecution(ProcessingJob):
    """Provides functionality to retrieve monitoring-specific files from monitoring executions."""

    def __init__(self, sagemaker_session, job_name, inputs, output, output_kms_key=None):
        """Initializes a MonitoringExecution job that tracks a monitoring execution.

        Its kicked off by an Amazon SageMaker Model Monitoring Schedule.

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            job_name (str): The name of the monitoring execution job.
            output (sagemaker.Processing.ProcessingOutput): The output associated with the
                monitoring execution.
            output_kms_key (str): The output kms key associated with the job. Defaults to None
                if not provided.

        """
        self.output = output
        super(MonitoringExecution, self).__init__(
            sagemaker_session=sagemaker_session,
            job_name=job_name,
            inputs=inputs,
            outputs=[output],
            output_kms_key=output_kms_key,
        )

    @classmethod
    def from_processing_arn(cls, sagemaker_session, processing_job_arn):
        """Initializes a Baselining job from a processing arn.

        Args:
            processing_job_arn (str): ARN of the processing job to create a MonitoringExecution
            out of.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.

        Returns:
            sagemaker.processing.BaseliningJob: The instance of ProcessingJob created
                using the current job name.

        """
        processing_job_name = processing_job_arn.split(":")[5][
            len("processing-job/") :
        ]  # This is necessary while the API only vends an arn.
        job_desc = sagemaker_session.describe_processing_job(job_name=processing_job_name)

        output_config = job_desc["ProcessingOutputConfig"]["Outputs"][0]
        return cls(
            sagemaker_session=sagemaker_session,
            job_name=processing_job_name,
            inputs=[
                ProcessingInput(
                    source=processing_input["S3Input"]["S3Uri"],
                    destination=processing_input["S3Input"]["LocalPath"],
                    input_name=processing_input["InputName"],
                    s3_data_type=processing_input["S3Input"].get("S3DataType"),
                    s3_input_mode=processing_input["S3Input"].get("S3InputMode"),
                    s3_data_distribution_type=processing_input["S3Input"].get(
                        "S3DataDistributionType"
                    ),
                    s3_compression_type=processing_input["S3Input"].get("S3CompressionType"),
                )
                for processing_input in job_desc["ProcessingInputs"]
            ],
            output=ProcessingOutput(
                source=output_config["S3Output"]["LocalPath"],
                destination=output_config["S3Output"]["S3Uri"],
                output_name=output_config["OutputName"],
            ),
            output_kms_key=job_desc["ProcessingOutputConfig"].get("KmsKeyId"),
        )

    def statistics(self, file_name=STATISTICS_JSON_DEFAULT_FILE_NAME, kms_key=None):
        """Returns a sagemaker.model_monitor.

        Statistics object representing the statistics
        JSON file generated by this monitoring execution.

        Args:
            file_name (str): The name of the json-formatted statistics file
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.Statistics: The Statistics object representing the file that
                was generated by the execution.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.

        """
        try:
            baselining_job_output_s3_path = self.outputs[0].destination
            return Statistics.from_s3_uri(
                statistics_file_s3_uri=s3.s3_path_join(baselining_job_output_s3_path, file_name),
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        except ClientError as client_error:
            if client_error.response["Error"]["Code"] == "NoSuchKey":
                status = self.sagemaker_session.describe_processing_job(job_name=self.job_name)[
                    "ProcessingJobStatus"
                ]
                if status != "Completed":
                    raise UnexpectedStatusException(
                        message="The underlying job is not in 'Completed' state. You may only "
                        "retrieve files for a job that has completed successfully.",
                        allowed_statuses="Completed",
                        actual_status=status,
                    )
            else:
                raise client_error

    def constraint_violations(
        self, file_name=CONSTRAINT_VIOLATIONS_JSON_DEFAULT_FILE_NAME, kms_key=None
    ):
        """Returns a sagemaker.model_monitor.

        ConstraintViolations object representing the constraint violations
        JSON file generated by this monitoring execution.

        Args:
            file_name (str): The name of the json-formatted constraint violations file.
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.ConstraintViolations: The ConstraintViolations object
                representing the file that was generated by the monitoring execution.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.

        """
        try:
            baselining_job_output_s3_path = self.outputs[0].destination
            return ConstraintViolations.from_s3_uri(
                constraint_violations_file_s3_uri=s3.s3_path_join(
                    baselining_job_output_s3_path, file_name
                ),
                kms_key=kms_key,
                sagemaker_session=self.sagemaker_session,
            )
        except ClientError as client_error:
            if client_error.response["Error"]["Code"] == "NoSuchKey":
                status = self.sagemaker_session.describe_processing_job(job_name=self.job_name)[
                    "ProcessingJobStatus"
                ]
                if status != "Completed":
                    raise UnexpectedStatusException(
                        message="The underlying job is not in 'Completed' state. You may only "
                        "retrieve files for a job that has completed successfully.",
                        allowed_statuses="Completed",
                        actual_status=status,
                    )
            else:
                raise client_error


class EndpointInput(object):
    """Accepts parameters that specify an endpoint input for monitoring execution.

    It also provides a method to turn those parameters into a dictionary.
    """

    def __init__(
        self,
        endpoint_name,
        destination,
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
        start_time_offset=None,
        end_time_offset=None,
        features_attribute=None,
        inference_attribute=None,
        probability_attribute=None,
        probability_threshold_attribute=None,
    ):
        """Initialize an ``EndpointInput`` instance.

        EndpointInput accepts parameters that specify an endpoint input for a monitoring
        job and provides a method to turn those parameters into a dictionary.

        Args:
            endpoint_name (str): The name of the endpoint.
            destination (str): The destination of the input.
            s3_input_mode (str): The S3 input mode. Can be one of: "File", "Pipe. Default: "File".
            s3_data_distribution_type (str): The S3 Data Distribution Type. Can be one of:
                "FullyReplicated", "ShardedByS3Key"
            start_time_offset (str): Monitoring start time offset, e.g. "-PT1H"
            end_time_offset (str): Monitoring end time offset, e.g. "-PT0H".
            features_attribute (str): JSONpath to locate features in JSONlines dataset.
                Only used for ModelBiasMonitor and ModelExplainabilityMonitor
            inference_attribute (str): Index or JSONpath to locate predicted label(s).
                Only used for ModelQualityMonitor, ModelBiasMonitor, and ModelExplainabilityMonitor
            probability_attribute (str or int): Index or JSONpath to locate probabilities.
                Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
            probability_threshold_attribute (float): threshold to convert probabilities to binaries
                Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
        """
        self.endpoint_name = endpoint_name
        self.destination = destination
        self.s3_input_mode = s3_input_mode
        self.s3_data_distribution_type = s3_data_distribution_type
        self.start_time_offset = start_time_offset
        self.end_time_offset = end_time_offset
        self.features_attribute = features_attribute
        self.inference_attribute = inference_attribute
        self.probability_attribute = probability_attribute
        self.probability_threshold_attribute = probability_threshold_attribute

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        endpoint_input = {
            "EndpointName": self.endpoint_name,
            "LocalPath": self.destination,
            "S3InputMode": self.s3_input_mode,
            "S3DataDistributionType": self.s3_data_distribution_type,
        }

        if self.start_time_offset is not None:
            endpoint_input["StartTimeOffset"] = self.start_time_offset
        if self.end_time_offset is not None:
            endpoint_input["EndTimeOffset"] = self.end_time_offset
        if self.features_attribute is not None:
            endpoint_input["FeaturesAttribute"] = self.features_attribute
        if self.inference_attribute is not None:
            endpoint_input["InferenceAttribute"] = self.inference_attribute
        if self.probability_attribute is not None:
            endpoint_input["ProbabilityAttribute"] = self.probability_attribute
        if self.probability_threshold_attribute is not None:
            endpoint_input["ProbabilityThresholdAttribute"] = self.probability_threshold_attribute

        endpoint_input_request = {"EndpointInput": endpoint_input}
        return endpoint_input_request


@attr.s
class MonitoringInput(object):
    """Accepts parameters specifying batch transform or endpoint inputs for monitoring execution.

    MonitoringInput accepts parameters that specify additional parameters while monitoring jobs.
    It also provides a method to turn those parameters into a dictionary.

    Args:
        start_time_offset (str): Monitoring start time offset, e.g. "-PT1H"
        end_time_offset (str): Monitoring end time offset, e.g. "-PT0H".
        features_attribute (str): JSONpath to locate features in JSONlines dataset.
            Only used for ModelBiasMonitor and ModelExplainabilityMonitor
        inference_attribute (str): Index or JSONpath to locate predicted label(s).
            Only used for ModelQualityMonitor, ModelBiasMonitor, and ModelExplainabilityMonitor
        probability_attribute (str): Index or JSONpath to locate probabilities.
            Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
        probability_threshold_attribute (float): threshold to convert probabilities to binaries
            Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
    """

    start_time_offset: str = attr.ib()
    end_time_offset: str = attr.ib()
    features_attribute: str = attr.ib()
    inference_attribute: str = attr.ib()
    probability_attribute: Union[str, int] = attr.ib()
    probability_threshold_attribute: float = attr.ib()


class BatchTransformInput(MonitoringInput):
    """Accepts parameters that specify a batch transform input for monitoring schedule.

    It also provides a method to turn those parameters into a dictionary.
    """

    def __init__(
        self,
        data_captured_destination_s3_uri: str,
        destination: str,
        dataset_format: MonitoringDatasetFormat,
        s3_input_mode: str = "File",
        s3_data_distribution_type: str = "FullyReplicated",
        start_time_offset: str = None,
        end_time_offset: str = None,
        features_attribute: str = None,
        inference_attribute: str = None,
        probability_attribute: str = None,
        probability_threshold_attribute: str = None,
    ):
        """Initialize a `BatchTransformInput` instance.

        Args:
            data_captured_destination_s3_uri (str): Location to the batch transform captured data
                file which needs to be analysed.
            destination (str): The destination of the input.
            s3_input_mode (str): The S3 input mode. Can be one of: "File", "Pipe. (default: File)
            s3_data_distribution_type (str): The S3 Data Distribution Type. Can be one of:
                "FullyReplicated", "ShardedByS3Key" (default: FullyReplicated)
            start_time_offset (str): Monitoring start time offset, e.g. "-PT1H" (default: None)
            end_time_offset (str): Monitoring end time offset, e.g. "-PT0H". (default: None)
            features_attribute (str): JSONpath to locate features in JSONlines dataset.
                Only used for ModelBiasMonitor and ModelExplainabilityMonitor (default: None)
            inference_attribute (str): Index or JSONpath to locate predicted label(s).
                Only used for ModelQualityMonitor, ModelBiasMonitor, and ModelExplainabilityMonitor
                (default: None)
            probability_attribute (str): Index or JSONpath to locate probabilities.
                Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
                (default: None)
            probability_threshold_attribute (float): threshold to convert probabilities to binaries
                Only used for ModelQualityMonitor, ModelBiasMonitor and ModelExplainabilityMonitor
                (default: None)

        """
        self.data_captured_destination_s3_uri = data_captured_destination_s3_uri
        self.destination = destination
        self.s3_input_mode = s3_input_mode
        self.s3_data_distribution_type = s3_data_distribution_type
        self.dataset_format = dataset_format

        super(BatchTransformInput, self).__init__(
            start_time_offset=start_time_offset,
            end_time_offset=end_time_offset,
            features_attribute=features_attribute,
            inference_attribute=inference_attribute,
            probability_attribute=probability_attribute,
            probability_threshold_attribute=probability_threshold_attribute,
        )

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        batch_transform_input_data = {
            "DataCapturedDestinationS3Uri": self.data_captured_destination_s3_uri,
            "LocalPath": self.destination,
            "S3InputMode": self.s3_input_mode,
            "S3DataDistributionType": self.s3_data_distribution_type,
            "DatasetFormat": self.dataset_format,
        }

        if self.start_time_offset is not None:
            batch_transform_input_data["StartTimeOffset"] = self.start_time_offset
        if self.end_time_offset is not None:
            batch_transform_input_data["EndTimeOffset"] = self.end_time_offset
        if self.features_attribute is not None:
            batch_transform_input_data["FeaturesAttribute"] = self.features_attribute
        if self.inference_attribute is not None:
            batch_transform_input_data["InferenceAttribute"] = self.inference_attribute
        if self.probability_attribute is not None:
            batch_transform_input_data["ProbabilityAttribute"] = self.probability_attribute
        if self.probability_threshold_attribute is not None:
            batch_transform_input_data[
                "ProbabilityThresholdAttribute"
            ] = self.probability_threshold_attribute

        batch_transform_input_request = {"BatchTransformInput": batch_transform_input_data}

        return batch_transform_input_request


class MonitoringOutput(object):
    """Accepts parameters that specify an S3 output for a monitoring job.

    It also provides a method to turn those parameters into a dictionary.
    """

    def __init__(self, source, destination=None, s3_upload_mode="Continuous"):
        """Initialize a ``MonitoringOutput`` instance.

        MonitoringOutput accepts parameters that specify an S3 output for a monitoring
        job and provides a method to turn those parameters into a dictionary.

        Args:
            source (str): The source for the output.
            destination (str): The destination of the output. Optional.
                Default: s3://<default-session-bucket/schedule_name/output
            s3_upload_mode (str): The S3 upload mode.

        """
        self.source = source
        self.destination = destination
        self.s3_upload_mode = s3_upload_mode

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class.

        Returns:
            dict: The request dictionary.

        """
        s3_output_request = {
            "S3Output": {
                "S3Uri": self.destination,
                "LocalPath": self.source,
                "S3UploadMode": self.s3_upload_mode,
            }
        }

        return s3_output_request
