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
"""This module contains code related to Amazon SageMaker Explainability AI Model Monitoring.

These classes assist with suggesting baselines and creating monitoring schedules for monitoring
bias metrics and feature attribution of SageMaker Endpoints.
"""
from __future__ import print_function, absolute_import

import copy
import json
import logging
import uuid

from sagemaker.model_monitor import model_monitoring as mm
from sagemaker import image_uris, s3
from sagemaker.session import Session
from sagemaker.utils import name_from_base
from sagemaker.clarify import SageMakerClarifyProcessor, ModelPredictedLabelConfig
from sagemaker.lineage._utils import get_resource_name_from_arn

_LOGGER = logging.getLogger(__name__)


class ClarifyModelMonitor(mm.ModelMonitor):
    """Base class of Amazon SageMaker Explainability API model monitors.

    This class is an ``abstract base class``, please instantiate its subclasses
    if you want to monitor bias metrics or feature attribution of an endpoint.
    """

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
        if type(self) == __class__:  # pylint: disable=unidiomatic-typecheck
            raise TypeError(
                "{} is abstract, please instantiate its subclasses instead.".format(
                    __class__.__name__
                )
            )

        session = sagemaker_session or Session()
        clarify_image_uri = image_uris.retrieve("clarify", session.boto_session.region_name)

        super(ClarifyModelMonitor, self).__init__(
            role=role,
            image_uri=clarify_image_uri,
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
        self.latest_baselining_job_config = None

    def run_baseline(self, **_):
        """Not implemented.

        '.run_baseline()' is only allowed for ModelMonitor objects.
        Please use `suggest_baseline` instead.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "'.run_baseline()' is only allowed for ModelMonitor objects."
            "Please use suggest_baseline instead."
        )

    def latest_monitoring_statistics(self, **_):
        """Not implemented.

        The class doesn't support statistics.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("{} doesn't support statistics.".format(self.__class__.__name__))

    def list_executions(self):
        """Get the list of the latest monitoring executions in descending order of "ScheduledTime".

        Returns:
            [sagemaker.model_monitor.ClarifyMonitoringExecution]: List of
                ClarifyMonitoringExecution in descending order of "ScheduledTime".
        """
        executions = super(ClarifyModelMonitor, self).list_executions()
        return [
            ClarifyMonitoringExecution(
                sagemaker_session=execution.sagemaker_session,
                job_name=execution.job_name,
                inputs=execution.inputs,
                output=execution.output,
                output_kms_key=execution.output_kms_key,
            )
            for execution in executions
        ]

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

    def _create_baselining_processor(self):
        """Create and return a SageMakerClarifyProcessor object which will run the baselining job.

        Returns:
            sagemaker.clarify.SageMakerClarifyProcessor object.
        """

        baselining_processor = SageMakerClarifyProcessor(
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            volume_size_in_gb=self.volume_size_in_gb,
            volume_kms_key=self.volume_kms_key,
            output_kms_key=self.output_kms_key,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            sagemaker_session=self.sagemaker_session,
            env=self.env,
            tags=self.tags,
            network_config=self.network_config,
        )
        baselining_processor.image_uri = self.image_uri
        baselining_processor.base_job_name = self.base_job_name
        return baselining_processor

    def _upload_analysis_config(self, analysis_config, output_s3_uri, job_definition_name, kms_key):
        """Upload analysis config to s3://<output path>/<job name>/analysis_config.json

        Args:
            analysis_config (dict): analysis config of a Clarify model monitor.
            output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
                Default: "s3://<default_session_bucket>/<job_name>/output"
            job_definition_name (str): Job definition name.
                If not specified then a default one will be generated.
            kms_key( str): The ARN of the KMS key that is used to encrypt the
            user code file (default: None).

        Returns:
            str: The S3 uri of the uploaded file(s).
        """
        s3_uri = s3.s3_path_join(
            output_s3_uri,
            job_definition_name,
            str(uuid.uuid4()),
            "analysis_config.json",
        )
        _LOGGER.info("Uploading analysis config to {s3_uri}.")
        return s3.S3Uploader.upload_string_as_file_body(
            json.dumps(analysis_config),
            desired_s3_uri=s3_uri,
            sagemaker_session=self.sagemaker_session,
            kms_key=kms_key,
        )

    def _build_create_job_definition_request(
        self,
        monitoring_schedule_name,
        job_definition_name,
        image_uri,
        latest_baselining_job_name=None,
        latest_baselining_job_config=None,
        existing_job_desc=None,
        endpoint_input=None,
        ground_truth_input=None,
        analysis_config=None,
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
            latest_baselining_job_config (ClarifyBaseliningConfig): analysis config from
                 last baselining job.
            existing_job_desc (dict): description of existing job definition. It will be updated by
                 values that were passed in, and then used to create the new job definition.
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            ground_truth_input (str): S3 URI to ground truth dataset.
            analysis_config (str or BiasAnalysisConfig or ExplainabilityAnalysisConfig): URI to the
                analysis_config.json for the bias job. If it is None then configuration of latest
                baselining job config will be reused. If no baselining job then fail the call.
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
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to run
                the monitoring schedule on the batch transform

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

        # job output
        if output_s3_uri is not None:
            normalized_monitoring_output = self._normalize_monitoring_output(
                monitoring_schedule_name, output_s3_uri
            )
            job_output["MonitoringOutputs"] = [normalized_monitoring_output._to_request_dict()]
        if output_kms_key is not None:
            job_output["KmsKeyId"] = output_kms_key

        # app specification
        if analysis_config is None:
            if latest_baselining_job_config is not None:
                analysis_config = latest_baselining_job_config.analysis_config
            elif app_specification:
                analysis_config = app_specification["ConfigUri"]
            else:
                raise ValueError("analysis_config is mandatory.")
            # backfill analysis_config
        if isinstance(analysis_config, str):
            analysis_config_uri = analysis_config
        else:
            analysis_config_uri = self._upload_analysis_config(
                analysis_config._to_dict(), output_s3_uri, job_definition_name, output_kms_key
            )
        app_specification["ConfigUri"] = analysis_config_uri
        app_specification["ImageUri"] = image_uri
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
        elif latest_baselining_job_name:
            baseline_config["BaseliningJobName"] = latest_baselining_job_name

        # job input
        if endpoint_input is not None:
            normalized_endpoint_input = self._normalize_endpoint_input(
                endpoint_input=endpoint_input
            )
            # backfill attributes to endpoint input
            if latest_baselining_job_config is not None:
                if normalized_endpoint_input.features_attribute is None:
                    normalized_endpoint_input.features_attribute = (
                        latest_baselining_job_config.features_attribute
                    )
                if normalized_endpoint_input.inference_attribute is None:
                    normalized_endpoint_input.inference_attribute = (
                        latest_baselining_job_config.inference_attribute
                    )
                if normalized_endpoint_input.probability_attribute is None:
                    normalized_endpoint_input.probability_attribute = (
                        latest_baselining_job_config.probability_attribute
                    )
                if normalized_endpoint_input.probability_threshold_attribute is None:
                    normalized_endpoint_input.probability_threshold_attribute = (
                        latest_baselining_job_config.probability_threshold_attribute
                    )
            job_input = normalized_endpoint_input._to_request_dict()
        elif batch_transform_input is not None:
            # backfill attributes to batch transform input
            if latest_baselining_job_config is not None:
                if batch_transform_input.features_attribute is None:
                    batch_transform_input.features_attribute = (
                        latest_baselining_job_config.features_attribute
                    )
                if batch_transform_input.inference_attribute is None:
                    batch_transform_input.inference_attribute = (
                        latest_baselining_job_config.inference_attribute
                    )
                if batch_transform_input.probability_attribute is None:
                    batch_transform_input.probability_attribute = (
                        latest_baselining_job_config.probability_attribute
                    )
                if batch_transform_input.probability_threshold_attribute is None:
                    batch_transform_input.probability_threshold_attribute = (
                        latest_baselining_job_config.probability_threshold_attribute
                    )
            job_input = batch_transform_input._to_request_dict()

        if ground_truth_input is not None:
            job_input["GroundTruthS3Input"] = dict(S3Uri=ground_truth_input)

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


class ModelBiasMonitor(ClarifyModelMonitor):
    """Amazon SageMaker model monitor to monitor bias metrics of an endpoint.

    Please see the __init__ method of its base class for how to instantiate it.
    """

    JOB_DEFINITION_BASE_NAME = "model-bias-job-definition"

    @classmethod
    def monitoring_type(cls):
        """Type of the monitoring job."""
        return "ModelBias"

    def suggest_baseline(
        self,
        data_config,
        bias_config,
        model_config,
        model_predicted_label_config=None,
        wait=False,
        logs=False,
        job_name=None,
        kms_key=None,
    ):
        """Suggests baselines for use with Amazon SageMaker Model Monitoring Schedules.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
            model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
                endpoint to be created.
            model_predicted_label_config (:class:`~sagemaker.clarify.ModelPredictedLabelConfig`):
                Config of how to extract the predicted label from the model output.
            wait (bool): Whether the call should wait until the job completes (default: False).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: False).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
                baselining job.
        """
        baselining_processor = self._create_baselining_processor()
        baselining_job_name = self._generate_baselining_job_name(job_name=job_name)
        baselining_processor.run_bias(
            data_config=data_config,
            bias_config=bias_config,
            model_config=model_config,
            model_predicted_label_config=model_predicted_label_config,
            wait=wait,
            logs=logs,
            job_name=baselining_job_name,
            kms_key=kms_key,
        )

        latest_baselining_job_config = ClarifyBaseliningConfig(
            analysis_config=BiasAnalysisConfig(
                bias_config=bias_config, headers=data_config.headers, label=data_config.label
            ),
            features_attribute=data_config.features,
        )
        if model_predicted_label_config is not None:
            latest_baselining_job_config.inference_attribute = (
                model_predicted_label_config.label
                if model_predicted_label_config.label is None
                else str(model_predicted_label_config.label)
            )
            latest_baselining_job_config.probability_attribute = (
                model_predicted_label_config.probability
                if model_predicted_label_config.probability is None
                else str(model_predicted_label_config.probability)
            )
            latest_baselining_job_config.probability_threshold_attribute = (
                model_predicted_label_config.probability_threshold
            )
        self.latest_baselining_job_config = latest_baselining_job_config
        self.latest_baselining_job_name = baselining_job_name
        self.latest_baselining_job = ClarifyBaseliningJob(
            processing_job=baselining_processor.latest_job
        )

        self.baselining_jobs.append(self.latest_baselining_job)
        return baselining_processor.latest_job

    # noinspection PyMethodOverriding
    def create_monitoring_schedule(
        self,
        endpoint_input=None,
        ground_truth_input=None,
        analysis_config=None,
        output_s3_uri=None,
        constraints=None,
        monitor_schedule_name=None,
        schedule_cron_expression=None,
        enable_cloudwatch_metrics=True,
        batch_transform_input=None,
    ):
        """Creates a monitoring schedule.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput. (default: None)
            ground_truth_input (str): S3 URI to ground truth dataset. (default: None)
            analysis_config (str or BiasAnalysisConfig): URI to analysis_config for the bias job.
                If it is None then configuration of the latest baselining job will be reused, but
                if no baselining job then fail the call. (default: None)
            output_s3_uri (str): S3 destination of the constraint_violations and analysis result.
                Default: "s3://<default_session_bucket>/<job_name>/output" (default: None)
            constraints (sagemaker.model_monitor.Constraints or str): If provided it will be used
                for monitoring the endpoint. It can be a Constraints object or an S3 uri pointing
                to a constraints JSON file. (default: None)
            monitor_schedule_name (str): Schedule name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
                (default: None)
            schedule_cron_expression (str): The cron expression that dictates the frequency that
                this job run. See sagemaker.model_monitor.CronExpressionGenerator for valid
                expressions. Default: Daily. (default: None)
            enable_cloudwatch_metrics (bool): Whether to publish cloudwatch metrics as part of
                the baselining or monitoring jobs. (default: True)
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to run
                the monitoring schedule on the batch transform (default: None)
        """
        # we default ground_truth_input to None in the function signature
        # but verify they are giving here for positional argument
        # backward compatibility reason.
        if not ground_truth_input:
            raise ValueError("ground_truth_input can not be None.")
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
        request_dict = self._build_create_job_definition_request(
            monitoring_schedule_name=monitor_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self.image_uri,
            latest_baselining_job_name=self.latest_baselining_job_name,
            latest_baselining_job_config=self.latest_baselining_job_config,
            endpoint_input=endpoint_input,
            ground_truth_input=ground_truth_input,
            analysis_config=analysis_config,
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
        self.sagemaker_session.sagemaker_client.create_model_bias_job_definition(**request_dict)

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
                self.sagemaker_session.sagemaker_client.delete_model_bias_job_definition(
                    JobDefinitionName=new_job_definition_name
                )
            except Exception:  # pylint: disable=W0703
                message = "Failed to delete job definition {}.".format(new_job_definition_name)
                _LOGGER.exception(message)
            raise

    # noinspection PyMethodOverriding
    def update_monitoring_schedule(
        self,
        endpoint_input=None,
        ground_truth_input=None,
        analysis_config=None,
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
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            ground_truth_input (str): S3 URI to ground truth dataset.
            analysis_config (str or BiasAnalysisConfig): URI to analysis_config for the bias job.
                If it is None then configuration of the latest baselining job will be reused, but
                if no baselining job then fail the call.
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
            batch_transform_input (sagemaker.model_monitor.BatchTransformInput): Inputs to run
                the monitoring schedule on the batch transform
        """
        valid_args = {
            arg: value for arg, value in locals().items() if arg != "self" and value is not None
        }

        # Nothing to update
        if len(valid_args) <= 0:
            return

        if batch_transform_input is not None and endpoint_input is not None:
            message = (
                "Need to have either batch_transform_input or endpoint_input to create an "
                "Amazon Model Monitoring Schedule. "
                "Please provide only one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        # Only need to update schedule expression
        if len(valid_args) == 1 and schedule_cron_expression is not None:
            self._update_monitoring_schedule(self.job_definition_name, schedule_cron_expression)
            return

        # Need to update schedule with a new job definition
        job_desc = self.sagemaker_session.sagemaker_client.describe_model_bias_job_definition(
            JobDefinitionName=self.job_definition_name
        )
        new_job_definition_name = name_from_base(self.JOB_DEFINITION_BASE_NAME)
        request_dict = self._build_create_job_definition_request(
            monitoring_schedule_name=self.monitoring_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self.image_uri,
            existing_job_desc=job_desc,
            endpoint_input=endpoint_input,
            ground_truth_input=ground_truth_input,
            analysis_config=analysis_config,
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
        self.sagemaker_session.sagemaker_client.create_model_bias_job_definition(**request_dict)
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
                self.sagemaker_session.sagemaker_client.delete_model_bias_job_definition(
                    JobDefinitionName=new_job_definition_name
                )
            except Exception:  # pylint: disable=W0703
                message = "Failed to delete job definition {}.".format(new_job_definition_name)
                _LOGGER.exception(message)
            raise

    def delete_monitoring_schedule(self):
        """Deletes the monitoring schedule and its job definition."""
        super(ModelBiasMonitor, self).delete_monitoring_schedule()
        # Delete job definition.
        message = "Deleting Model Bias Job Definition with name: {}".format(
            self.job_definition_name
        )
        _LOGGER.info(message)
        self.sagemaker_session.sagemaker_client.delete_model_bias_job_definition(
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
            raise TypeError("{} can only attach to ModelBias schedule.".format(__class__.__name__))
        job_definition_name = schedule_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinitionName"
        ]
        job_desc = sagemaker_session.sagemaker_client.describe_model_bias_job_definition(
            JobDefinitionName=job_definition_name
        )
        tags = sagemaker_session.list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])
        return ClarifyModelMonitor._attach(
            clazz=cls,
            sagemaker_session=sagemaker_session,
            schedule_desc=schedule_desc,
            job_desc=job_desc,
            tags=tags,
        )


class BiasAnalysisConfig:
    """Analysis configuration for ModelBiasMonitor."""

    def __init__(self, bias_config, headers=None, label=None):
        """Creates an analysis config dictionary.

        Args:
            bias_config (sagemaker.clarify.BiasConfig): Config object related to bias
                configurations.
            headers (list[str]): A list of column names in the input dataset.
            label (str): Target attribute for the model required by bias metrics. Specified as
                column name or index for CSV dataset, or as JMESPath expression for JSONLines.
        """
        self.analysis_config = bias_config.get_config()
        if headers is not None:
            self.analysis_config["headers"] = headers
        if label is not None:
            self.analysis_config["label"] = label

    def _to_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        return self.analysis_config


class ModelExplainabilityMonitor(ClarifyModelMonitor):
    """Amazon SageMaker model monitor to monitor feature attribution of an endpoint.

    Please see the __init__ method of its base class for how to instantiate it.
    """

    JOB_DEFINITION_BASE_NAME = "model-explainability-job-definition"

    @classmethod
    def monitoring_type(cls):
        """Type of the monitoring job."""
        return "ModelExplainability"

    def suggest_baseline(
        self,
        data_config,
        explainability_config,
        model_config,
        model_scores=None,
        wait=False,
        logs=False,
        job_name=None,
        kms_key=None,
    ):
        """Suggest baselines for use with Amazon SageMaker Model Monitoring Schedules.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            explainability_config (:class:`~sagemaker.clarify.ExplainabilityConfig`): Config of the
                specific explainability method. Currently, only SHAP is supported.
            model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
                endpoint to be created.
            model_scores (int or str or :class:`~sagemaker.clarify.ModelPredictedLabelConfig`):
                Index or JMESPath expression to locate the predicted scores in the model output.
                This is not required if the model output is a single score. Alternatively,
                it can be an instance of ModelPredictedLabelConfig to provide more parameters
                like label_headers.
            wait (bool): Whether the call should wait until the job completes (default: False).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: False).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).

        Returns:
            sagemaker.processing.ProcessingJob: The ProcessingJob object representing the
                baselining job.
        """
        baselining_processor = self._create_baselining_processor()
        baselining_job_name = self._generate_baselining_job_name(job_name=job_name)
        baselining_processor.run_explainability(
            data_config=data_config,
            model_config=model_config,
            explainability_config=explainability_config,
            model_scores=model_scores,
            wait=wait,
            logs=logs,
            job_name=baselining_job_name,
            kms_key=kms_key,
        )

        # Explainability analysis doesn't need label
        headers = copy.deepcopy(data_config.headers)
        if headers and data_config.label in headers:
            headers.remove(data_config.label)
        if model_scores is None:
            inference_attribute = None
            label_headers = None
        elif isinstance(model_scores, ModelPredictedLabelConfig):
            inference_attribute = str(model_scores.label)
            label_headers = model_scores.label_headers
        else:
            inference_attribute = str(model_scores)
            label_headers = None
        self.latest_baselining_job_config = ClarifyBaseliningConfig(
            analysis_config=ExplainabilityAnalysisConfig(
                explainability_config=explainability_config,
                model_config=model_config,
                headers=headers,
                label_headers=label_headers,
            ),
            features_attribute=data_config.features,
            inference_attribute=inference_attribute,
        )
        self.latest_baselining_job_name = baselining_job_name
        self.latest_baselining_job = ClarifyBaseliningJob(
            processing_job=baselining_processor.latest_job
        )

        self.baselining_jobs.append(self.latest_baselining_job)
        return baselining_processor.latest_job

    # noinspection PyMethodOverriding
    def create_monitoring_schedule(
        self,
        endpoint_input=None,
        analysis_config=None,
        output_s3_uri=None,
        constraints=None,
        monitor_schedule_name=None,
        schedule_cron_expression=None,
        enable_cloudwatch_metrics=True,
        batch_transform_input=None,
    ):
        """Creates a monitoring schedule.

        Args:
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput. (default: None)
            analysis_config (str or ExplainabilityAnalysisConfig): URI to the analysis_config for
                the explainability job. If it is None then configuration of the latest baselining
                job will be reused, but if no baselining job then fail the call.
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
                "Amazon Model Monitoring Schedule."
                "Please provide only one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        # create job definition
        monitor_schedule_name = self._generate_monitoring_schedule_name(
            schedule_name=monitor_schedule_name
        )
        new_job_definition_name = name_from_base(self.JOB_DEFINITION_BASE_NAME)
        request_dict = self._build_create_job_definition_request(
            monitoring_schedule_name=monitor_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self.image_uri,
            latest_baselining_job_name=self.latest_baselining_job_name,
            latest_baselining_job_config=self.latest_baselining_job_config,
            endpoint_input=endpoint_input,
            analysis_config=analysis_config,
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
        self.sagemaker_session.sagemaker_client.create_model_explainability_job_definition(
            **request_dict
        )

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
                self.sagemaker_session.sagemaker_client.delete_model_explainability_job_definition(
                    JobDefinitionName=new_job_definition_name
                )
            except Exception:  # pylint: disable=W0703
                message = "Failed to delete job definition {}.".format(new_job_definition_name)
                _LOGGER.exception(message)
            raise

    # noinspection PyMethodOverriding
    def update_monitoring_schedule(
        self,
        endpoint_input=None,
        analysis_config=None,
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
            endpoint_input (str or sagemaker.model_monitor.EndpointInput): The endpoint to monitor.
                This can either be the endpoint name or an EndpointInput.
            analysis_config (str or BiasAnalysisConfig): URI to analysis_config for the bias job.
                If it is None then configuration of the latest baselining job will be reused, but
                if no baselining job then fail the call.
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
            raise ValueError("Nothing to update.")

        if batch_transform_input is not None and endpoint_input is not None:
            message = (
                "Need to have either batch_transform_input or endpoint_input to create an "
                "Amazon Model Monitoring Schedule. "
                "Please provide only one of the above required inputs"
            )
            _LOGGER.error(message)
            raise ValueError(message)

        # Only need to update schedule expression
        if len(valid_args) == 1 and schedule_cron_expression is not None:
            self._update_monitoring_schedule(self.job_definition_name, schedule_cron_expression)
            return

        # Need to update schedule with a new job definition
        job_desc = (
            self.sagemaker_session.sagemaker_client.describe_model_explainability_job_definition(
                JobDefinitionName=self.job_definition_name
            )
        )
        new_job_definition_name = name_from_base(self.JOB_DEFINITION_BASE_NAME)
        request_dict = self._build_create_job_definition_request(
            monitoring_schedule_name=self.monitoring_schedule_name,
            job_definition_name=new_job_definition_name,
            image_uri=self.image_uri,
            existing_job_desc=job_desc,
            endpoint_input=endpoint_input,
            analysis_config=analysis_config,
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
        self.sagemaker_session.sagemaker_client.create_model_explainability_job_definition(
            **request_dict
        )
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
                self.sagemaker_session.sagemaker_client.delete_model_explainability_job_definition(
                    JobDefinitionName=new_job_definition_name
                )
            except Exception:  # pylint: disable=W0703
                message = "Failed to delete job definition {}.".format(new_job_definition_name)
                _LOGGER.exception(message)
            raise

    def delete_monitoring_schedule(self):
        """Deletes the monitoring schedule and its job definition."""
        super(ModelExplainabilityMonitor, self).delete_monitoring_schedule()
        # Delete job definition.
        message = "Deleting Model Explainability Job Definition with name: {}".format(
            self.job_definition_name
        )
        _LOGGER.info(message)
        self.sagemaker_session.sagemaker_client.delete_model_explainability_job_definition(
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
                "{} can only attach to ModelExplainability schedule.".format(__class__.__name__)
            )
        job_definition_name = schedule_desc["MonitoringScheduleConfig"][
            "MonitoringJobDefinitionName"
        ]
        job_desc = sagemaker_session.sagemaker_client.describe_model_explainability_job_definition(
            JobDefinitionName=job_definition_name
        )
        tags = sagemaker_session.list_tags(resource_arn=schedule_desc["MonitoringScheduleArn"])
        return ClarifyModelMonitor._attach(
            clazz=cls,
            sagemaker_session=sagemaker_session,
            schedule_desc=schedule_desc,
            job_desc=job_desc,
            tags=tags,
        )


class ExplainabilityAnalysisConfig:
    """Analysis configuration for ModelExplainabilityMonitor."""

    def __init__(self, explainability_config, model_config, headers=None, label_headers=None):
        """Creates an analysis config dictionary.

        Args:
            explainability_config (sagemaker.clarify.ExplainabilityConfig): Config object related
                to explainability configurations.
            model_config (sagemaker.clarify.ModelConfig): Config object related to bias
                configurations.
            headers (list[str]): A list of feature names (without label) of model/endpint input.
            label_headers (list[str]): List of headers, each for a predicted score in model output.
                It is used to beautify the analysis report by replacing placeholders like "label0".

        """
        predictor_config = model_config.get_predictor_config()
        self.analysis_config = {
            "methods": explainability_config.get_explainability_config(),
            "predictor": predictor_config,
        }
        if headers is not None:
            self.analysis_config["headers"] = headers
        if label_headers is not None:
            predictor_config["label_headers"] = label_headers

    def _to_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        return self.analysis_config


class ClarifyBaseliningConfig:
    """Data class to hold some essential analysis configuration of ClarifyBaseliningJob"""

    def __init__(
        self,
        analysis_config,
        features_attribute=None,
        inference_attribute=None,
        probability_attribute=None,
        probability_threshold_attribute=None,
    ):
        """Initialization.

        Args:
            analysis_config (BiasAnalysisConfig or ExplainabilityAnalysisConfig): analysis config
                from configurations of the baselining job.
            features_attribute (str): JMESPath expression to locate features in predictor request
                payload. Only required when predictor content type is JSONlines.
            inference_attribute (str): Index, header or JMESPath expression to locate predicted
                label in predictor response payload.
            probability_attribute (str): Index or JMESPath expression to locate probabilities or
                scores in the model output for computing feature attribution.
            probability_threshold_attribute (float): Value to indicate the threshold to select
                the binary label in the case of binary classification. Default is 0.5.
        """
        self.analysis_config = analysis_config
        self.features_attribute = features_attribute
        self.inference_attribute = inference_attribute
        self.probability_attribute = probability_attribute
        self.probability_threshold_attribute = probability_threshold_attribute


class ClarifyBaseliningJob(mm.BaseliningJob):
    """Provides functionality to retrieve baseline-specific output from Clarify baselining job."""

    def __init__(
        self,
        processing_job,
    ):
        """Initializes a ClarifyBaseliningJob that tracks a baselining job by suggest_baseline()

        Args:
            processing_job (sagemaker.processing.ProcessingJob): The ProcessingJob used for
                baselining instance.
        """
        super(ClarifyBaseliningJob, self).__init__(
            sagemaker_session=processing_job.sagemaker_session,
            job_name=processing_job.job_name,
            inputs=processing_job.inputs,
            outputs=processing_job.outputs,
            output_kms_key=processing_job.output_kms_key,
        )

    def baseline_statistics(self, **_):
        """Not implemented.

        The class doesn't support statistics.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("{} doesn't support statistics.".format(__class__.__name__))

    def suggested_constraints(self, file_name=None, kms_key=None):
        """Returns a sagemaker.model_monitor.

        Constraints object representing the constraints JSON file generated by this baselining job.

        Args:
            file_name (str): Keep this parameter to align with method signature in super class,
                but it will be ignored.
            kms_key (str): The kms key to use when retrieving the file.

        Returns:
            sagemaker.model_monitor.Constraints: The Constraints object representing the file that
                was generated by the job.

        Raises:
            UnexpectedStatusException: This is thrown if the job is not in a 'Complete' state.
        """
        return super(ClarifyBaseliningJob, self).suggested_constraints("analysis.json", kms_key)


class ClarifyMonitoringExecution(mm.MonitoringExecution):
    """Provides functionality to retrieve monitoring-specific files output from executions."""

    def __init__(self, sagemaker_session, job_name, inputs, output, output_kms_key=None):
        """Initializes an object that tracks a monitoring execution by a Clarify model monitor

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
        super(ClarifyMonitoringExecution, self).__init__(
            sagemaker_session=sagemaker_session,
            job_name=job_name,
            inputs=inputs,
            output=output,
            output_kms_key=output_kms_key,
        )

    def statistics(self, **_):
        """Not implemented.

        The class doesn't support statistics.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("{} doesn't support statistics.".format(__class__.__name__))
