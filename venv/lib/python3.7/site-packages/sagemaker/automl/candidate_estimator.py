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
"""A class for AutoML Job's Candidate."""
from __future__ import absolute_import

from six import string_types
from sagemaker.config import (
    TRAINING_JOB_VPC_CONFIG_PATH,
    TRAINING_JOB_VOLUME_KMS_KEY_ID_PATH,
    TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
)
from sagemaker.session import Session
from sagemaker.job import _Job
from sagemaker.utils import name_from_base, resolve_value_from_config


class CandidateEstimator(object):
    """A class for SageMaker AutoML Job Candidate"""

    def __init__(self, candidate, sagemaker_session=None):
        """Constructor of CandidateEstimator.

        Args:
            candidate (dict): a dictionary of candidate returned by AutoML.list_candidates()
                or AutoML.best_candidate().
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
        """
        self.name = candidate["CandidateName"]
        self.containers = candidate["InferenceContainers"]
        self.steps = self._process_steps(candidate["CandidateSteps"])
        self.sagemaker_session = sagemaker_session or Session()

    def get_steps(self):
        """Get the step job of a candidate so that users can construct estimators/transformers

        Returns:
            list: a list of dictionaries that provide information about each step job's name,
                type, inputs and description
        """
        candidate_steps = []
        for step in self.steps:
            step_type = step["type"]
            step_name = step["name"]
            if step_type == "TrainingJob":
                training_job = self.sagemaker_session.sagemaker_client.describe_training_job(
                    TrainingJobName=step_name
                )

                inputs = training_job["InputDataConfig"]
                candidate_step = CandidateStep(step_name, inputs, step_type, training_job)
                candidate_steps.append(candidate_step)
            elif step_type == "TransformJob":
                transform_job = self.sagemaker_session.sagemaker_client.describe_transform_job(
                    TransformJobName=step_name
                )
                inputs = transform_job["TransformInput"]
                candidate_step = CandidateStep(step_name, inputs, step_type, transform_job)
                candidate_steps.append(candidate_step)
        return candidate_steps

    def fit(
        self,
        inputs,
        candidate_name=None,
        volume_kms_key=None,
        # default of False for training job, checked inside function
        encrypt_inter_container_traffic=None,
        vpc_config=None,
        wait=True,
        logs=True,
    ):
        """Rerun a candidate's step jobs with new input datasets or security config.

        Args:
            inputs (str or list[str]): Local path or S3 Uri where the training data is stored. If a
                local path is provided, the dataset will be uploaded to an S3 location.
            candidate_name (str): name of the candidate to be rerun, if None, candidate's original
                name will be used.
            volume_kms_key (str): The KMS key id to encrypt data on the storage volume attached to
                the ML compute instance(s).
            encrypt_inter_container_traffic (bool): To encrypt all communications between ML compute
                instances in distributed training. If not passed, will be fetched from
                sagemaker_config if a value is defined there. Default: False.
            vpc_config (dict): Specifies a VPC that jobs and hosted models have access to.
                Control access to and from training and model containers by configuring the VPC
            wait (bool): Whether the call should wait until all jobs completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
        """
        if logs and not wait:
            raise ValueError(
                """Logs can only be shown if wait is set to True.
                Please either set wait to True or set logs to False."""
            )
        vpc_config = resolve_value_from_config(
            vpc_config, TRAINING_JOB_VPC_CONFIG_PATH, sagemaker_session=self.sagemaker_session
        )
        volume_kms_key = resolve_value_from_config(
            volume_kms_key,
            TRAINING_JOB_VOLUME_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.name = candidate_name or self.name
        running_jobs = {}

        # convert inputs to TrainingInput format
        if isinstance(inputs, string_types):
            if not inputs.startswith("s3://"):
                inputs = self.sagemaker_session.upload_data(inputs, key_prefix="auto-ml-input-data")

        for step in self.steps:
            step_type = step["type"]
            step_name = step["name"]
            if step_type == "TrainingJob":
                # prepare inputs
                input_dict = {}
                if isinstance(inputs, string_types):
                    input_dict["train"] = _Job._format_string_uri_input(inputs)
                else:
                    msg = "Cannot format input {}. Expecting a string."
                    raise ValueError(msg.format(inputs))

                channels = [
                    _Job._convert_input_to_channel(name, input)
                    for name, input in input_dict.items()
                ]

                desc = self.sagemaker_session.sagemaker_client.describe_training_job(
                    TrainingJobName=step_name
                )
                base_name = "sagemaker-automl-training-rerun"
                step_name = name_from_base(base_name)
                step["name"] = step_name

                # Check training_job config not auto_ml_job config because this function calls
                # training job API
                _encrypt_inter_container_traffic = resolve_value_from_config(
                    direct_input=encrypt_inter_container_traffic,
                    config_path=TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
                    default_value=False,
                    sagemaker_session=self.sagemaker_session,
                )

                train_args = self._get_train_args(
                    desc,
                    channels,
                    step_name,
                    volume_kms_key,
                    _encrypt_inter_container_traffic,
                    vpc_config,
                )
                self.sagemaker_session.train(**train_args)
                running_jobs[step_name] = True

            elif step_type == "TransformJob":
                # prepare inputs
                if not isinstance(inputs, string_types) or not inputs.startswith("s3://"):
                    msg = "Cannot format input {}. Expecting a string starts with file:// or s3://"
                    raise ValueError(msg.format(inputs))

                desc = self.sagemaker_session.sagemaker_client.describe_transform_job(
                    TransformJobName=step_name
                )
                base_name = "sagemaker-automl-transform-rerun"
                step_name = name_from_base(base_name)
                step["name"] = step_name
                transform_args = self._get_transform_args(desc, inputs, step_name, volume_kms_key)
                self.sagemaker_session.transform(**transform_args)
                running_jobs[step_name] = True

        if wait:
            while True:
                for step in self.steps:
                    status = None
                    step_type = step["type"]
                    step_name = step["name"]
                    if step_type == "TrainingJob":
                        status = self.sagemaker_session.sagemaker_client.describe_training_job(
                            TrainingJobName=step_name
                        )["TrainingJobStatus"]
                    elif step_type == "TransformJob":
                        status = self.sagemaker_session.sagemaker_client.describe_transform_job(
                            TransformJobName=step_name
                        )["TransformJobStatus"]
                    if status in ("Completed", "Failed", "Stopped"):
                        running_jobs[step_name] = False
                if self._check_all_job_finished(running_jobs):
                    break

    def _check_all_job_finished(self, running_jobs):
        """Check if all step jobs are finished.

        Args:
            running_jobs (dict): a dictionary that keeps track of the status
                of each step job.

        Returns (bool): True if all step jobs are finished. False if one or
            more step jobs are still running.
        """
        for _, v in running_jobs.items():
            if v:
                return False
        return True

    def _get_train_args(
        self, desc, inputs, name, volume_kms_key, encrypt_inter_container_traffic, vpc_config
    ):
        """Format training args to pass in sagemaker_session.train.

        Args:
            desc (dict): the response from DescribeTrainingJob API.
            inputs (list): a list of input data channels.
            name (str): the name of the step job.
            volume_kms_key (str): The KMS key id to encrypt data on the storage volume attached to
                the ML compute instance(s).
            encrypt_inter_container_traffic (bool): To encrypt all communications between ML compute
                instances in distributed training.
            vpc_config (dict): Specifies a VPC that jobs and hosted models have access to.
                Control access to and from training and model containers by configuring the VPC

        Returns (dcit): a dictionary that can be used as args of
            sagemaker_session.train method.
        """
        train_args = {
            "input_config": inputs,
            "job_name": name,
            "input_mode": desc["AlgorithmSpecification"]["TrainingInputMode"],
            "role": desc["RoleArn"],
            "output_config": desc["OutputDataConfig"],
            "resource_config": desc["ResourceConfig"],
            "image_uri": desc["AlgorithmSpecification"]["TrainingImage"],
            "enable_network_isolation": desc["EnableNetworkIsolation"],
            "encrypt_inter_container_traffic": encrypt_inter_container_traffic,
            "use_spot_instances": desc["EnableManagedSpotTraining"],
            "hyperparameters": {},
            "stop_condition": {},
            "metric_definitions": None,
            "checkpoint_s3_uri": None,
            "checkpoint_local_path": None,
            "tags": [],
            "vpc_config": None,
        }

        if volume_kms_key is not None:
            train_args["resource_config"]["VolumeKmsKeyId"] = volume_kms_key
        if "VpcConfig" in desc:
            train_args["vpc_config"] = desc["VpcConfig"]
        elif vpc_config is not None:
            train_args["vpc_config"] = vpc_config
        if "Hyperparameters" in desc:
            train_args["hyperparameters"] = desc["Hyperparameters"]
        if "CheckpointConfig" in desc:
            train_args["checkpoint_s3_uri"] = desc["CheckpointConfig"]["S3Uri"]
            train_args["checkpoint_local_path"] = desc["CheckpointConfig"]["LocalPath"]
        if "StoppingCondition" in desc:
            train_args["stop_condition"] = desc["StoppingCondition"]
        return train_args

    def _get_transform_args(self, desc, inputs, name, volume_kms_key):
        """Format training args to pass in sagemaker_session.train.

        Args:
            desc (dict): the response from DescribeTrainingJob API.
            inputs (str): an S3 uri where new input dataset is stored.
            name (str): the name of the step job.
            volume_kms_key (str): The KMS key id to encrypt data on the storage volume attached to
                the ML compute instance(s).

        Returns (dcit): a dictionary that can be used as args of
            sagemaker_session.transform method.
        """
        transform_args = {}
        transform_args["job_name"] = name
        transform_args["model_name"] = desc["ModelName"]
        transform_args["output_config"] = desc["TransformOutput"]
        transform_args["resource_config"] = desc["TransformResources"]
        transform_args["data_processing"] = desc["DataProcessing"]
        transform_args["tags"] = []
        transform_args["strategy"] = None
        transform_args["max_concurrent_transforms"] = None
        transform_args["max_payload"] = None
        transform_args["env"] = None
        transform_args["experiment_config"] = None

        input_config = desc["TransformInput"]
        input_config["DataSource"]["S3DataSource"]["S3Uri"] = inputs
        transform_args["input_config"] = input_config

        if volume_kms_key is not None:
            transform_args["resource_config"]["VolumeKmsKeyId"] = volume_kms_key
        if "BatchStrategy" in desc:
            transform_args["strategy"] = desc["BatchStrategy"]
        if "MaxConcurrentTransforms" in desc:
            transform_args["max_concurrent_transforms"] = desc["MaxConcurrentTransforms"]
        if "MaxPayloadInMB" in desc:
            transform_args["max_payload"] = desc["MaxPayloadInMB"]
        if "Environment" in desc:
            transform_args["env"] = desc["Environment"]

        return transform_args

    def _process_steps(self, steps):
        """Extract candidate's step jobs name and type.

        Args:
            steps (list): a list of a candidate's step jobs.

        Returns (list): a list of extracted information about step jobs'
            name and type.
        """
        processed_steps = []
        for step in steps:
            step_name = step["CandidateStepName"]
            step_type = step["CandidateStepType"].split("::")[2]
            processed_steps.append({"name": step_name, "type": step_type})
        return processed_steps


class CandidateStep(object):
    """A class that maintains an AutoML Candidate step's name, inputs, type, and description."""

    def __init__(self, name, inputs, step_type, description):
        self._name = name
        self._inputs = inputs
        self._type = step_type
        self._description = description

    @property
    def name(self):
        """Name of the candidate step -> (str)"""
        return self._name

    @property
    def inputs(self):
        """Inputs of the candidate step -> (dict)"""
        return self._inputs

    @property
    def type(self):
        """Type of the candidate step, Training or Transform -> (str)"""
        return self._type

    @property
    def description(self):
        """Description of candidate step job -> (dict)"""
        return self._description
