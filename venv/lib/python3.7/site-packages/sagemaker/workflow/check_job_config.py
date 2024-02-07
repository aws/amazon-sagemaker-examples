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
"""Common config for QualityCheckStep and ClarifyCheckStep."""
from __future__ import absolute_import

import logging
from typing import Optional

from sagemaker import Session
from sagemaker.model_monitor import (
    ModelMonitor,
    DefaultModelMonitor,
    ModelQualityMonitor,
    ModelBiasMonitor,
    ModelExplainabilityMonitor,
)


class CheckJobConfig:
    """Check job config for QualityCheckStep and ClarifyCheckStep."""

    def __init__(
        self,
        role,
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
        """Constructs a CheckJobConfig instance.

        Args:
            role (str): An AWS IAM role. The Amazon SageMaker jobs use this role.
            instance_count (int): The number of instances to run the jobs with (default: 1).
            instance_type (str): Type of EC2 instance to use for the job
                (default: 'ml.m5.xlarge').
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing volume (default: None).
            output_kms_key (str): The KMS key id for the job's outputs (default: None).
            max_runtime_in_seconds (int): Timeout in seconds. After this amount of
                time, Amazon SageMaker terminates the job regardless of its current status.
                Default: 3600 if not specified
            base_job_name (str): Prefix for the job name. If not specified,
                a default name is generated based on the training image name and
                current timestamp (default: None).
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed (default: None). If not specified, one is
                created using the default AWS configuration chain.
            env (dict): Environment variables to be passed to the job (default: None).
            tags ([dict]): List of tags to be passed to the job (default: None).
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets (default: None).

        """
        self.role = role
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.volume_size_in_gb = volume_size_in_gb
        self.volume_kms_key = volume_kms_key
        self.output_kms_key = output_kms_key
        self.max_runtime_in_seconds = max_runtime_in_seconds
        self.base_job_name = base_job_name
        self.sagemaker_session = sagemaker_session or Session()
        self.env = env
        self.tags = tags
        self.network_config = network_config

    def _generate_model_monitor(self, mm_type: str) -> Optional[ModelMonitor]:
        """Generates a ModelMonitor object

        Generates a ModelMonitor object with required config attributes for
            QualityCheckStep and ClarifyCheckStep

        Args:
            mm_type (str): The subclass type of ModelMonitor object.
                A valid mm_type should be one of the following: "DefaultModelMonitor",
                "ModelQualityMonitor", "ModelBiasMonitor", "ModelExplainabilityMonitor"

        Return:
            sagemaker.model_monitor.ModelMonitor or None if the mm_type is not valid

        """
        if mm_type == "DefaultModelMonitor":
            monitor = DefaultModelMonitor(
                role=self.role,
                instance_count=self.instance_count,
                instance_type=self.instance_type,
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
        elif mm_type == "ModelQualityMonitor":
            monitor = ModelQualityMonitor(
                role=self.role,
                instance_count=self.instance_count,
                instance_type=self.instance_type,
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
        elif mm_type == "ModelBiasMonitor":
            monitor = ModelBiasMonitor(
                role=self.role,
                instance_count=self.instance_count,
                instance_type=self.instance_type,
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
        elif mm_type == "ModelExplainabilityMonitor":
            monitor = ModelExplainabilityMonitor(
                role=self.role,
                instance_count=self.instance_count,
                instance_type=self.instance_type,
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
        else:
            logging.warning(
                'Expected model monitor types: "DefaultModelMonitor", "ModelQualityMonitor", '
                '"ModelBiasMonitor", "ModelExplainabilityMonitor"'
            )
            return None
        return monitor
