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
"""This module contains code related to SKLearn Processors which are used for Processing jobs.

These jobs let customers perform data pre-processing, post-processing, feature engineering,
data validation, and model evaluation and interpretation on SageMaker.
"""
from __future__ import absolute_import

from typing import Union, List, Dict, Optional

from sagemaker.network import NetworkConfig
from sagemaker import image_uris, Session
from sagemaker.processing import ScriptProcessor
from sagemaker.sklearn import defaults
from sagemaker.workflow.entities import PipelineVariable


class SKLearnProcessor(ScriptProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using scikit-learn."""

    def __init__(
        self,
        framework_version: str,  # New arg
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        command: Optional[List[str]] = None,
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
        """Initialize an ``SKLearnProcessor`` instance.

        The SKLearnProcessor handles Amazon SageMaker processing tasks for jobs using scikit-learn.

        Args:
            framework_version (str): The version of scikit-learn.
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker training jobs
                and APIs that create Amazon SageMaker endpoints use this role
                to access training data and model artifacts. After the endpoint
                is created, the inference code might use the IAM role, if it
                needs to access an AWS resource.
            instance_type (str or PipelineVariable): Type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            instance_count (int or PipelineVariable): The number of instances to run
                the Processing job with. Defaults to 1.
            command ([str]): The command to run, along with any command-line flags.
                Example: ["python3", "-v"]. If not provided, ["python3"] or ["python2"]
                will be chosen based on the py_version parameter.
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume to
                use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing
                volume.
            output_kms_key (str or PipelineVariable): The KMS key id for all ProcessingOutputs.
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
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables
                to be passed to the processing job.
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): List of tags
                to be passed to the processing job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        if not command:
            command = ["python3"]

        session = sagemaker_session or Session()
        region = session.boto_region_name

        image_uri = image_uris.retrieve(
            defaults.SKLEARN_NAME, region, version=framework_version, instance_type=instance_type
        )

        super(SKLearnProcessor, self).__init__(
            role=role,
            image_uri=image_uri,
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
