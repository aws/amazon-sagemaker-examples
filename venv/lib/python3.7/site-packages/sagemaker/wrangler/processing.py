#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License"). You
#  may not use this file except in compliance with the License. A copy of
#  the License is located at
#  #
#      http://aws.amazon.com/apache2.0/
#  #
#  or in the "license" file accompanying this file. This file is
#  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
#  ANY KIND, either express or implied. See the License for the specific
#  language governing permissions and limitations under the License.
"""The process definitions for data wrangler."""

from __future__ import absolute_import

from typing import Dict, List

from sagemaker.network import NetworkConfig
from sagemaker.processing import (
    ProcessingInput,
    Processor,
)
from sagemaker import image_uris
from sagemaker.session import Session


class DataWranglerProcessor(Processor):
    """Handles Amazon SageMaker DataWrangler tasks"""

    def __init__(
        self,
        role: str = None,
        data_wrangler_flow_source: str = None,
        instance_count: int = None,
        instance_type: str = None,
        volume_size_in_gb: int = 30,
        volume_kms_key: str = None,
        output_kms_key: str = None,
        max_runtime_in_seconds: int = None,
        base_job_name: str = None,
        sagemaker_session: Session = None,
        env: Dict[str, str] = None,
        tags: List[dict] = None,
        network_config: NetworkConfig = None,
    ):
        """Initializes a ``Processor`` instance.

        The ``Processor`` handles Amazon SageMaker Processing tasks.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            data_wrangler_flow_source (str): The source of the DaraWrangler flow which will be
                used for the DataWrangler job. If a local path is provided, it will automatically
                be uploaded to S3 under:
                "s3://<default-bucket-name>/<job-name>/input/<input-name>".
            instance_count (int): The number of instances to run
                a processing job with.
            instance_type (str): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume (default: None).
            output_kms_key (str): The KMS key ID for processing job outputs (default: None).
            max_runtime_in_seconds (int): Timeout in seconds (default: None).
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
            env (dict[str, str]): Environment variables to be passed to
                the processing jobs (default: None).
            tags (list[dict]): List of tags to be passed to the processing job
                (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        self.data_wrangler_flow_source = data_wrangler_flow_source
        self.sagemaker_session = sagemaker_session or Session()
        image_uri = image_uris.retrieve(
            "data-wrangler", region=self.sagemaker_session.boto_region_name
        )
        super().__init__(
            role,
            image_uri,
            instance_count,
            instance_type,
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
        inputs = inputs or []
        found = any(element.input_name == "flow" for element in inputs)
        if not found:
            inputs.append(self._get_recipe_input())
        return super()._normalize_args(job_name, arguments, inputs, outputs, code, kms_key)

    def _get_recipe_input(self):
        """Creates a ProcessingInput with Data Wrangler recipe uri and appends it to inputs"""
        return ProcessingInput(
            source=self.data_wrangler_flow_source,
            destination="/opt/ml/processing/flow",
            input_name="flow",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
            s3_data_distribution_type="FullyReplicated",
        )
