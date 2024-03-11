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
"""This module contains code related to MXNet Processors which are used for Processing jobs.

These jobs let customers perform data pre-processing, post-processing, feature engineering,
data validation, and model evaluation and interpretation on SageMaker.
"""
from __future__ import absolute_import

from typing import Union, List, Dict, Optional

from sagemaker.session import Session
from sagemaker.network import NetworkConfig
from sagemaker.processing import FrameworkProcessor
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.entities import PipelineVariable


class XGBoostProcessor(FrameworkProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using XGBoost containers."""

    estimator_cls = XGBoost

    def __init__(
        self,
        framework_version: str,  # New arg
        role: str = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        py_version: str = "py3",  # New kwarg
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        command: Optional[List[str]] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        code_location: Optional[str] = None,  # New arg
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        network_config: Optional[NetworkConfig] = None,
    ):
        """This processor executes a Python script in an XGBoost execution environment.

        Unless ``image_uri`` is specified, the XGBoost environment is an Amazon-built
        Docker container that executes functions defined in the supplied ``code``
        Python script.

        The arguments have the exact same meaning as in ``FrameworkProcessor``.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.processing.FrameworkProcessor`.
        """
        super().__init__(
            self.estimator_cls,
            framework_version,
            role,
            instance_count,
            instance_type,
            py_version,
            image_uri,
            command,
            volume_size_in_gb,
            volume_kms_key,
            output_kms_key,
            code_location,
            max_runtime_in_seconds,
            base_job_name,
            sagemaker_session,
            env,
            tags,
            network_config,
        )
