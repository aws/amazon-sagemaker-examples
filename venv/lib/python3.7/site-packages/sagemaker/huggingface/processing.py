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
"""This module contains code related to HuggingFace Processors which are used for Processing jobs.

These jobs let customers perform data pre-processing, post-processing, feature engineering,
data validation, and model evaluation and interpretation on SageMaker.
"""
from __future__ import absolute_import

from typing import Union, Optional, List, Dict

from sagemaker.session import Session
from sagemaker.network import NetworkConfig
from sagemaker.processing import FrameworkProcessor
from sagemaker.huggingface.estimator import HuggingFace

from sagemaker.workflow.entities import PipelineVariable


class HuggingFaceProcessor(FrameworkProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using HuggingFace containers."""

    estimator_cls = HuggingFace

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        transformers_version: Optional[str] = None,
        tensorflow_version: Optional[str] = None,
        pytorch_version: Optional[str] = None,
        py_version: str = "py36",
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
        """This processor executes a Python script in a HuggingFace execution environment.

        Unless ``image_uri`` is specified, the environment is an Amazon-built Docker container
        that executes functions defined in the supplied ``code`` Python script.

        The arguments have the same meaning as in ``FrameworkProcessor``, with the following
        exceptions.

        Args:
            transformers_version (str): Transformers version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided. The current supported version is ``4.4.2``.
            tensorflow_version (str): TensorFlow version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``pytorch_version`` is provided. The current supported version is ``2.4.1``.
            pytorch_version (str): PyTorch version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``tensorflow_version`` is provided. The current supported version is ``1.6.0``.
            py_version (str): Python version you want to use for executing your model training
                code. Defaults to ``None``. Required unless ``image_uri`` is provided.  If
                using PyTorch, the current supported version is ``py36``. If using TensorFlow,
                the current supported version is ``py37``.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.processing.FrameworkProcessor`.
        """
        self.pytorch_version = pytorch_version
        self.tensorflow_version = tensorflow_version
        super().__init__(
            self.estimator_cls,
            transformers_version,
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

    def _create_estimator(
        self,
        entry_point="",
        source_dir=None,
        dependencies=None,
        git_config=None,
    ):
        """Override default estimator factory function for HuggingFace's different parameters

        HuggingFace estimators have 3 framework version parameters instead of one: The version for
        Transformers, PyTorch, and TensorFlow.
        """
        return self.estimator_cls(
            transformers_version=self.framework_version,
            tensorflow_version=self.tensorflow_version,
            pytorch_version=self.pytorch_version,
            py_version=self.py_version,
            entry_point=entry_point,
            source_dir=source_dir,
            dependencies=dependencies,
            git_config=git_config,
            code_location=self.code_location,
            enable_network_isolation=False,
            image_uri=self.image_uri,
            role=self.role,
            instance_count=self.instance_count,
            instance_type=self.instance_type,
            sagemaker_session=self.sagemaker_session,
            debugger_hook_config=False,
            disable_profiler=True,
        )
