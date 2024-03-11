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
"""Configuration for collecting system and framework metrics in SageMaker training jobs."""
from __future__ import absolute_import

import logging
from typing import Optional, Union

from sagemaker.debugger.framework_profile import FrameworkProfile
from sagemaker.debugger.profiler import Profiler
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.deprecations import deprecation_warn_base

logger = logging.getLogger(__name__)


class ProfilerConfig(object):
    """Configuration for collecting system and framework metrics of SageMaker training jobs.

    SageMaker Debugger collects system and framework profiling
    information of training jobs and identify performance bottlenecks.

    """

    def __init__(
        self,
        s3_output_path: Optional[Union[str, PipelineVariable]] = None,
        system_monitor_interval_millis: Optional[Union[int, PipelineVariable]] = None,
        framework_profile_params: Optional[FrameworkProfile] = None,
        profile_params: Optional[Profiler] = None,
        disable_profiler: Optional[Union[str, PipelineVariable]] = False,
    ):
        """Initialize a ``ProfilerConfig`` instance.

        Pass the output of this class
        to the ``profiler_config`` parameter of the generic :class:`~sagemaker.estimator.Estimator`
        class and SageMaker Framework estimators.

        Args:
            s3_output_path (str or PipelineVariable): The location in Amazon S3 to store
                the output.
                The default Debugger output path for profiling data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/profiler-output/.
            system_monitor_interval_millis (int or PipelineVariable): The time interval in
                milliseconds to collect system metrics. Available values are 100, 200, 500,
                1000 (1 second), 5000 (5 seconds), and 60000 (1 minute) milliseconds.
                The default is 500 milliseconds.
            framework_profile_params (:class:`~sagemaker.debugger.FrameworkProfile`):
                (Deprecated) A parameter object for framework metrics profiling. Configure it using
                the :class:`~sagemaker.debugger.FrameworkProfile` class.
                To use the default framework profile parameters, pass ``FrameworkProfile()``.
                For more information about the default values,
                see :class:`~sagemaker.debugger.FrameworkProfile`.
            disable_profiler (bool): Switch the basic monitoring on or off using this parameter.
                The default is ``False``.
            profile_params (dict or an object of :class:`sagemaker.Profiler`): Pass this parameter
                to activate SageMaker Profiler using the :class:`sagemaker.Profiler` class.

        **Basic profiling using SageMaker Debugger**

        By default, if you submit training jobs using SageMaker Python SDK's estimator classes,
        SageMaker runs basic profiling automatically.
        The following example shows the basic profiling configuration
        that you can utilize to update the time interval for collecting system resource utilization.

        .. code:: python

            import sagemaker
            from sagemaker.pytorch import PyTorch
            from sagemaker.debugger import ProfilerConfig

            profiler_config = ProfilerConfig(
                system_monitor_interval_millis = 500
            )

            estimator = PyTorch(
                framework_version="2.0.0",
                ... # Set up other essential parameters for the estimator class
                profiler_config=profiler_config
            )

        For a complete instruction on activating and using SageMaker Debugger, see
        `Monitor AWS compute resource utilization in Amazon SageMaker Studio
        <https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html>`_.

        **Deep profiling using SageMaker Profiler**

        The following example shows an example configration for activating
        SageMaker Profiler.

        .. code:: python

            import sagemaker
            from sagemaker.pytorch import PyTorch
            from sagemaker import ProfilerConfig, Profiler

            profiler_config = ProfilerConfig(
                profiler_params = Profiler(cpu_profiling_duration=3600)
            )

            estimator = PyTorch(
                framework_version="2.0.0",
                ... # Set up other essential parameters for the estimator class
                profiler_config=profiler_config
            )

        For a complete instruction on activating and using SageMaker Profiler, see
        `Use Amazon SageMaker Profiler to profile activities on AWS compute resources
        <https://docs.aws.amazon.com/sagemaker/latest/dg/train-profile-computational-performance.html>`_.

        """
        assert framework_profile_params is None or isinstance(
            framework_profile_params, FrameworkProfile
        ), "framework_profile_params must be of type FrameworkProfile if specified."

        assert profile_params is None or isinstance(
            profile_params, Profiler
        ), "profile_params must be of type Profiler if specified."

        if profile_params and framework_profile_params:
            raise ValueError("Profiler will not work when Framework Profiler is ON")

        self.s3_output_path = s3_output_path
        self.system_monitor_interval_millis = system_monitor_interval_millis
        self.framework_profile_params = framework_profile_params
        self.profile_params = profile_params
        self.disable_profiler = disable_profiler

        if self.framework_profile_params is not None:
            deprecation_warn_base(
                "Framework profiling will be deprecated from tensorflow 2.12 and pytorch 2.0"
            )

    def _to_request_dict(self):
        """Generate a request dictionary using the parameters provided when initializing the object.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        profiler_config_request = {}

        if (
            self.s3_output_path is not None
            and self.disable_profiler is not None
            and self.disable_profiler is False
        ):
            profiler_config_request["S3OutputPath"] = self.s3_output_path

        profiler_config_request["DisableProfiler"] = self.disable_profiler

        if self.system_monitor_interval_millis is not None:
            profiler_config_request[
                "ProfilingIntervalInMilliseconds"
            ] = self.system_monitor_interval_millis

        if self.framework_profile_params is not None:
            profiler_config_request[
                "ProfilingParameters"
            ] = self.framework_profile_params.profiling_parameters

        if self.profile_params is not None:
            profiler_config_request[
                "ProfilingParameters"
            ] = self.profile_params.profiling_parameters

        return profiler_config_request

    @classmethod
    def _to_profiler_disabled_request_dict(cls):
        """Generate a request dictionary for updating the training job to disable profiler.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        profiler_config_request = {"DisableProfiler": True}
        return profiler_config_request
