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
"""Configuration for collecting framework metrics in SageMaker training jobs."""
from __future__ import absolute_import

from sagemaker.debugger.metrics_config import (
    DetailedProfilingConfig,
    DataloaderProfilingConfig,
    SMDataParallelProfilingConfig,
    HorovodProfilingConfig,
    PythonProfilingConfig,
)
from sagemaker.debugger.profiler_constants import (
    BASE_FOLDER_DEFAULT,
    CLOSE_FILE_INTERVAL_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
)
from sagemaker.debugger.utils import ErrorMessages

ALL_METRIC_CONFIGS = [
    DetailedProfilingConfig,
    DataloaderProfilingConfig,
    PythonProfilingConfig,
    HorovodProfilingConfig,
    SMDataParallelProfilingConfig,
]


class FrameworkProfile:
    """Sets up the profiling configuration for framework metrics.

    Validates user inputs and fills in default values if no input is provided.
    There are three main profiling options to choose from:
    :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig`,
    :class:`~sagemaker.debugger.metrics_config.DataloaderProfilingConfig`, and
    :class:`~sagemaker.debugger.metrics_config.PythonProfilingConfig`.

    The following list shows available scenarios of configuring the profiling options.

    1. None of the profiling configuration, step range, or time range is specified.
    SageMaker Debugger activates framework profiling based on the default settings
    of each profiling option.

    .. code-block:: python

        from sagemaker.debugger import ProfilerConfig, FrameworkProfile

        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile()
        )

    2. Target step or time range is specified to
    this :class:`~sagemaker.debugger.metrics_config.FrameworkProfile` class.
    The requested target step or time range setting propagates to all of
    the framework profiling options.
    For example, if you configure this class as following, all of the profiling options
    profiles the 6th step:

    .. code-block:: python

        from sagemaker.debugger import ProfilerConfig, FrameworkProfile

        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile(start_step=6, num_steps=1)
        )

    3. Individual profiling configurations are specified through
    the ``*_profiling_config`` parameters.
    SageMaker Debugger profiles framework metrics only for the specified profiling configurations.
    For example, if the :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class
    is configured but not the other profiling options, Debugger only profiles based on the settings
    specified to the
    :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class.
    For example, the following example shows a profiling configuration to perform
    detailed profiling at step 10, data loader profiling at step 9 and 10,
    and Python profiling at step 12.

    .. code-block:: python

        from sagemaker.debugger import ProfilerConfig, FrameworkProfile

        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile(
                detailed_profiling_config=DetailedProfilingConfig(start_step=10, num_steps=1),
                dataloader_profiling_config=DataloaderProfilingConfig(start_step=9, num_steps=2),
                python_profiling_config=PythonProfilingConfig(start_step=12, num_steps=1),
            )
        )

    If the individual profiling configurations are specified in addition to
    the step or time range,
    SageMaker Debugger prioritizes the individual profiling configurations and ignores
    the step or time range. For example, in the following code,
    the ``start_step=1`` and ``num_steps=10`` will be ignored.

    .. code-block:: python

        from sagemaker.debugger import ProfilerConfig, FrameworkProfile

        profiler_config=ProfilerConfig(
            framework_profile_params=FrameworkProfile(
                start_step=1,
                num_steps=10,
                detailed_profiling_config=DetailedProfilingConfig(start_step=10, num_steps=1),
                dataloader_profiling_config=DataloaderProfilingConfig(start_step=9, num_steps=2),
                python_profiling_config=PythonProfilingConfig(start_step=12, num_steps=1)
            )
        )

    """

    def __init__(
        self,
        local_path=BASE_FOLDER_DEFAULT,
        file_max_size=MAX_FILE_SIZE_DEFAULT,
        file_close_interval=CLOSE_FILE_INTERVAL_DEFAULT,
        file_open_fail_threshold=FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
        detailed_profiling_config=None,
        dataloader_profiling_config=None,
        python_profiling_config=None,
        horovod_profiling_config=None,
        smdataparallel_profiling_config=None,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
    ):
        """Initialize the FrameworkProfile class object.

        Args:
            detailed_profiling_config (DetailedProfilingConfig): The configuration for detailed
                profiling. Configure it using the
                :class:`~sagemaker.debugger.metrics_config.DetailedProfilingConfig` class.
                Pass ``DetailedProfilingConfig()`` to use the default configuration.

                .. warning::
                    This detailed framework profiling feature discontinues support for TensorFlow v2.11
                    and later. To use the detailed profiling feature, use previous versions of
                    TensorFlow between v2.3.1 and v2.10.0.

            dataloader_profiling_config (DataloaderProfilingConfig): The configuration for
                dataloader metrics profiling. Configure it using the
                :class:`~sagemaker.debugger.metrics_config.DataloaderProfilingConfig` class.
                Pass ``DataloaderProfilingConfig()`` to use the default configuration.
            python_profiling_config (PythonProfilingConfig): The configuration for stats
                collected by the Python profiler (cProfile or Pyinstrument).
                Configure it using the
                :class:`~sagemaker.debugger.metrics_config.PythonProfilingConfig` class.
                Pass ``PythonProfilingConfig()`` to use the default configuration.
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The Unix time at which to start profiling.
            duration (float): The duration in seconds to profile.

        .. tip::
            Available profiling range parameter pairs are
            (**start_step** and **num_steps**) and (**start_unix_time** and **duration**).
            The two parameter pairs are mutually exclusive, and this class validates
            if one of the two pairs is used. If both pairs are specified, a
            conflict error occurs.

        """
        self.profiling_parameters = {}
        self._use_default_metrics_configs = False
        self._use_one_config_for_all_metrics = False
        self._use_custom_metrics_configs = False

        self._process_trace_file_parameters(
            local_path, file_max_size, file_close_interval, file_open_fail_threshold
        )
        use_custom_metrics_configs = self._process_metrics_configs(
            detailed_profiling_config,
            dataloader_profiling_config,
            python_profiling_config,
            horovod_profiling_config,
            smdataparallel_profiling_config,
        )

        use_one_config_for_all_metrics = (
            self._process_range_fields(start_step, num_steps, start_unix_time, duration)
            if not use_custom_metrics_configs
            else False
        )

        if not use_custom_metrics_configs and not use_one_config_for_all_metrics:
            self._create_default_metrics_configs()

    def _process_trace_file_parameters(
        self, local_path, file_max_size, file_close_interval, file_open_fail_threshold
    ):
        """Helper function to validate and set the provided trace file parameters.

        Args:
            local_path (str): The path where profiler events have to be saved.
            file_max_size (int): Max size a trace file can be, before being rotated.
            file_close_interval (float): Interval in seconds from the last close, before being
                rotated.
            file_open_fail_threshold (int): Number of times to attempt to open a trace fail before
                marking the writer as unhealthy.

        """
        assert isinstance(local_path, str), ErrorMessages.INVALID_LOCAL_PATH.value
        assert (
            isinstance(file_max_size, int) and file_max_size > 0
        ), ErrorMessages.INVALID_FILE_MAX_SIZE.value
        assert (
            isinstance(file_close_interval, (float, int)) and file_close_interval > 0
        ), ErrorMessages.INVALID_FILE_CLOSE_INTERVAL.value
        assert (
            isinstance(file_open_fail_threshold, int) and file_open_fail_threshold > 0
        ), ErrorMessages.INVALID_FILE_OPEN_FAIL_THRESHOLD.value

        self.profiling_parameters["LocalPath"] = local_path
        self.profiling_parameters["RotateMaxFileSizeInBytes"] = str(file_max_size)
        self.profiling_parameters["RotateFileCloseIntervalInSeconds"] = str(file_close_interval)
        self.profiling_parameters["FileOpenFailThreshold"] = str(file_open_fail_threshold)

    def _process_metrics_configs(self, *metrics_configs):
        """Helper function to validate and set the provided metrics_configs.

        In this case,
        the user specifies configurations for the metrics they want to profile.
        Profiling does not occur
        for metrics if the configurations are not specified for them.

        Args:
            metrics_configs: The list of metrics configs specified by the user.

        Returns:
            bool: Indicates whether custom metrics configs will be used for profiling.

        """
        metrics_configs = [config for config in metrics_configs if config is not None]
        if len(metrics_configs) == 0:
            return False

        for config in metrics_configs:
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json
        return True

    def _process_range_fields(self, start_step, num_steps, start_unix_time, duration):
        """Helper function to validate and set the provided range fields.

        Profiling occurs
        for all of the metrics using these fields as the specified range and default parameters
        for the rest of the configuration fields (if necessary).

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile.

        Returns:
            bool: Indicates whether a custom step or time range will be used for profiling.

        """
        if start_step is num_steps is start_unix_time is duration is None:
            return False

        for config_class in ALL_METRIC_CONFIGS:
            config = config_class(
                start_step=start_step,
                num_steps=num_steps,
                start_unix_time=start_unix_time,
                duration=duration,
            )
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json
        return True

    def _create_default_metrics_configs(self):
        """Helper function for creating the default configs for each set of metrics."""
        for config_class in ALL_METRIC_CONFIGS:
            config = config_class(profile_default_steps=True)
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json
