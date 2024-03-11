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
"""The various types of metrics configurations that can be specified in FrameworkProfile."""
from __future__ import absolute_import

from sagemaker.debugger.profiler_constants import (
    DATALOADER_PROFILING_CONFIG_NAME,
    DATALOADER_PROFILING_START_STEP_DEFAULT,
    DETAILED_PROFILING_CONFIG_NAME,
    DETAILED_PROFILING_START_STEP_DEFAULT,
    SMDATAPARALLEL_PROFILING_CONFIG_NAME,
    SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT,
    HOROVOD_PROFILING_CONFIG_NAME,
    HOROVOD_PROFILING_START_STEP_DEFAULT,
    PROFILING_NUM_STEPS_DEFAULT,
    PYTHON_PROFILING_CONFIG_NAME,
    PYTHON_PROFILING_NUM_STEPS_DEFAULT,
    PYTHON_PROFILING_START_STEP_DEFAULT,
    START_STEP_DEFAULT,
)
from sagemaker.debugger.utils import (
    convert_json_config_to_string,
    cProfileTimer,
    is_valid_regex,
    is_valid_unix_time,
    ErrorMessages,
    PythonProfiler,
)


class StepRange:
    """Configuration for the range of steps to profile.

    It returns the target steps in dictionary format that you can pass to the
    :class:`~sagemaker.debugger.FrameworkProfile` class.

    """

    def __init__(self, start_step, num_steps):
        """Set the start step and num steps.

        If the start step is not specified,
        Debugger starts profiling
        at step 0. If num steps is not specified, profile for 1 step.

        Args:
            start_step (int): The step to start profiling.
            num_steps (int): The number of steps to profile.

        """
        if start_step is None:
            start_step = START_STEP_DEFAULT
        elif num_steps is None:
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        self.start_step = start_step
        self.num_steps = num_steps

    def to_json(self):
        """Convert the step range into a dictionary.

        Returns:
            dict: The step range as a dictionary.

        """
        return {"StartStep": self.start_step, "NumSteps": self.num_steps}


class TimeRange:
    """Configuration for the range of Unix time to profile.

    It returns the target time duration in dictionary format that you can pass to the
    :class:`~sagemaker.debugger.FrameworkProfile` class.

    """

    def __init__(self, start_unix_time, duration):
        """Set the start Unix time and duration.

        If the start Unix time is not specified,
        profile starting at step 0. If the duration is not specified, profile for 1 step.

        Args:
            start_unix_time (int): The Unix time to start profiling.
            duration (float): The duration in seconds to profile.

        """
        self.start_unix_time = start_unix_time
        self.duration = duration

    def to_json(self):
        """Convert the time range into a dictionary.

        Returns:
            dict: The time range as a dictionary.

        """
        time_range_json = {}
        if self.start_unix_time is not None:
            time_range_json["StartTimeInSecSinceEpoch"] = self.start_unix_time
        if self.duration is not None:
            time_range_json["Duration"] = self.duration
        return time_range_json


class MetricsConfigBase:
    """The base class for the metrics configuration.

    It determines the step or time range that needs to be
    profiled and validates the input value pairs. Available profiling range parameter pairs are
    (**start_step** and **num_steps**) and (**start_unix_time** and **duration**).
    The two parameter pairs are mutually exclusive, and this class validates
    if one of the two pairs is used. If both pairs are specified, a
    FOUND_BOTH_STEP_AND_TIME_FIELDS error occurs.

    """

    def __init__(self, name, start_step, num_steps, start_unix_time, duration):
        """Validate the provided range fields and set the range to be profiled accordingly.

        Args:
            name (str): The name of the metrics config.
            start_step (int): The step to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The Unix time to start profiling.
            duration (float): The duration in seconds to profile.

        """
        self.name = name

        assert (
            start_step is None or isinstance(start_step, int) and start_step >= 0
        ), ErrorMessages.INVALID_START_STEP.value
        assert (
            num_steps is None or isinstance(num_steps, int) and num_steps > 0
        ), ErrorMessages.INVALID_NUM_STEPS.value
        assert (
            start_unix_time is None
            or isinstance(start_unix_time, int)
            and is_valid_unix_time(start_unix_time)
        ), ErrorMessages.INVALID_START_UNIX_TIME.value
        assert (
            duration is None or isinstance(duration, (float, int)) and duration > 0
        ), ErrorMessages.INVALID_DURATION.value

        has_step_range = start_step is not None or num_steps is not None
        has_time_range = start_unix_time is not None or duration is not None
        assert not (
            has_step_range and has_time_range
        ), ErrorMessages.FOUND_BOTH_STEP_AND_TIME_FIELDS.value

        self.range = (
            StepRange(start_step, num_steps)
            if has_step_range
            else TimeRange(start_unix_time, duration)
        )

    def _to_json(self):
        """Convert the metrics configuration to a dictionary.

        Convert the range object into a
        dictionary.

        Returns:
            dict: This metrics config as a dictionary.

        """
        return self.range.to_json()

    def to_json_string(self):
        """Convert this metrics configuration to dictionary formatted as a string.

        Calling eval on the
        return value is the same as calling _to_json directly.

        Returns:
            str: This metrics configuration as a dictionary and formatted as a string.

        """
        return convert_json_config_to_string(self._to_json())


class DetailedProfilingConfig(MetricsConfigBase):
    """The configuration for framework metrics to be collected for detailed profiling."""

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
    ):
        """Specify target steps or a target duration to profile.

        By default, it profiles step 5 of the training job.

        If **profile_default_steps** is set to `True` and none of the other
        range parameters is specified,
        the class uses the default configuration for detailed profiling.

        Args:
            start_step (int): The step to start profiling. The default is step 5.
            num_steps (int): The number of steps to profile. The default is for 1 step.
            start_unix_time (int): The Unix time to start profiling.
            duration (float): The duration in seconds to profile.
            profile_default_steps (bool): Indicates whether the default config should be used.

        .. tip::
            Available profiling range parameter pairs are
            (**start_step** and **num_steps**) and (**start_unix_time** and **duration**).
            The two parameter pairs are mutually exclusive, and this class validates
            if one of the two pairs is used. If both pairs are specified, a
            conflict error occurs.

        .. warning::
            This detailed framework profiling feature discontinues support for TensorFlow v2.11
            and later. To use the detailed profiling feature, use previous versions of
            TensorFlow between v2.3.1 and v2.10.0.

        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = DETAILED_PROFILING_START_STEP_DEFAULT
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            DETAILED_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )


class DataloaderProfilingConfig(MetricsConfigBase):
    """The configuration for framework metrics to be collected for data loader profiling."""

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
        metrics_regex=".*",
    ):
        """Specify target steps or a target duration to profile.

        By default, it profiles step 7 of
        training. If **profile_default_steps** is set to `True` and none of the other
        range parameters is specified,
        the class uses the default config for dataloader profiling.

        Args:
            start_step (int): The step to start profiling. The default is step 7.
            num_steps (int): The number of steps to profile. The default is for 1 step.
            start_unix_time (int): The Unix time to start profiling. The default is for 1 step.
            duration (float): The duration in seconds to profile.
            profile_default_steps (bool): Indicates whether the default config should be used.

        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = DATALOADER_PROFILING_START_STEP_DEFAULT
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            DATALOADER_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )

        assert is_valid_regex(metrics_regex), ErrorMessages.INVALID_METRICS_REGEX.value
        self.metrics_regex = metrics_regex

    def _to_json(self):
        """Convert the dataloader profiling config to a dictionary.

        Build off of the base metrics
        configuration dictionary to add the metrics regex.

        Returns:
            dict: The dataloader that profiles the configuration as a dictionary.

        """
        dataloader_profiling_config = super()._to_json()
        dataloader_profiling_config["MetricsRegex"] = self.metrics_regex
        return dataloader_profiling_config


class PythonProfilingConfig(MetricsConfigBase):
    """The configuration for framework metrics to be collected for Python profiling."""

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
        python_profiler=PythonProfiler.CPROFILE,
        cprofile_timer=cProfileTimer.TOTAL_TIME,
    ):
        """Choose a Python profiler: cProfile or Pyinstrument.

        Specify target steps or a target duration to profile.
        If no parameter is specified,
        it profiles based on profiling configurations
        preset by the **profile_default_steps** parameter,
        which is set to `True` by default.
        If you specify the following parameters,
        then the **profile_default_steps** parameter
        will be ignored.

        Args:
            start_step (int): The step to start profiling. The default is step 9.
            num_steps (int): The number of steps to profile. The default is for 3 steps.
            start_unix_time (int): The Unix time to start profiling.
            duration (float): The duration in seconds to profile.
            profile_default_steps (bool): Indicates whether the default configuration
                should be used. If set to `True`, Python profiling will be done
                at step 9, 10, and 11 of training, using cProfiler
                and collecting metrics based on the total time, cpu time,
                and off cpu time for these three steps respectively.
                The default is ``True``.
            python_profiler (PythonProfiler): The Python profiler to use to collect
                python profiling stats. Available options are ``"cProfile"``
                and ``"Pyinstrument"``. The default is ``"cProfile"``.
                Instead of passing the string values, you can also use the enumerator util,
                :class:`~sagemaker.debugger.utils.PythonProfiler`,
                to choose one of the available options.
            cprofile_timer (cProfileTimer): The timer to be used by cProfile when collecting
                python profiling stats. Available options are ``"total_time"``, ``"cpu_time"``,
                and ``"off_cpu_time"``. The default is ``"total_time"``.
                If you choose Pyinstrument, this parameter is ignored.
                Instead of passing the string values, you can also use the enumerator util,
                :class:`~sagemaker.debugger.utils.cProfileTimer`,
                to choose one of the available options.

        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = PYTHON_PROFILING_START_STEP_DEFAULT
            num_steps = PYTHON_PROFILING_NUM_STEPS_DEFAULT

        if profile_default_steps:
            cprofile_timer = cProfileTimer.DEFAULT

        super().__init__(
            PYTHON_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )

        assert isinstance(
            python_profiler, PythonProfiler
        ), ErrorMessages.INVALID_PYTHON_PROFILER.value
        assert isinstance(cprofile_timer, cProfileTimer), ErrorMessages.INVALID_CPROFILE_TIMER.value

        self.python_profiler = python_profiler

        # The cprofile timer can only be used when the python profiler is cProfile.
        if python_profiler == PythonProfiler.PYINSTRUMENT:
            self.cprofile_timer = None
        else:
            self.cprofile_timer = cprofile_timer

    def _to_json(self):
        """Convert the Python profiling config to a dictionary.

        Build off of the base metrics configuration
        dictionary to add the Python profiler and cProfile timer.

        Returns:
            dict: The python profiling config as a dictionary.

        """
        python_profiling_config = super()._to_json()
        python_profiling_config["ProfilerName"] = self.python_profiler.value
        if self.cprofile_timer is not None:
            python_profiling_config["cProfileTimer"] = self.cprofile_timer.value
        return python_profiling_config


class HorovodProfilingConfig(MetricsConfigBase):
    """The configuration for framework metrics from Horovod distributed training."""

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
    ):
        """Specify target steps or a target duration to profile.

        By default, it profiles step 13 of training.
        If **profile_default_steps** is set to `True` and none of the other range
        parameters is specified,
        the class uses the default config for horovod profiling.

        Args:
            start_step (int): The step to start profiling. The default is step 13.
            num_steps (int): The number of steps to profile. The default is for 1 steps.
            start_unix_time (int): The Unix time to start profiling.
            duration (float): The duration in seconds to profile.
            profile_default_steps (bool): Indicates whether the default config should be used.

        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = HOROVOD_PROFILING_START_STEP_DEFAULT
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            HOROVOD_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )


class SMDataParallelProfilingConfig(MetricsConfigBase):
    """Configuration for framework metrics collected from a SageMaker Distributed training job."""

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
    ):
        """Specify target steps or a target duration to profile.

        By default, it profiles step 15 of training.
        If **profile_default_steps** is set to `True` and none of the other
        range parameters is specified,
        the class uses the default configuration for SageMaker Distributed profiling.

        Args:
            start_step (int): The step to start profiling. The default is step 15.
            num_steps (int): The number of steps to profile. The default is for 1 steps.
            start_unix_time (int): The Unix time to start profiling.
            duration (float): The duration in seconds to profile.
            profile_default_steps (bool): Indicates whether the default configuration
                should be used.

        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            SMDATAPARALLEL_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )
