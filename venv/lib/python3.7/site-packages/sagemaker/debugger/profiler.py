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

"""Configuration for collecting profiler v2 metrics in SageMaker training jobs."""
from __future__ import absolute_import

from sagemaker.debugger.profiler_constants import (
    FILE_ROTATION_INTERVAL_DEFAULT,
    CPU_PROFILING_DURATION,
    DETAIL_PROF_PROCESSING_DEFAULT_INSTANCE_TYPE,
    DETAIL_PROF_PROCESSING_DEFAULT_VOLUME_SIZE,
)


class Profiler:
    """A configuration class to activate SageMaker Profiler."""

    def __init__(
        self,
        cpu_profiling_duration: str = str(CPU_PROFILING_DURATION),
        file_rotation_interval: str = str(FILE_ROTATION_INTERVAL_DEFAULT),
    ):
        """To specify values to adjust the Profiler configuration, use the following parameters.

        :param cpu_profiling_duration: Specify the time duration in seconds for
            profiling CPU activities. The default value is 3600 seconds.
        """
        self.profiling_parameters = {}
        self.profiling_parameters["CPUProfilingDuration"] = str(cpu_profiling_duration)
        self.profiling_parameters["SMPFileRotationSecs"] = str(file_rotation_interval)
        self.instanceType = DETAIL_PROF_PROCESSING_DEFAULT_INSTANCE_TYPE
        self.volumeSizeInGB = DETAIL_PROF_PROCESSING_DEFAULT_VOLUME_SIZE
