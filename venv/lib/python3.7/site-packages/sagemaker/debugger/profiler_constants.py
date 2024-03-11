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
"""Utils file that contains constants for the profiler."""
from __future__ import absolute_import  # noqa: F401

BASE_FOLDER_DEFAULT = "/opt/ml/output/profiler"
MAX_FILE_SIZE_DEFAULT = 10485760  # default 10MB
CLOSE_FILE_INTERVAL_DEFAULT = 60  # default 60 seconds
FILE_OPEN_FAIL_THRESHOLD_DEFAULT = 50

DETAILED_PROFILING_CONFIG_NAME = "DetailedProfilingConfig"
DATALOADER_PROFILING_CONFIG_NAME = "DataloaderProfilingConfig"
PYTHON_PROFILING_CONFIG_NAME = "PythonProfilingConfig"
HOROVOD_PROFILING_CONFIG_NAME = "HorovodProfilingConfig"
SMDATAPARALLEL_PROFILING_CONFIG_NAME = "SMDataParallelProfilingConfig"

DETAILED_PROFILING_START_STEP_DEFAULT = 5
DATALOADER_PROFILING_START_STEP_DEFAULT = 7
PYTHON_PROFILING_START_STEP_DEFAULT = 9
HOROVOD_PROFILING_START_STEP_DEFAULT = 13
SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT = 15
PROFILING_NUM_STEPS_DEFAULT = 1
START_STEP_DEFAULT = 0
PYTHON_PROFILING_NUM_STEPS_DEFAULT = 3

# These options are used in detail profiler (NOT framework profile)
CPU_PROFILING_DURATION = 3600
FILE_ROTATION_INTERVAL_DEFAULT = 600  # 600 secs
DETAIL_PROF_PROCESSING_DEFAULT_INSTANCE_TYPE = "ml.m5.4xlarge"
DETAIL_PROF_PROCESSING_DEFAULT_VOLUME_SIZE = 128
