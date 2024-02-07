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
"""Classes for using debugger and profiler with Amazon SageMaker."""
from __future__ import absolute_import

from sagemaker.debugger.debugger import (  # noqa: F401
    CollectionConfig,
    DEBUGGER_FLAG,
    DebuggerHookConfig,
    framework_name,
    get_default_profiler_processing_job,
    get_rule_container_image_uri,
    ProfilerRule,
    Rule,
    RuleBase,
    rule_configs,
    TensorBoardOutputConfig,
)
from sagemaker.debugger.framework_profile import FrameworkProfile  # noqa: F401
from sagemaker.debugger.profiler import Profiler  # noqa: F401
from sagemaker.debugger.metrics_config import (  # noqa: F401
    DataloaderProfilingConfig,
    DetailedProfilingConfig,
    SMDataParallelProfilingConfig,
    HorovodProfilingConfig,
    PythonProfilingConfig,
)
from sagemaker.debugger.profiler_config import ProfilerConfig  # noqa: F401
from sagemaker.debugger.utils import PythonProfiler, cProfileTimer  # noqa: F401
