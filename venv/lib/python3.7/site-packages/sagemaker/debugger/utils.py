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
"""Utils file that contains util functions for the profiler."""
from __future__ import absolute_import

import re
from datetime import datetime
from enum import Enum


def _convert_key_and_value(key, value):
    """Helper function to convert the provided key and value pair (from a dictionary) to a string.

    Args:
        key (str): The key in the dictionary.
        value: The value for this key.

    Returns:
        str: The provided key value pair as a string.

    """
    updated_key = f'"{key}"' if isinstance(key, str) else key
    updated_value = f'"{value}"' if isinstance(value, str) else value

    return f"{updated_key}: {updated_value}, "


def convert_json_config_to_string(config):
    """Helper function to convert the dictionary config to a string.

    Calling eval on this string should result in the original dictionary.

    Args:
        config (dict): The config to be converted to a string.

    Returns:
        str: The config dictionary formatted as a string.

    """
    json_string = "{"
    for key, value in config.items():
        json_string += _convert_key_and_value(key, value)
    json_string += "}"
    return json_string


def is_valid_unix_time(unix_time):
    """Helper function to determine whether the provided UNIX time is valid.

    Args:
        unix_time (int): The user provided UNIX time.

    Returns:
        bool: Indicates whether the provided UNIX time was valid or not.

    """
    try:
        datetime.fromtimestamp(unix_time)
        return True
    except (OverflowError, ValueError):
        return False


def is_valid_regex(regex):
    """Helper function to determine whether the provided regex is valid.

    Args:
        regex (str): The user provided regex.

    Returns:
        bool: Indicates whether the provided regex was valid or not.

    """
    try:
        re.compile(regex)
        return True
    except (re.error, TypeError):
        return False


class ErrorMessages(Enum):
    """Enum to store all possible messages during failures in validation of user arguments."""

    INVALID_LOCAL_PATH = "local_path must be a string!"
    INVALID_FILE_MAX_SIZE = "file_max_size must be an integer greater than 0!"
    INVALID_FILE_CLOSE_INTERVAL = "file_close_interval must be a float/integer greater than 0!"
    INVALID_FILE_OPEN_FAIL_THRESHOLD = "file_open_fail threshold must be an integer greater than 0!"
    INVALID_PROFILE_DEFAULT_STEPS = "profile_default_steps must be a boolean!"
    INVALID_START_STEP = "start_step must be integer greater or equal to 0!"
    INVALID_NUM_STEPS = "num_steps must be integer greater than 0!"
    INVALID_START_UNIX_TIME = "start_unix_time must be valid integer unix time!"
    INVALID_DURATION = "duration must be float greater than 0!"
    FOUND_BOTH_STEP_AND_TIME_FIELDS = (
        "Both step and time fields cannot be specified in the metrics config!"
    )
    INVALID_METRICS_REGEX = "metrics_regex is invalid!"
    INVALID_PYTHON_PROFILER = "python_profiler must be of type PythonProfiler!"
    INVALID_CPROFILE_TIMER = "cprofile_timer must be of type cProfileTimer"


class PythonProfiler(Enum):
    """Enum to list the Python profiler options for Python profiling.

    .. py:attribute:: CPROFILE

        Use to choose ``"cProfile"``.

    .. py:attribute:: PYINSTRUMENT

        Use to choose ``"Pyinstrument"``.

    """

    CPROFILE = "cprofile"
    PYINSTRUMENT = "pyinstrument"


class cProfileTimer(Enum):
    """Enum to list the possible cProfile timers for Python profiling.

    .. py:attribute:: TOTAL_TIME

        Use to choose ``"total_time"``.

    .. py:attribute:: CPU_TIME

        Use to choose ``"cpu_time"``.

    .. py:attribute:: OFF_CPU_TIME

        Use to choose ``"off_cpu_time"``.

    """

    TOTAL_TIME = "total_time"
    CPU_TIME = "cpu_time"
    OFF_CPU_TIME = "off_cpu_time"
    DEFAULT = "default"
