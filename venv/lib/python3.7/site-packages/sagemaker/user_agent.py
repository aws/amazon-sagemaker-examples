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
"""Placeholder docstring"""
from __future__ import absolute_import

import platform
import sys

import importlib_metadata

SDK_VERSION = importlib_metadata.version("sagemaker")
OS_NAME = platform.system() or "UnresolvedOS"
OS_VERSION = platform.release() or "UnresolvedOSVersion"
OS_NAME_VERSION = "{}/{}".format(OS_NAME, OS_VERSION)
PYTHON_VERSION = "Python/{}.{}.{}".format(
    sys.version_info.major, sys.version_info.minor, sys.version_info.micro
)


def determine_prefix(user_agent=""):
    """Placeholder docstring"""
    prefix = "AWS-SageMaker-Python-SDK/{}".format(SDK_VERSION)

    if PYTHON_VERSION not in user_agent:
        prefix = "{} {}".format(prefix, PYTHON_VERSION)

    if OS_NAME_VERSION not in user_agent:
        prefix = "{} {}".format(prefix, OS_NAME_VERSION)

    try:
        with open("/etc/opt/ml/sagemaker-notebook-instance-version.txt") as sagemaker_nbi_file:
            prefix = "{} AWS-SageMaker-Notebook-Instance/{}".format(
                prefix, sagemaker_nbi_file.read().strip()
            )
    except IOError:
        # This file isn't expected to always exist, and we DO want to silently ignore failures.
        pass

    return prefix


def prepend_user_agent(client):
    """Placeholder docstring"""
    prefix = determine_prefix(client._client_config.user_agent)

    if client._client_config.user_agent is None:
        client._client_config.user_agent = prefix
    else:
        client._client_config.user_agent = "{} {}".format(prefix, client._client_config.user_agent)
