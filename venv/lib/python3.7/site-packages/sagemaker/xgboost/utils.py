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

from sagemaker.xgboost import defaults


def validate_py_version(py_version):
    """Placeholder docstring"""
    if py_version != "py3":
        raise ValueError("Unsupported Python version: {}.".format(py_version))


def validate_framework_version(framework_version):
    """Placeholder docstring"""

    xgboost_version = framework_version.split("-")[0]
    if xgboost_version in defaults.XGBOOST_UNSUPPORTED_VERSIONS:
        msg = defaults.XGBOOST_UNSUPPORTED_VERSIONS[xgboost_version]
        raise ValueError(msg)
