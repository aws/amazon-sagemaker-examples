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
"""Utilities related to logging."""
from __future__ import absolute_import

import logging
import time


class _UTCFormatter(logging.Formatter):
    """Class that overrides the default local time provider in log formatter."""

    converter = time.gmtime


def get_logger():
    """Return a logger with the name 'sagemaker'"""
    sagemaker_logger = logging.getLogger("sagemaker.remote_function")
    if len(sagemaker_logger.handlers) == 0:
        sagemaker_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = _UTCFormatter("%(asctime)s %(name)s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        sagemaker_logger.addHandler(handler)
        # don't stream logs with the root logger handler
        sagemaker_logger.propagate = 0

    return sagemaker_logger
