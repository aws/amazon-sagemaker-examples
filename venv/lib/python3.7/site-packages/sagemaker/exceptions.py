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
"""Custom exception classes for Sagemaker SDK"""
from __future__ import absolute_import


class UnexpectedStatusException(ValueError):
    """Raised when resource status is not expected and thus not allowed for further execution"""

    def __init__(self, message, allowed_statuses, actual_status):
        self.allowed_statuses = allowed_statuses
        self.actual_status = actual_status
        super(UnexpectedStatusException, self).__init__(message)


class CapacityError(UnexpectedStatusException):
    """Raised when resource status is not expected and fails with a reason of CapacityError"""


class AsyncInferenceError(Exception):
    """The base exception class for Async Inference exceptions."""

    fmt = "An unspecified error occurred"

    def __init__(self, **kwargs):
        msg = self.fmt.format(**kwargs)
        Exception.__init__(self, msg)
        self.kwargs = kwargs


class ObjectNotExistedError(AsyncInferenceError):
    """Raised when Amazon S3 object not exist in the given path"""

    fmt = "Object not exist at {output_path}. {message}"

    def __init__(self, message, output_path):
        super().__init__(message=message, output_path=output_path)


class PollingTimeoutError(AsyncInferenceError):
    """Raised when wait longer than expected and no result object in Amazon S3 bucket yet"""

    fmt = "No result at {output_path} after polling for {seconds} seconds. {message}"

    def __init__(self, message, output_path, seconds):
        super().__init__(message=message, output_path=output_path, seconds=seconds)


class UnexpectedClientError(AsyncInferenceError):
    """Raised when ClientError's error code is not expected"""

    fmt = "Encountered unexpected client error: {message}"

    def __init__(self, message):
        super().__init__(message=message)


class AutoMLStepInvalidModeError(Exception):
    """Raised when the automl mode passed into AutoMLStep in invalid"""

    fmt = (
        "Mode in AutoMLJobConfig must be defined for AutoMLStep. "
        "AutoMLStep currently only supports ENSEMBLING mode"
    )

    def __init__(self, **kwargs):
        msg = self.fmt.format(**kwargs)
        Exception.__init__(self, msg)
        self.kwargs = kwargs


class AsyncInferenceModelError(AsyncInferenceError):
    """Raised when model returns errors for failed requests"""

    fmt = "Model returned error: {message} "

    def __init__(self, message):
        super().__init__(message=message)
