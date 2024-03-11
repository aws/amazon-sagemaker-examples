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
"""Definitions for reomote job errors and error handling"""
from __future__ import absolute_import

import os

from tblib import pickling_support
from sagemaker.s3 import s3_path_join
import sagemaker.remote_function.core.serialization as serialization


DEFAULT_FAILURE_CODE = 1
FAILURE_REASON_PATH = "/opt/ml/output/failure"


@pickling_support.install
class RemoteFunctionError(Exception):
    """The base exception class for remote function exceptions"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


@pickling_support.install
class ServiceError(RemoteFunctionError):
    """Raised when errors encountered during interaction with SageMaker, S3 service APIs"""


@pickling_support.install
class SerializationError(RemoteFunctionError):
    """Raised when errors encountered during serialization of remote function objects"""


@pickling_support.install
class DeserializationError(RemoteFunctionError):
    """Raised when errors encountered during deserialization of remote function objects"""


def _get_valid_failure_exit_code(exit_code) -> int:
    """Normalize exit code for terminating the process"""
    try:
        valid_exit_code = int(exit_code)
    except (TypeError, ValueError):
        valid_exit_code = DEFAULT_FAILURE_CODE

    return valid_exit_code


def _write_failure_reason_file(failure_msg):
    """Create a file 'failure' with failure reason written if remote function execution failed.

    See: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    Args:
        failure_msg: The content of file to be written.
    """
    if not os.path.exists(FAILURE_REASON_PATH):
        with open(FAILURE_REASON_PATH, "w") as f:
            f.write(failure_msg)


def handle_error(error, sagemaker_session, s3_base_uri, s3_kms_key, hmac_key) -> int:
    """Handle all exceptions raised during remote function execution.

    Args:
        error (Exception): The error to be handled.
        sagemaker_session (sagemaker.session.Session): The underlying Boto3 session which
             AWS service calls are delegated to.
        s3_base_uri (str): S3 root uri to which resulting serialized exception will be uploaded.
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
        hmac_key (str): Key used to calculate hmac hash of the serialized exception.
    Returns :
        exit_code (int): Exit code to terminate current job.
    """

    failure_reason = repr(error)
    if isinstance(error, RemoteFunctionError):
        exit_code = DEFAULT_FAILURE_CODE
    else:
        error_number = getattr(error, "errno", DEFAULT_FAILURE_CODE)
        exit_code = _get_valid_failure_exit_code(error_number)

    _write_failure_reason_file(failure_reason)

    serialization.serialize_exception_to_s3(
        exc=error,
        sagemaker_session=sagemaker_session,
        s3_uri=s3_path_join(s3_base_uri, "exception"),
        hmac_key=hmac_key,
        s3_kms_key=s3_kms_key,
    )

    return exit_code
