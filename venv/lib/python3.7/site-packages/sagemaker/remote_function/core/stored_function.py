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
"""SageMaker job function serializer/deserializer."""
from __future__ import absolute_import

from sagemaker.s3 import s3_path_join
from sagemaker.remote_function import logging_config

import sagemaker.remote_function.core.serialization as serialization
from sagemaker.session import Session


logger = logging_config.get_logger()


FUNCTION_FOLDER = "function"
ARGUMENTS_FOLDER = "arguments"
RESULTS_FOLDER = "results"
EXCEPTION_FOLDER = "exception"


class StoredFunction:
    """Class representing a remote function stored in S3."""

    def __init__(
        self, sagemaker_session: Session, s3_base_uri: str, hmac_key: str, s3_kms_key: str = None
    ):
        """Construct a StoredFunction object.

        Args:
            sagemaker_session: (sagemaker.session.Session): The underlying sagemaker session which
                AWS service calls are delegated to.
            s3_base_uri: the base uri to which serialized artifacts will be uploaded.
            s3_kms_key: KMS key used to encrypt artifacts uploaded to S3.
            hmac_key: Key used to encrypt serialized and deserialied function and arguments
        """
        self.sagemaker_session = sagemaker_session
        self.s3_base_uri = s3_base_uri
        self.s3_kms_key = s3_kms_key
        self.hmac_key = hmac_key

    def save(self, func, *args, **kwargs):
        """Serialize and persist the function and arguments.

        Args:
            func: the python function.
            args: the positional arguments to func.
            kwargs: the keyword arguments to func.
        Returns:
            None
        """
        logger.info(
            f"Serializing function code to {s3_path_join(self.s3_base_uri, FUNCTION_FOLDER)}"
        )
        serialization.serialize_func_to_s3(
            func=func,
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.s3_base_uri, FUNCTION_FOLDER),
            s3_kms_key=self.s3_kms_key,
            hmac_key=self.hmac_key,
        )

        logger.info(
            f"Serializing function arguments to {s3_path_join(self.s3_base_uri, ARGUMENTS_FOLDER)}"
        )
        serialization.serialize_obj_to_s3(
            obj=(args, kwargs),
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.s3_base_uri, ARGUMENTS_FOLDER),
            hmac_key=self.hmac_key,
            s3_kms_key=self.s3_kms_key,
        )

    def load_and_invoke(self) -> None:
        """Load and deserialize the function and the arguments and then execute it."""

        logger.info(
            f"Deserializing function code from {s3_path_join(self.s3_base_uri, FUNCTION_FOLDER)}"
        )
        func = serialization.deserialize_func_from_s3(
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.s3_base_uri, FUNCTION_FOLDER),
            hmac_key=self.hmac_key,
        )

        logger.info(
            f"Deserializing function arguments from {s3_path_join(self.s3_base_uri, ARGUMENTS_FOLDER)}"
        )
        args, kwargs = serialization.deserialize_obj_from_s3(
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.s3_base_uri, ARGUMENTS_FOLDER),
            hmac_key=self.hmac_key,
        )

        logger.info("Invoking the function")
        result = func(*args, **kwargs)

        logger.info(
            f"Serializing the function return and uploading to {s3_path_join(self.s3_base_uri, RESULTS_FOLDER)}"
        )
        serialization.serialize_obj_to_s3(
            obj=result,
            sagemaker_session=self.sagemaker_session,
            s3_uri=s3_path_join(self.s3_base_uri, RESULTS_FOLDER),
            hmac_key=self.hmac_key,
            s3_kms_key=self.s3_kms_key,
        )
