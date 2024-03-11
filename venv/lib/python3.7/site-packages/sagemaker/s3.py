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
"""This module contains Enums and helper methods related to S3."""
from __future__ import print_function, absolute_import

import logging
import io

from typing import Union
from sagemaker.session import Session

# These were defined inside s3.py initially. Kept here for backward compatibility
from sagemaker.s3_utils import (  # pylint: disable=unused-import # noqa: F401
    parse_s3_url,
    s3_path_join,
    determine_bucket_and_prefix,
)

logger = logging.getLogger("sagemaker")


class S3Uploader(object):
    """Contains static methods for uploading directories or files to S3."""

    @staticmethod
    def upload(local_path, desired_s3_uri, kms_key=None, sagemaker_session=None):
        """Static method that uploads a given file or directory to S3.

        Args:
            local_path (str): Path (absolute or relative) of local file or directory to upload.
            desired_s3_uri (str): The desired S3 location to upload to. It is the prefix to
                which the local filename will be added.
            kms_key (str): The KMS key to use to encrypt the files.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            The S3 uri of the uploaded file(s).

        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=desired_s3_uri)
        if kms_key is not None:
            extra_args = {"SSEKMSKeyId": kms_key, "ServerSideEncryption": "aws:kms"}

        else:
            extra_args = None

        return sagemaker_session.upload_data(
            path=local_path, bucket=bucket, key_prefix=key_prefix, extra_args=extra_args
        )

    @staticmethod
    def upload_string_as_file_body(
        body: str, desired_s3_uri=None, kms_key=None, sagemaker_session=None
    ):
        """Static method that uploads a given file or directory to S3.

        Args:
            body (str): String representing the body of the file.
            desired_s3_uri (str): The desired S3 uri to upload to.
            kms_key (str): The KMS key to use to encrypt the files.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.

        Returns:
            str: The S3 uri of the uploaded file.

        """
        sagemaker_session = sagemaker_session or Session()

        bucket, key = parse_s3_url(desired_s3_uri)

        sagemaker_session.upload_string_as_file_body(
            body=body, bucket=bucket, key=key, kms_key=kms_key
        )

        return desired_s3_uri

    @staticmethod
    def upload_bytes(b: Union[bytes, io.BytesIO], s3_uri, kms_key=None, sagemaker_session=None):
        """Static method that uploads a given file or directory to S3.

        Args:
            b (bytes or io.BytesIO): bytes.
            s3_uri (str): The S3 uri to upload to.
            kms_key (str): The KMS key to use to encrypt the files.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            str: The S3 uri of the uploaded file.

        """
        sagemaker_session = sagemaker_session or Session()

        bucket, object_key = parse_s3_url(s3_uri)

        if kms_key is not None:
            extra_args = {"SSEKMSKeyId": kms_key, "ServerSideEncryption": "aws:kms"}
        else:
            extra_args = None

        b = b if isinstance(b, io.BytesIO) else io.BytesIO(b)
        sagemaker_session.s3_resource.Bucket(bucket).upload_fileobj(
            b, object_key, ExtraArgs=extra_args
        )

        return s3_uri


class S3Downloader(object):
    """Contains static methods for downloading directories or files from S3."""

    @staticmethod
    def download(s3_uri, local_path, kms_key=None, sagemaker_session=None):
        """Static method that downloads a given S3 uri to the local machine.

        Args:
            s3_uri (str): An S3 uri to download from.
            local_path (str): A local path to download the file(s) to.
            kms_key (str): The KMS key to use to decrypt the files.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            list[str]: List of local paths of downloaded files
        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=s3_uri)
        if kms_key is not None:
            extra_args = {"SSECustomerKey": kms_key}
        else:
            extra_args = None

        return sagemaker_session.download_data(
            path=local_path, bucket=bucket, key_prefix=key_prefix, extra_args=extra_args
        )

    @staticmethod
    def read_file(s3_uri, sagemaker_session=None) -> str:
        """Static method that returns the contents of a s3 uri file body as a string.

        Args:
            s3_uri (str): An S3 uri that refers to a single file.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            str: The body of the file.
        """
        sagemaker_session = sagemaker_session or Session()

        bucket, object_key = parse_s3_url(url=s3_uri)

        return sagemaker_session.read_s3_file(bucket=bucket, key_prefix=object_key)

    @staticmethod
    def read_bytes(s3_uri, sagemaker_session=None) -> bytes:
        """Static method that returns the contents of a s3 object as bytes.

        Args:
            s3_uri (str): An S3 uri that refers to a s3 object.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            bytes: The body of the file.
        """
        sagemaker_session = sagemaker_session or Session()

        bucket, object_key = parse_s3_url(s3_uri)

        bytes_io = io.BytesIO()
        sagemaker_session.s3_resource.Bucket(bucket).download_fileobj(object_key, bytes_io)
        bytes_io.seek(0)
        return bytes_io.read()

    @staticmethod
    def list(s3_uri, sagemaker_session=None):
        """Static method that lists the contents of an S3 uri.

        Args:
            s3_uri (str): The S3 base uri to list objects in.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            [str]: The list of S3 URIs in the given S3 base uri.
        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=s3_uri)

        file_keys = sagemaker_session.list_s3_files(bucket=bucket, key_prefix=key_prefix)
        return [s3_path_join("s3://", bucket, file_key) for file_key in file_keys]
