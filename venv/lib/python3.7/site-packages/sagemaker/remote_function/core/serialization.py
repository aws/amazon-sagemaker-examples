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
"""SageMaker remote function data serializer/deserializer."""
from __future__ import absolute_import

import dataclasses
import json
import os
import sys
import hmac
import hashlib

import cloudpickle

from typing import Any, Callable
from sagemaker.remote_function.errors import ServiceError, SerializationError, DeserializationError
from sagemaker.s3 import S3Downloader, S3Uploader
from sagemaker.session import Session

from tblib import pickling_support


def _get_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@dataclasses.dataclass
class _MetaData:
    """Metadata about the serialized data or functions."""

    sha256_hash: str
    version: str = "2023-04-24"
    python_version: str = _get_python_version()
    serialization_module: str = "cloudpickle"

    def to_json(self):
        return json.dumps(dataclasses.asdict(self)).encode()

    @staticmethod
    def from_json(s):
        try:
            obj = json.loads(s)
        except json.decoder.JSONDecodeError:
            raise DeserializationError("Corrupt metadata file. It is not a valid json file.")

        sha256_hash = obj.get("sha256_hash")
        metadata = _MetaData(sha256_hash=sha256_hash)
        metadata.version = obj.get("version")
        metadata.python_version = obj.get("python_version")
        metadata.serialization_module = obj.get("serialization_module")

        if not sha256_hash:
            raise DeserializationError(
                "Corrupt metadata file. SHA256 hash for the serialized data does not exist. "
                "Please make sure to install SageMaker SDK version >= 2.156.0 on the client side "
                "and try again."
            )

        if not (
            metadata.version == "2023-04-24" and metadata.serialization_module == "cloudpickle"
        ):
            raise DeserializationError(
                f"Corrupt metadata file. Serialization approach {s} is not supported."
            )

        return metadata


class CloudpickleSerializer:
    """Serializer using cloudpickle."""

    @staticmethod
    def serialize(obj: Any) -> Any:
        """Serializes data object and uploads it to S3.

        Args:
            obj: object to be serialized and persisted
        Raises:
            SerializationError: when fail to serialize object to bytes.
        """
        try:
            return cloudpickle.dumps(obj)
        except Exception as e:
            if isinstance(
                e, NotImplementedError
            ) and "Instance of Run type is not allowed to be pickled." in str(e):
                raise SerializationError(
                    """You are trying to pass a sagemaker.experiments.run.Run object to a remote function
                       or are trying to access a global sagemaker.experiments.run.Run object from within the function.
                       This is not supported. You must use `load_run` to load an existing Run in the remote function
                       or instantiate a new Run in the function."""
                ) from e

            raise SerializationError(
                "Error when serializing object of type [{}]: {}".format(type(obj).__name__, repr(e))
            ) from e

    @staticmethod
    def deserialize(s3_uri: str, bytes_to_deserialize) -> Any:
        """Downloads from S3 and then deserializes data objects.

        Args:
            sagemaker_session (sagemaker.session.Session): The underlying sagemaker session which AWS service
                 calls are delegated to.
            s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        Returns :
            List of deserialized python objects.
        Raises:
            DeserializationError: when fail to serialize object to bytes.
        """

        try:
            return cloudpickle.loads(bytes_to_deserialize)
        except Exception as e:
            raise DeserializationError(
                "Error when deserializing bytes downloaded from {}: {}".format(s3_uri, repr(e))
            ) from e


# TODO: use dask serializer in case dask distributed is installed in users' environment.
def serialize_func_to_s3(
    func: Callable, sagemaker_session: Session, s3_uri: str, hmac_key: str, s3_kms_key: str = None
):
    """Serializes function and uploads it to S3.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying Boto3 session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        hmac_key (str): Key used to calculate hmac-sha256 hash of the serialized func.
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
        func: function to be serialized and persisted
    Raises:
        SerializationError: when fail to serialize function to bytes.
    """

    bytes_to_upload = CloudpickleSerializer.serialize(func)

    _upload_bytes_to_s3(
        bytes_to_upload, os.path.join(s3_uri, "payload.pkl"), s3_kms_key, sagemaker_session
    )

    sha256_hash = _compute_hash(bytes_to_upload, secret_key=hmac_key)

    _upload_bytes_to_s3(
        _MetaData(sha256_hash).to_json(),
        os.path.join(s3_uri, "metadata.json"),
        s3_kms_key,
        sagemaker_session,
    )


def deserialize_func_from_s3(sagemaker_session: Session, s3_uri: str, hmac_key: str) -> Callable:
    """Downloads from S3 and then deserializes data objects.

    This method downloads the serialized training job outputs to a temporary directory and
    then deserializes them using dask.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying sagemaker session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        hmac_key (str): Key used to calculate hmac-sha256 hash of the serialized func.
    Returns :
        The deserialized function.
    Raises:
        DeserializationError: when fail to serialize function to bytes.
    """
    metadata = _MetaData.from_json(
        _read_bytes_from_s3(os.path.join(s3_uri, "metadata.json"), sagemaker_session)
    )

    bytes_to_deserialize = _read_bytes_from_s3(
        os.path.join(s3_uri, "payload.pkl"), sagemaker_session
    )

    _perform_integrity_check(
        expected_hash_value=metadata.sha256_hash, secret_key=hmac_key, buffer=bytes_to_deserialize
    )

    return CloudpickleSerializer.deserialize(
        os.path.join(s3_uri, "payload.pkl"), bytes_to_deserialize
    )


def serialize_obj_to_s3(
    obj: Any, sagemaker_session: Session, s3_uri: str, hmac_key: str, s3_kms_key: str = None
):
    """Serializes data object and uploads it to S3.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying Boto3 session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
        hmac_key (str): Key used to calculate hmac-sha256 hash of the serialized obj.
        obj: object to be serialized and persisted
    Raises:
        SerializationError: when fail to serialize object to bytes.
    """

    bytes_to_upload = CloudpickleSerializer.serialize(obj)

    _upload_bytes_to_s3(
        bytes_to_upload, os.path.join(s3_uri, "payload.pkl"), s3_kms_key, sagemaker_session
    )

    sha256_hash = _compute_hash(bytes_to_upload, secret_key=hmac_key)

    _upload_bytes_to_s3(
        _MetaData(sha256_hash).to_json(),
        os.path.join(s3_uri, "metadata.json"),
        s3_kms_key,
        sagemaker_session,
    )


def deserialize_obj_from_s3(sagemaker_session: Session, s3_uri: str, hmac_key: str) -> Any:
    """Downloads from S3 and then deserializes data objects.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying sagemaker session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        hmac_key (str): Key used to calculate hmac-sha256 hash of the serialized obj.
    Returns :
        Deserialized python objects.
    Raises:
        DeserializationError: when fail to serialize object to bytes.
    """

    metadata = _MetaData.from_json(
        _read_bytes_from_s3(os.path.join(s3_uri, "metadata.json"), sagemaker_session)
    )

    bytes_to_deserialize = _read_bytes_from_s3(
        os.path.join(s3_uri, "payload.pkl"), sagemaker_session
    )

    _perform_integrity_check(
        expected_hash_value=metadata.sha256_hash, secret_key=hmac_key, buffer=bytes_to_deserialize
    )

    return CloudpickleSerializer.deserialize(
        os.path.join(s3_uri, "payload.pkl"), bytes_to_deserialize
    )


def serialize_exception_to_s3(
    exc: Exception, sagemaker_session: Session, s3_uri: str, hmac_key: str, s3_kms_key: str = None
):
    """Serializes exception with traceback and uploads it to S3.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying Boto3 session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        s3_kms_key (str): KMS key used to encrypt artifacts uploaded to S3.
        hmac_key (str): Key used to calculate hmac-sha256 hash of the serialized exception.
        exc: Exception to be serialized and persisted
    Raises:
        SerializationError: when fail to serialize object to bytes.
    """
    pickling_support.install()

    bytes_to_upload = CloudpickleSerializer.serialize(exc)

    _upload_bytes_to_s3(
        bytes_to_upload, os.path.join(s3_uri, "payload.pkl"), s3_kms_key, sagemaker_session
    )

    sha256_hash = _compute_hash(bytes_to_upload, secret_key=hmac_key)

    _upload_bytes_to_s3(
        _MetaData(sha256_hash).to_json(),
        os.path.join(s3_uri, "metadata.json"),
        s3_kms_key,
        sagemaker_session,
    )


def deserialize_exception_from_s3(sagemaker_session: Session, s3_uri: str, hmac_key: str) -> Any:
    """Downloads from S3 and then deserializes exception.

    Args:
        sagemaker_session (sagemaker.session.Session): The underlying sagemaker session which AWS service
             calls are delegated to.
        s3_uri (str): S3 root uri to which resulting serialized artifacts will be uploaded.
        hmac_key (str): Key used to calculate hmac-sha256 hash of the serialized exception.
    Returns :
        Deserialized exception with traceback.
    Raises:
        DeserializationError: when fail to serialize object to bytes.
    """

    metadata = _MetaData.from_json(
        _read_bytes_from_s3(os.path.join(s3_uri, "metadata.json"), sagemaker_session)
    )

    bytes_to_deserialize = _read_bytes_from_s3(
        os.path.join(s3_uri, "payload.pkl"), sagemaker_session
    )

    _perform_integrity_check(
        expected_hash_value=metadata.sha256_hash, secret_key=hmac_key, buffer=bytes_to_deserialize
    )

    return CloudpickleSerializer.deserialize(
        os.path.join(s3_uri, "payload.pkl"), bytes_to_deserialize
    )


def _upload_bytes_to_s3(bytes, s3_uri, s3_kms_key, sagemaker_session):
    """Wrapping s3 uploading with exception translation for remote function."""
    try:
        S3Uploader.upload_bytes(
            bytes, s3_uri, kms_key=s3_kms_key, sagemaker_session=sagemaker_session
        )
    except Exception as e:
        raise ServiceError(
            "Failed to upload serialized bytes to {}: {}".format(s3_uri, repr(e))
        ) from e


def _read_bytes_from_s3(s3_uri, sagemaker_session):
    """Wrapping s3 downloading with exception translation for remote function."""
    try:
        return S3Downloader.read_bytes(s3_uri, sagemaker_session=sagemaker_session)
    except Exception as e:
        raise ServiceError(
            "Failed to read serialized bytes from {}: {}".format(s3_uri, repr(e))
        ) from e


def _compute_hash(buffer: bytes, secret_key: str) -> str:
    """Compute the hmac-sha256 hash"""
    return hmac.new(secret_key.encode(), msg=buffer, digestmod=hashlib.sha256).hexdigest()


def _perform_integrity_check(expected_hash_value: str, secret_key: str, buffer: bytes):
    """Performs integrify checks for serialized code/arguments uploaded to s3.

    Verifies whether the hash read from s3 matches the hash calculated
    during remote function execution.
    """
    actual_hash_value = _compute_hash(buffer=buffer, secret_key=secret_key)
    if not hmac.compare_digest(expected_hash_value, actual_hash_value):
        raise DeserializationError(
            "Integrity check for the serialized function or data failed. "
            "Please restrict access to your S3 bucket"
        )
