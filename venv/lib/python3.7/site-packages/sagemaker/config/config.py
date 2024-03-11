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
"""This module configures the default values defined by the user for SageMaker Python SDK calls.

It supports loading config files from the local file system and Amazon S3.
The schema of the config file is dictated in config_schema.py in the same module.

"""
from __future__ import absolute_import

import pathlib
import os
from typing import List
import boto3
import yaml
import jsonschema
from platformdirs import site_config_dir, user_config_dir
from botocore.utils import merge_dicts
from six.moves.urllib.parse import urlparse
from sagemaker.config.config_schema import SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA
from sagemaker.config.config_utils import get_sagemaker_config_logger

logger = get_sagemaker_config_logger()

_APP_NAME = "sagemaker"
# The default config file location of the Administrator provided config file. This path can be
# overridden with `SAGEMAKER_ADMIN_CONFIG_OVERRIDE` environment variable.
_DEFAULT_ADMIN_CONFIG_FILE_PATH = os.path.join(site_config_dir(_APP_NAME), "config.yaml")
# The default config file location of the user provided config file. This path can be
# overridden with `SAGEMAKER_USER_CONFIG_OVERRIDE` environment variable.
_DEFAULT_USER_CONFIG_FILE_PATH = os.path.join(user_config_dir(_APP_NAME), "config.yaml")

ENV_VARIABLE_ADMIN_CONFIG_OVERRIDE = "SAGEMAKER_ADMIN_CONFIG_OVERRIDE"
ENV_VARIABLE_USER_CONFIG_OVERRIDE = "SAGEMAKER_USER_CONFIG_OVERRIDE"

S3_PREFIX = "s3://"


def load_sagemaker_config(additional_config_paths: List[str] = None, s3_resource=None) -> dict:
    """Loads config files and merges them.

    By default, this method first searches for config files in the default locations
    defined by the SDK.

    Users can override the default admin and user config file paths using the
    ``SAGEMAKER_ADMIN_CONFIG_OVERRIDE`` and ``SAGEMAKER_USER_CONFIG_OVERRIDE`` environment
    variables, respectively.

    Additional config file paths can also be provided as a parameter.

    This method then:
        * Loads each config file, whether it is Amazon S3 or the local file system.
        * Validates the schema of the config files.
        * Merges the files in the same order.

    This method throws exceptions in the following cases:
        * ``jsonschema.exceptions.ValidationError``: Schema validation fails for one or more
          config files.
        * ``RuntimeError``: The method is unable to retrieve the list of all S3 files with the
          same prefix or is unable to retrieve the file.
        * ``ValueError``: There are no S3 files with the prefix when an S3 URI is provided.
        * ``ValueError``: There is no config.yaml file in the S3 bucket when an S3 URI is
          provided.
        * ``ValueError``: A file doesn't exist in a path that was specified by the user as
          part of an environment variable or additional configuration file path. This doesn't
          include the default config file locations.

    Args:
        additional_config_paths: List of config file paths.
            These paths can be one of the following. In the case of a directory, this method
            searches for a ``config.yaml`` file in that directory. This method does not perform a
            recursive search of folders in that directory.

                * Local file path
                * Local directory path
                * S3 URI of the config file
                * S3 URI of the directory containing the config file

            Note: S3 URI follows the format ``s3://<bucket>/<Key prefix>``
        s3_resource (boto3.resource("s3")): The Boto3 S3 resource. This is used to fetch
            config files from S3. If it is not provided but config files are present in S3,
            this method creates a default S3 resource. See `Boto3 Session documentation
            <https://boto3.amazonaws.com/v1/documentation/api\
            /latest/reference/core/session.html#boto3.session.Session.resource>`__.
            This argument is not needed if the config files are present in the local file system.
    """
    default_config_path = os.getenv(
        ENV_VARIABLE_ADMIN_CONFIG_OVERRIDE, _DEFAULT_ADMIN_CONFIG_FILE_PATH
    )
    user_config_path = os.getenv(ENV_VARIABLE_USER_CONFIG_OVERRIDE, _DEFAULT_USER_CONFIG_FILE_PATH)
    config_paths = [default_config_path, user_config_path]
    if additional_config_paths:
        config_paths += additional_config_paths
    config_paths = list(filter(lambda item: item is not None, config_paths))
    merged_config = {}
    for file_path in config_paths:
        config_from_file = {}
        if file_path.startswith(S3_PREFIX):
            config_from_file = _load_config_from_s3(file_path, s3_resource)
        else:
            try:
                config_from_file = _load_config_from_file(file_path)
            except ValueError:
                if file_path not in (
                    _DEFAULT_ADMIN_CONFIG_FILE_PATH,
                    _DEFAULT_USER_CONFIG_FILE_PATH,
                ):
                    # Throw exception only when User provided file path is invalid.
                    # If there are no files in the Default config file locations, don't throw
                    # Exceptions.
                    raise
        if config_from_file:
            validate_sagemaker_config(config_from_file)
            merge_dicts(merged_config, config_from_file)
            logger.info("Fetched defaults config from location: %s", file_path)
        else:
            logger.debug("Fetched defaults config from location: %s, but it was empty", file_path)
    return merged_config


def validate_sagemaker_config(sagemaker_config: dict = None):
    """Validates whether a given dictionary adheres to the schema.

    The schema is defined at
    ``sagemaker.config.config_schema.SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA``.

    Args:
        sagemaker_config: A dictionary containing default values for the
                SageMaker Python SDK. (default: None).
    """
    jsonschema.validate(sagemaker_config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def _load_config_from_file(file_path: str) -> dict:
    """Placeholder docstring"""
    inferred_file_path = file_path
    if os.path.isdir(file_path):
        inferred_file_path = os.path.join(file_path, "config.yaml")
    if not os.path.exists(inferred_file_path):
        raise ValueError(
            f"Unable to load the config file from the location: {file_path}"
            f"Provide a valid file path"
        )
    logger.debug("Fetching defaults config from location: %s", file_path)
    return yaml.safe_load(open(inferred_file_path, "r"))


def _load_config_from_s3(s3_uri, s3_resource_for_config) -> dict:
    """Placeholder docstring"""
    if not s3_resource_for_config:
        # Constructing a default Boto3 S3 Resource from a default Boto3 session.
        boto_session = boto3.DEFAULT_SESSION or boto3.Session()
        boto_region_name = boto_session.region_name
        if boto_region_name is None:
            raise ValueError(
                "Must setup local AWS configuration with a region supported by SageMaker."
            )
        s3_resource_for_config = boto_session.resource("s3", region_name=boto_region_name)

    logger.debug("Fetching defaults config from location: %s", s3_uri)
    inferred_s3_uri = _get_inferred_s3_uri(s3_uri, s3_resource_for_config)
    parsed_url = urlparse(inferred_s3_uri)
    bucket, key_prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
    s3_object = s3_resource_for_config.Object(bucket, key_prefix)
    s3_file_content = s3_object.get()["Body"].read()
    return yaml.safe_load(s3_file_content.decode("utf-8"))


def _get_inferred_s3_uri(s3_uri, s3_resource_for_config):
    """Placeholder docstring"""
    parsed_url = urlparse(s3_uri)
    bucket, key_prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
    s3_bucket = s3_resource_for_config.Bucket(name=bucket)
    s3_objects = s3_bucket.objects.filter(Prefix=key_prefix).all()
    s3_files_with_same_prefix = [
        "{}{}/{}".format(S3_PREFIX, bucket, s3_object.key) for s3_object in s3_objects
    ]
    if len(s3_files_with_same_prefix) == 0:
        # Customer provided us with an incorrect s3 path.
        raise ValueError("Provide a valid S3 path instead of {}".format(s3_uri))
    if len(s3_files_with_same_prefix) > 1:
        # Customer has provided us with a S3 URI which points to a directory
        # search for s3://<bucket>/directory-key-prefix/config.yaml
        inferred_s3_uri = str(pathlib.PurePosixPath(s3_uri, "config.yaml")).replace("s3:/", "s3://")
        if inferred_s3_uri not in s3_files_with_same_prefix:
            # We don't know which file we should be operating with.
            raise ValueError("Provide an S3 URI of a directory that has a config.yaml file.")
        # Customer has a config.yaml present in the directory that was provided as the S3 URI
        return inferred_s3_uri
    return s3_uri
