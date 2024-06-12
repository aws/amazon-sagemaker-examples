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

import contextlib
import copy
import errno
import inspect
import logging
import os
import random
import re
import shutil
import tarfile
import tempfile
import time
from typing import Any, List, Optional
import json
import abc
import uuid
from datetime import datetime

from importlib import import_module
import botocore
from botocore.utils import merge_dicts
from six.moves.urllib import parse

from sagemaker import deprecations
from sagemaker.config import validate_sagemaker_config
from sagemaker.config.config_utils import (
    _log_sagemaker_config_single_substitution,
    _log_sagemaker_config_merge,
)
from sagemaker.session_settings import SessionSettings
from sagemaker.workflow import is_pipeline_variable, is_pipeline_parameter_string

ECR_URI_PATTERN = r"^(\d+)(\.)dkr(\.)ecr(\.)(.+)(\.)(.*)(/)(.*:.*)$"
MAX_BUCKET_PATHS_COUNT = 5
S3_PREFIX = "s3://"
HTTP_PREFIX = "http://"
HTTPS_PREFIX = "https://"
DEFAULT_SLEEP_TIME_SECONDS = 10
WAITING_DOT_NUMBER = 10


logger = logging.getLogger(__name__)


# Use the base name of the image as the job name if the user doesn't give us one
def name_from_image(image, max_length=63):
    """Create a training job name based on the image name and a timestamp.

    Args:
        image (str): Image name.

    Returns:
        str: Training job name using the algorithm from the image name and a
            timestamp.
        max_length (int): Maximum length for the resulting string (default: 63).
    """
    return name_from_base(base_name_from_image(image), max_length=max_length)


def name_from_base(base, max_length=63, short=False):
    """Append a timestamp to the provided string.

    This function assures that the total length of the resulting string is
    not longer than the specified max length, trimming the input parameter if
    necessary.

    Args:
        base (str): String used as prefix to generate the unique name.
        max_length (int): Maximum length for the resulting string (default: 63).
        short (bool): Whether or not to use a truncated timestamp (default: False).

    Returns:
        str: Input parameter with appended timestamp.
    """
    timestamp = sagemaker_short_timestamp() if short else sagemaker_timestamp()
    trimmed_base = base[: max_length - len(timestamp) - 1]
    return "{}-{}".format(trimmed_base, timestamp)


def unique_name_from_base(base, max_length=63):
    """Placeholder Docstring"""
    random.seed(int(uuid.uuid4()))  # using uuid to randomize, otherwise system timestamp is used.
    unique = "%04x" % random.randrange(16**4)  # 4-digit hex
    ts = str(int(time.time()))
    available_length = max_length - 2 - len(ts) - len(unique)
    trimmed = base[:available_length]
    return "{}-{}-{}".format(trimmed, ts, unique)


def base_name_from_image(image, default_base_name=None):
    """Extract the base name of the image to use as the 'algorithm name' for the job.

    Args:
        image (str): Image name.
        default_base_name (str): The default base name

    Returns:
        str: Algorithm name, as extracted from the image name.
    """
    if is_pipeline_variable(image):
        if is_pipeline_parameter_string(image) and image.default_value:
            image_str = image.default_value
        else:
            return default_base_name if default_base_name else "base_name"
    else:
        image_str = image

    m = re.match("^(.+/)?([^:/]+)(:[^:]+)?$", image_str)
    base_name = m.group(2) if m else image_str
    return base_name


def base_from_name(name):
    """Extract the base name of the resource name (for use with future resource name generation).

    This function looks for timestamps that match the ones produced by
    :func:`~sagemaker.utils.name_from_base`.

    Args:
        name (str): The resource name.

    Returns:
        str: The base name, as extracted from the resource name.
    """
    m = re.match(r"^(.+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}|\d{6}-\d{4})", name)
    return m.group(1) if m else name


def sagemaker_timestamp():
    """Return a timestamp with millisecond precision."""
    moment = time.time()
    moment_ms = repr(moment).split(".")[1][:3]
    return time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_ms), time.gmtime(moment))


def sagemaker_short_timestamp():
    """Return a timestamp that is relatively short in length"""
    return time.strftime("%y%m%d-%H%M")


def build_dict(key, value):
    """Return a dict of key and value pair if value is not None, otherwise return an empty dict.

    Args:
        key (str): input key
        value (str): input value

    Returns:
        dict: dict of key and value or an empty dict.
    """
    if value:
        return {key: value}
    return {}


def get_config_value(key_path, config):
    """Placeholder Docstring"""
    if config is None:
        return None

    current_section = config
    for key in key_path.split("."):
        if key in current_section:
            current_section = current_section[key]
        else:
            return None

    return current_section


def get_nested_value(dictionary: dict, nested_keys: List[str]):
    """Returns a nested value from the given dictionary, and None if none present.

    Raises
        ValueError if the dictionary structure does not match the nested_keys
    """
    if (
        dictionary is not None
        and isinstance(dictionary, dict)
        and nested_keys is not None
        and len(nested_keys) > 0
    ):

        current_section = dictionary

        for key in nested_keys[:-1]:
            current_section = current_section.get(key, None)
            if current_section is None:
                # means the full path of nested_keys doesnt exist in the dictionary
                # or the value was set to None
                return None
            if not isinstance(current_section, dict):
                raise ValueError(
                    "Unexpected structure of dictionary.",
                    "Expected value of type dict at key '{}' but got '{}' for dict '{}'".format(
                        key, current_section, dictionary
                    ),
                )
        return current_section.get(nested_keys[-1], None)

    return None


def set_nested_value(dictionary: dict, nested_keys: List[str], value_to_set: object):
    """Sets a nested value in a dictionary.

    This sets a nested value inside the given dictionary and returns the new dictionary. Note: if
    provided an unintended list of nested keys, this can overwrite an unexpected part of the dict.
    Recommended to use after a check with get_nested_value first
    """

    if dictionary is None:
        dictionary = {}

    if (
        dictionary is not None
        and isinstance(dictionary, dict)
        and nested_keys is not None
        and len(nested_keys) > 0
    ):
        current_section = dictionary
        for key in nested_keys[:-1]:
            if (
                key not in current_section
                or current_section[key] is None
                or not isinstance(current_section[key], dict)
            ):
                current_section[key] = {}
            current_section = current_section[key]

        current_section[nested_keys[-1]] = value_to_set
    return dictionary


def get_short_version(framework_version):
    """Return short version in the format of x.x

    Args:
        framework_version: The version string to be shortened.

    Returns:
        str: The short version string
    """
    return ".".join(framework_version.split(".")[:2])


def secondary_training_status_changed(current_job_description, prev_job_description):
    """Returns true if training job's secondary status message has changed.

    Args:
        current_job_description: Current job description, returned from DescribeTrainingJob call.
        prev_job_description: Previous job description, returned from DescribeTrainingJob call.

    Returns:
        boolean: Whether the secondary status message of a training job changed
        or not.
    """
    current_secondary_status_transitions = current_job_description.get("SecondaryStatusTransitions")
    if (
        current_secondary_status_transitions is None
        or len(current_secondary_status_transitions) == 0
    ):
        return False

    prev_job_secondary_status_transitions = (
        prev_job_description.get("SecondaryStatusTransitions")
        if prev_job_description is not None
        else None
    )

    last_message = (
        prev_job_secondary_status_transitions[-1]["StatusMessage"]
        if prev_job_secondary_status_transitions is not None
        and len(prev_job_secondary_status_transitions) > 0
        else ""
    )

    message = current_job_description["SecondaryStatusTransitions"][-1]["StatusMessage"]

    return message != last_message


def secondary_training_status_message(job_description, prev_description):
    """Returns a string contains last modified time and the secondary training job status message.

    Args:
        job_description: Returned response from DescribeTrainingJob call
        prev_description: Previous job description from DescribeTrainingJob call

    Returns:
        str: Job status string to be printed.
    """

    if (
        job_description is None
        or job_description.get("SecondaryStatusTransitions") is None
        or len(job_description.get("SecondaryStatusTransitions")) == 0
    ):
        return ""

    prev_description_secondary_transitions = (
        prev_description.get("SecondaryStatusTransitions") if prev_description is not None else None
    )
    prev_transitions_num = (
        len(prev_description["SecondaryStatusTransitions"])
        if prev_description_secondary_transitions is not None
        else 0
    )
    current_transitions = job_description["SecondaryStatusTransitions"]

    if len(current_transitions) == prev_transitions_num:
        # Secondary status is not changed but the message changed.
        transitions_to_print = current_transitions[-1:]
    else:
        # Secondary status is changed we need to print all the entries.
        transitions_to_print = current_transitions[
            prev_transitions_num - len(current_transitions) :
        ]

    status_strs = []
    for transition in transitions_to_print:
        message = transition["StatusMessage"]
        time_str = datetime.utcfromtimestamp(
            time.mktime(job_description["LastModifiedTime"].timetuple())
        ).strftime("%Y-%m-%d %H:%M:%S")
        status_strs.append("{} {} - {}".format(time_str, transition["Status"], message))

    return "\n".join(status_strs)


def download_folder(bucket_name, prefix, target, sagemaker_session):
    """Download a folder from S3 to a local path

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 prefix within the bucket that will be downloaded. Can
            be a single file.
        target (str): destination path where the downloaded items will be placed
        sagemaker_session (sagemaker.session.Session): a sagemaker session to
            interact with S3.
    """
    boto_session = sagemaker_session.boto_session
    s3 = boto_session.resource("s3", region_name=boto_session.region_name)

    prefix = prefix.lstrip("/")

    # Try to download the prefix as an object first, in case it is a file and not a 'directory'.
    # Do this first, in case the object has broader permissions than the bucket.
    if not prefix.endswith("/"):
        try:
            file_destination = os.path.join(target, os.path.basename(prefix))
            s3.Object(bucket_name, prefix).download_file(file_destination)
            return
        except botocore.exceptions.ClientError as e:
            err_info = e.response["Error"]
            if err_info["Code"] == "404" and err_info["Message"] == "Not Found":
                # S3 also throws this error if the object is a folder,
                # so assume that is the case here, and then raise for an actual 404 later.
                pass
            else:
                raise

    _download_files_under_prefix(bucket_name, prefix, target, s3)


def _download_files_under_prefix(bucket_name, prefix, target, s3):
    """Download all S3 files which match the given prefix

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 prefix within the bucket that will be downloaded
        target (str): destination path where the downloaded items will be placed
        s3 (boto3.resources.base.ServiceResource): S3 resource
    """
    bucket = s3.Bucket(bucket_name)
    for obj_sum in bucket.objects.filter(Prefix=prefix):
        # if obj_sum is a folder object skip it.
        if obj_sum.key.endswith("/"):
            continue
        obj = s3.Object(obj_sum.bucket_name, obj_sum.key)
        s3_relative_path = obj_sum.key[len(prefix) :].lstrip("/")
        file_path = os.path.join(target, s3_relative_path)

        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:
            # EEXIST means the folder already exists, this is safe to skip
            # anything else will be raised.
            if exc.errno != errno.EEXIST:
                raise
        obj.download_file(file_path)


def create_tar_file(source_files, target=None):
    """Create a tar file containing all the source_files

    Args:
        source_files: (List[str]): List of file paths that will be contained in the tar file
        target:

    Returns:
        (str): path to created tar file
    """
    if target:
        filename = target
    else:
        _, filename = tempfile.mkstemp()

    with tarfile.open(filename, mode="w:gz", dereference=True) as t:
        for sf in source_files:
            # Add all files from the directory into the root of the directory structure of the tar
            t.add(sf, arcname=os.path.basename(sf))
    return filename


@contextlib.contextmanager
def _tmpdir(suffix="", prefix="tmp", directory=None):
    """Create a temporary directory with a context manager.

    The file is deleted when the context exits, even when there's an exception.
    The prefix, suffix, and dir arguments are the same as for mkstemp().

    Args:
        suffix (str): If suffix is specified, the file name will end with that
            suffix, otherwise there will be no suffix.
        prefix (str): If prefix is specified, the file name will begin with that
            prefix; otherwise, a default prefix is used.
        directory (str): If a directory is specified, the file will be downloaded
            in this directory; otherwise, a default directory is used.

    Returns:
        str: path to the directory
    """
    if directory is not None and not (os.path.exists(directory) and os.path.isdir(directory)):
        raise ValueError(
            "Inputted directory for storing newly generated temporary "
            f"directory does not exist: '{directory}'"
        )
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=directory)
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp)


def repack_model(
    inference_script,
    source_directory,
    dependencies,
    model_uri,
    repacked_model_uri,
    sagemaker_session,
    kms_key=None,
):
    """Unpack model tarball and creates a new model tarball with the provided code script.

    This function does the following: - uncompresses model tarball from S3 or
    local system into a temp folder - replaces the inference code from the model
    with the new code provided - compresses the new model tarball and saves it
    in S3 or local file system

    Args:
        inference_script (str): path or basename of the inference script that
            will be packed into the model
        source_directory (str): path including all the files that will be packed
            into the model
        dependencies (list[str]): A list of paths to directories (absolute or
            relative) with any additional libraries that will be exported to the
            container (default: []). The library folders will be copied to
            SageMaker in the same folder where the entrypoint is copied.
            Example

                The following call >>> Estimator(entry_point='train.py',
                dependencies=['my/libs/common', 'virtual-env']) results in the
                following inside the container:

                >>> $ ls

                >>> opt/ml/code
                >>>     |------ train.py
                >>>     |------ common
                >>>     |------ virtual-env
        model_uri (str): S3 or file system location of the original model tar
        repacked_model_uri (str): path or file system location where the new
            model will be saved
        sagemaker_session (sagemaker.session.Session): a sagemaker session to
            interact with S3.
        kms_key (str): KMS key ARN for encrypting the repacked model file

    Returns:
        str: path to the new packed model
    """
    dependencies = dependencies or []

    local_download_dir = (
        None
        if sagemaker_session.settings is None
        or sagemaker_session.settings.local_download_dir is None
        else sagemaker_session.settings.local_download_dir
    )
    with _tmpdir(directory=local_download_dir) as tmp:
        model_dir = _extract_model(model_uri, sagemaker_session, tmp)

        _create_or_update_code_dir(
            model_dir,
            inference_script,
            source_directory,
            dependencies,
            sagemaker_session,
            tmp,
        )

        tmp_model_path = os.path.join(tmp, "temp-model.tar.gz")
        with tarfile.open(tmp_model_path, mode="w:gz") as t:
            t.add(model_dir, arcname=os.path.sep)

        _save_model(repacked_model_uri, tmp_model_path, sagemaker_session, kms_key=kms_key)


def _save_model(repacked_model_uri, tmp_model_path, sagemaker_session, kms_key):
    """Placeholder docstring"""
    if repacked_model_uri.lower().startswith("s3://"):
        url = parse.urlparse(repacked_model_uri)
        bucket, key = url.netloc, url.path.lstrip("/")
        new_key = key.replace(os.path.basename(key), os.path.basename(repacked_model_uri))

        settings = (
            sagemaker_session.settings if sagemaker_session is not None else SessionSettings()
        )
        encrypt_artifact = settings.encrypt_repacked_artifacts

        if kms_key:
            extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key}
        elif encrypt_artifact:
            extra_args = {"ServerSideEncryption": "aws:kms"}
        else:
            extra_args = None
        sagemaker_session.boto_session.resource(
            "s3", region_name=sagemaker_session.boto_region_name
        ).Object(bucket, new_key).upload_file(tmp_model_path, ExtraArgs=extra_args)
    else:
        shutil.move(tmp_model_path, repacked_model_uri.replace("file://", ""))


def _create_or_update_code_dir(
    model_dir, inference_script, source_directory, dependencies, sagemaker_session, tmp
):
    """Placeholder docstring"""
    code_dir = os.path.join(model_dir, "code")
    if source_directory and source_directory.lower().startswith("s3://"):
        local_code_path = os.path.join(tmp, "local_code.tar.gz")
        download_file_from_url(source_directory, local_code_path, sagemaker_session)

        with tarfile.open(name=local_code_path, mode="r:gz") as t:
            t.extractall(path=code_dir)

    elif source_directory:
        if os.path.exists(code_dir):
            shutil.rmtree(code_dir)
        shutil.copytree(source_directory, code_dir)
    else:
        if not os.path.exists(code_dir):
            os.mkdir(code_dir)
        try:
            shutil.copy2(inference_script, code_dir)
        except FileNotFoundError:
            if os.path.exists(os.path.join(code_dir, inference_script)):
                pass
            else:
                raise

    for dependency in dependencies:
        lib_dir = os.path.join(code_dir, "lib")
        if os.path.isdir(dependency):
            shutil.copytree(dependency, os.path.join(lib_dir, os.path.basename(dependency)))
        else:
            if not os.path.exists(lib_dir):
                os.mkdir(lib_dir)
            shutil.copy2(dependency, lib_dir)


def _extract_model(model_uri, sagemaker_session, tmp):
    """Placeholder docstring"""
    tmp_model_dir = os.path.join(tmp, "model")
    os.mkdir(tmp_model_dir)
    if model_uri.lower().startswith("s3://"):
        local_model_path = os.path.join(tmp, "tar_file")
        download_file_from_url(model_uri, local_model_path, sagemaker_session)
    else:
        local_model_path = model_uri.replace("file://", "")
    with tarfile.open(name=local_model_path, mode="r:gz") as t:
        t.extractall(path=tmp_model_dir)
    return tmp_model_dir


def download_file_from_url(url, dst, sagemaker_session):
    """Placeholder docstring"""
    url = parse.urlparse(url)
    bucket, key = url.netloc, url.path.lstrip("/")

    download_file(bucket, key, dst, sagemaker_session)


def download_file(bucket_name, path, target, sagemaker_session):
    """Download a Single File from S3 into a local path

    Args:
        bucket_name (str): S3 bucket name
        path (str): file path within the bucket
        target (str): destination directory for the downloaded file.
        sagemaker_session (sagemaker.session.Session): a sagemaker session to
            interact with S3.
    """
    path = path.lstrip("/")
    boto_session = sagemaker_session.boto_session

    s3 = boto_session.resource("s3", region_name=sagemaker_session.boto_region_name)
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(path, target)


def sts_regional_endpoint(region):
    """Get the AWS STS endpoint specific for the given region.

    We need this function because the AWS SDK does not yet honor
    the ``region_name`` parameter when creating an AWS STS client.

    For the list of regional endpoints, see
    https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_enable-regions.html#id_credentials_region-endpoints.

    Args:
        region (str): AWS region name

    Returns:
        str: AWS STS regional endpoint
    """
    endpoint_data = _botocore_resolver().construct_endpoint("sts", region)
    if region == "il-central-1" and not endpoint_data:
        endpoint_data = {"hostname": "sts.{}.amazonaws.com".format(region)}
    return "https://{}".format(endpoint_data["hostname"])


def retries(
    max_retry_count,
    exception_message_prefix,
    seconds_to_sleep=DEFAULT_SLEEP_TIME_SECONDS,
):
    """Retries until max retry count is reached.

    Args:
        max_retry_count (int): The retry count.
        exception_message_prefix (str): The message to include in the exception on failure.
        seconds_to_sleep (int): The number of seconds to sleep between executions.

    """
    for i in range(max_retry_count):
        yield i
        time.sleep(seconds_to_sleep)

    raise Exception(
        "'{}' has reached the maximum retry count of {}".format(
            exception_message_prefix, max_retry_count
        )
    )


def retry_with_backoff(callable_func, num_attempts=8, botocore_client_error_code=None):
    """Retry with backoff until maximum attempts are reached

    Args:
        callable_func (callable): The callable function to retry.
        num_attempts (int): The maximum number of attempts to retry.(Default: 8)
        botocore_client_error_code (str): The specific Botocore ClientError exception error code
            on which to retry on.
            If provided other exceptions will be raised directly w/o retry.
            If not provided, retry on any exception.
            (Default: None)
    """
    if num_attempts < 1:
        raise ValueError(
            "The num_attempts must be >= 1, but the given value is {}.".format(num_attempts)
        )
    for i in range(num_attempts):
        try:
            return callable_func()
        except Exception as ex:  # pylint: disable=broad-except
            if not botocore_client_error_code or (
                botocore_client_error_code
                and isinstance(ex, botocore.exceptions.ClientError)
                and ex.response["Error"]["Code"]  # pylint: disable=no-member
                == botocore_client_error_code
            ):
                if i == num_attempts - 1:
                    raise ex
            else:
                raise ex
            logger.error("Retrying in attempt %s, due to %s", (i + 1), str(ex))
            time.sleep(2**i)


def _botocore_resolver():
    """Get the DNS suffix for the given region.

    Args:
        region (str): AWS region name

    Returns:
        str: the DNS suffix
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data("endpoints"))


def _aws_partition(region):
    """Given a region name (ex: "cn-north-1"), return the corresponding aws partition ("aws-cn").

    Args:
        region (str): The region name for which to return the corresponding partition.
        Ex: "cn-north-1"

    Returns:
        str: partition corresponding to the region name passed in. Ex: "aws-cn"
    """
    endpoint_data = _botocore_resolver().construct_endpoint("sts", region)
    if region == "il-central-1" and not endpoint_data:
        endpoint_data = {"hostname": "sts.{}.amazonaws.com".format(region)}
    return endpoint_data["partition"]


class DeferredError(object):
    """Stores an exception and raises it at a later time if this object is accessed in any way.

    Useful to allow soft-dependencies on imports, so that the ImportError can be raised again
    later if code actually relies on the missing library.

    Example::

        try:
            import obscurelib
        except ImportError as e:
            logger.warning("Failed to import obscurelib. Obscure features will not work.")
            obscurelib = DeferredError(e)
    """

    def __init__(self, exception):
        """Placeholder docstring"""
        self.exc = exception

    def __getattr__(self, name):
        """Called by Python interpreter before using any method or property on the object.

        So this will short-circuit essentially any access to this object.

        Args:
            name:
        """
        raise self.exc


def _module_import_error(py_module, feature, extras):
    """Return error message for module import errors, provide installation details.

    Args:
        py_module (str): Module that failed to be imported
        feature (str): Affected SageMaker feature
        extras (str): Name of the `extras_require` to install the relevant dependencies

    Returns:
        str: Error message with installation instructions.
    """
    error_msg = (
        "Failed to import {}. {} features will be impaired or broken. "
        "Please run \"pip install 'sagemaker[{}]'\" "
        "to install all required dependencies."
    )
    return error_msg.format(py_module, feature, extras)


class DataConfig(abc.ABC):
    """Abstract base class for accessing data config hosted in AWS resources.

    Provides a skeleton for customization by overriding of method fetch_data_config.
    """

    @abc.abstractmethod
    def fetch_data_config(self):
        """Abstract method implementing retrieval of data config from a pre-configured data source.

        Returns:
            object: The data configuration object.
        """


class S3DataConfig(DataConfig):
    """This class extends the DataConfig class to fetch a data config file hosted on S3"""

    def __init__(
        self,
        sagemaker_session,
        bucket_name,
        prefix,
    ):
        """Initialize a ``S3DataConfig`` instance.

        Args:
            sagemaker_session (Session): SageMaker session instance to use for boto configuration.
            bucket_name (str): Required. Bucket name from which data config needs to be fetched.
            prefix (str): Required. The object prefix for the hosted data config.

        """
        if bucket_name is None or prefix is None:
            raise ValueError(
                "Bucket Name and S3 file Prefix are required arguments and must be provided."
            )

        super(S3DataConfig, self).__init__()

        self.bucket_name = bucket_name
        self.prefix = prefix
        self.sagemaker_session = sagemaker_session

    def fetch_data_config(self):
        """Fetches data configuration from a S3 bucket.

        Returns:
            object: The JSON object containing data configuration.
        """

        json_string = self.sagemaker_session.read_s3_file(self.bucket_name, self.prefix)
        return json.loads(json_string)

    def get_data_bucket(self, region_requested=None):
        """Provides the bucket containing the data for specified region.

        Args:
            region_requested (str): The region for which the data is beig requested.

        Returns:
            str: Name of the S3 bucket containing datasets in the requested region.
        """

        config = self.fetch_data_config()
        region = region_requested if region_requested else self.sagemaker_session.boto_region_name
        return config[region] if region in config.keys() else config["default"]


get_ecr_image_uri_prefix = deprecations.removed_function("get_ecr_image_uri_prefix")


def update_container_with_inference_params(
    framework=None,
    framework_version=None,
    nearest_model_name=None,
    data_input_configuration=None,
    container_def=None,
    container_list=None,
):
    """Function to check if inference recommender parameters exist and update container.

    Args:
        framework (str): Machine learning framework of the model package container image
                (default: None).
        framework_version (str): Framework version of the Model Package Container Image
            (default: None).
        nearest_model_name (str): Name of a pre-trained machine learning benchmarked by
            Amazon SageMaker Inference Recommender (default: None).
        data_input_configuration (str): Input object for the model (default: None).
        container_def (dict): object to be updated.
        container_list (list): list to be updated.

    Returns:
        dict: dict with inference recommender params
    """

    if container_list is not None:
        for obj in container_list:
            construct_container_object(
                obj, data_input_configuration, framework, framework_version, nearest_model_name
            )

    if container_def is not None:
        construct_container_object(
            container_def,
            data_input_configuration,
            framework,
            framework_version,
            nearest_model_name,
        )

    return container_list or container_def


def construct_container_object(
    obj, data_input_configuration, framework, framework_version, nearest_model_name
):
    """Function to construct container object.

    Args:
        framework (str): Machine learning framework of the model package container image
                (default: None).
        framework_version (str): Framework version of the Model Package Container Image
            (default: None).
        nearest_model_name (str): Name of a pre-trained machine learning benchmarked by
            Amazon SageMaker Inference Recommender (default: None).
        data_input_configuration (str): Input object for the model (default: None).
        obj (dict): object to be updated.

    Returns:
        dict: container object
    """

    if framework is not None:
        obj.update(
            {
                "Framework": framework,
            }
        )

    if framework_version is not None:
        obj.update(
            {
                "FrameworkVersion": framework_version,
            }
        )

    if nearest_model_name is not None:
        obj.update(
            {
                "NearestModelName": nearest_model_name,
            }
        )

    if data_input_configuration is not None:
        obj.update(
            {
                "ModelInput": {
                    "DataInputConfig": data_input_configuration,
                },
            }
        )

    return obj


def pop_out_unused_kwarg(arg_name: str, kwargs: dict, override_val: Optional[str] = None):
    """Pop out the unused key-word argument and give a warning.

    Args:
        arg_name (str): The name of the argument to be checked if it is unused.
        kwargs (dict): The key-word argument dict.
        override_val (str): The value used to override the unused argument (default: None).
    """
    if arg_name not in kwargs:
        return
    warn_msg = "{} supplied in kwargs will be ignored".format(arg_name)
    if override_val:
        warn_msg += " and further overridden with {}.".format(override_val)
    logging.warning(warn_msg)
    kwargs.pop(arg_name)


def to_string(obj: object):
    """Convert an object to string

    This helper function handles converting PipelineVariable object to string as well

    Args:
        obj (object): The object to be converted
    """
    return obj.to_string() if is_pipeline_variable(obj) else str(obj)


def _start_waiting(waiting_time: int):
    """Waiting and print the in progress animation to stdout.

    Args:
        waiting_time (int): The total waiting time.
    """
    interval = float(waiting_time) / WAITING_DOT_NUMBER

    progress = ""
    for _ in range(WAITING_DOT_NUMBER):
        progress += "."
        print(progress, end="\r")
        time.sleep(interval)
    print(len(progress) * " ", end="\r")


def get_module(module_name):
    """Import a module.

    Args:
        module_name (str): name of the module to import.

    Returns:
        object: The imported module.

    Raises:
        Exception: when the module name is not found
    """
    try:
        return import_module(module_name)
    except ImportError:
        raise Exception("Cannot import module {}, please try again.".format(module_name))


def check_and_get_run_experiment_config(experiment_config: Optional[dict] = None) -> dict:
    """Check user input experiment_config or get it from the current Run object if exists.

    Args:
        experiment_config (dict): The experiment_config supplied by the user.

    Returns:
        dict: Return the user supplied experiment_config if it is not None.
            Otherwise fetch the experiment_config from the current Run object if exists.
    """
    from sagemaker.experiments._run_context import _RunContext

    run_obj = _RunContext.get_current_run()
    if experiment_config:
        if run_obj:
            logger.warning(
                "The function is invoked within an Experiment Run context "
                "but another experiment_config (%s) was supplied, so "
                "ignoring the experiment_config fetched from the Run object.",
                experiment_config,
            )
        return experiment_config

    return run_obj.experiment_config if run_obj else None


def resolve_value_from_config(
    direct_input=None,
    config_path: str = None,
    default_value=None,
    sagemaker_session=None,
    sagemaker_config: dict = None,
):
    """Decides which value for the caller to use.

    Note: This method incorporates information from the sagemaker config.

    Uses this order of prioritization:
    1. direct_input
    2. config value
    3. default_value
    4. None

    Args:
        direct_input: The value that the caller of this method starts with. Usually this is an
            input to the caller's class or method.
        config_path (str): A string denoting the path used to lookup the value in the
            sagemaker config.
        default_value: The value used if not present elsewhere.
        sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
            SageMaker interactions (default: None).
        sagemaker_config (dict): The sdk defaults config that is normally accessed through a
            Session object by doing `session.sagemaker_config`. (default: None) This parameter will
            be checked for the config value if (and only if) sagemaker_session is None. This
            parameter exists for the rare cases where the user provided no Session but a default
            Session cannot be initialized before config injection is needed. In that case,
            the config dictionary may be loaded and passed here before a default Session object
            is created.

    Returns:
        The value that should be used by the caller
    """

    config_value = (
        get_sagemaker_config_value(
            sagemaker_session, config_path, sagemaker_config=sagemaker_config
        )
        if config_path
        else None
    )
    _log_sagemaker_config_single_substitution(direct_input, config_value, config_path)

    if direct_input is not None:
        return direct_input

    if config_value is not None:
        return config_value

    return default_value


def get_sagemaker_config_value(sagemaker_session, key, sagemaker_config: dict = None):
    """Returns the value that corresponds to the provided key from the configuration file.

    Args:
        key: Key Path of the config file entry.
        sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
            SageMaker interactions.
        sagemaker_config (dict): The sdk defaults config that is normally accessed through a
            Session object by doing `session.sagemaker_config`. (default: None) This parameter will
            be checked for the config value if (and only if) sagemaker_session is None. This
            parameter exists for the rare cases where no Session provided but a default Session
            cannot be initialized before config injection is needed. In that case, the config
            dictionary may be loaded and passed here before a default Session object is created.

    Returns:
        object: The corresponding default value in the configuration file.
    """
    if sagemaker_session:
        config_to_check = sagemaker_session.sagemaker_config
    else:
        config_to_check = sagemaker_config

    if not config_to_check:
        return None

    validate_sagemaker_config(config_to_check)
    config_value = get_config_value(key, config_to_check)
    # Copy the value so any modifications to the output will not modify the source config
    return copy.deepcopy(config_value)


def resolve_class_attribute_from_config(
    clazz: Optional[type],
    instance: Optional[object],
    attribute: str,
    config_path: str,
    default_value=None,
    sagemaker_session=None,
):
    """Utility method that merges config values to data classes.

    Takes an instance of a class and, if not already set, sets the instance's attribute to a
    value fetched from the sagemaker_config or the default_value.

    Uses this order of prioritization to determine what the value of the attribute should be:
    1. current value of attribute
    2. config value
    3. default_value
    4. does not set it

    Args:
        clazz (Optional[type]): Class of 'instance'. Used to generate a new instance if the
               instance is None. If None is provided here, no new object will be created
               if 'instance' doesnt exist. Note: if provided, the constructor should set default
               values to None; Otherwise, the constructor's non-None default will be left
               as-is even if a config value was defined.
        instance (Optional[object]): instance of the Class 'clazz' that has an attribute
                 of 'attribute' to set
        attribute (str): attribute of the instance to set if not already set
        config_path (str): a string denoting the path to use to lookup the config value in the
                           sagemaker config
        default_value: the value to use if not present elsewhere
        sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
                SageMaker interactions (default: None).

    Returns:
        The updated class instance that should be used by the caller instead of the
        'instance' parameter that was passed in.
    """
    config_value = get_sagemaker_config_value(sagemaker_session, config_path)

    if config_value is None and default_value is None:
        # return instance unmodified. Could be None or populated
        return instance

    if instance is None:
        if clazz is None or not inspect.isclass(clazz):
            return instance
        # construct a new instance if the instance does not exist
        instance = clazz()

    if not hasattr(instance, attribute):
        raise TypeError(
            "Unexpected structure of object.",
            "Expected attribute {} to be present inside instance {} of class {}".format(
                attribute, instance, clazz
            ),
        )

    current_value = getattr(instance, attribute)
    if current_value is None:
        # only set value if object does not already have a value set
        if config_value is not None:
            setattr(instance, attribute, config_value)
        elif default_value is not None:
            setattr(instance, attribute, default_value)

    _log_sagemaker_config_single_substitution(current_value, config_value, config_path)

    return instance


def resolve_nested_dict_value_from_config(
    dictionary: dict,
    nested_keys: List[str],
    config_path: str,
    default_value: object = None,
    sagemaker_session=None,
):
    """Utility method that sets the value of a key path in a nested dictionary .

    This method takes a dictionary and, if not already set, sets the value for the provided
    list of nested keys to the value fetched from the sagemaker_config or the default_value.

    Uses this order of prioritization to determine what the value of the attribute should be:
    (1) current value of nested key, (2) config value, (3) default_value, (4) does not set it

    Args:
        dictionary: The dict to update.
        nested_keys: The paths of keys where the value should be checked and set if needed.
        config_path (str): A string denoting the path used to find the config value in the
        sagemaker config.
        default_value: The value to use if not present elsewhere.
        sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
            SageMaker interactions (default: None).

    Returns:
        The updated dictionary that should be used by the caller instead of the
        'dictionary' parameter that was passed in.
    """
    config_value = get_sagemaker_config_value(sagemaker_session, config_path)

    if config_value is None and default_value is None:
        # if there is nothing to set, return early. And there is no need to traverse through
        # the dictionary or add nested dicts to it
        return dictionary

    try:
        current_nested_value = get_nested_value(dictionary, nested_keys)
    except ValueError as e:
        logging.error("Failed to check dictionary for applying sagemaker config: %s", e)
        return dictionary

    if current_nested_value is None:
        # only set value if not already set
        if config_value is not None:
            dictionary = set_nested_value(dictionary, nested_keys, config_value)
        elif default_value is not None:
            dictionary = set_nested_value(dictionary, nested_keys, default_value)

    _log_sagemaker_config_single_substitution(current_nested_value, config_value, config_path)

    return dictionary


def update_list_of_dicts_with_values_from_config(
    input_list,
    config_key_path,
    required_key_paths: List[str] = None,
    union_key_paths: List[List[str]] = None,
    sagemaker_session=None,
):
    """Updates a list of dictionaries with missing values that are present in Config.

    In some cases, config file might introduce new parameters which requires certain other
    parameters to be provided as part of the input list. Without those parameters, the underlying
    service will throw an exception. This method provides the capability to specify required key
    paths.

    In some other cases, config file might introduce new parameters but the service API requires
    either an existing parameter or the new parameter that was supplied by config but not both

    Args:
        input_list: The input list that was provided as a method parameter.
        config_key_path: The Key Path in the Config file that corresponds to the input_list
        parameter.
        required_key_paths (List[str]): List of required key paths that should be verified in the
        merged output. If a required key path is missing, we will not perform the merge for that
        item.
        union_key_paths (List[List[str]]): List of List of Key paths for which we need to verify
        whether exactly zero/one of the parameters exist.
        For example: If the resultant dictionary can have either 'X1' or 'X2' as parameter or
        neither but not both, then pass [['X1', 'X2']]
        sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
            SageMaker interactions (default: None).

    Returns:
        No output. In place merge happens.
    """
    if not input_list:
        return
    inputs_copy = copy.deepcopy(input_list)
    inputs_from_config = get_sagemaker_config_value(sagemaker_session, config_key_path) or []
    unmodified_inputs_from_config = copy.deepcopy(inputs_from_config)

    for i in range(min(len(input_list), len(inputs_from_config))):
        dict_from_inputs = input_list[i]
        dict_from_config = inputs_from_config[i]
        merge_dicts(dict_from_config, dict_from_inputs)
        # Check if required key paths are present in merged dict (dict_from_config)
        required_key_path_check_passed = _validate_required_paths_in_a_dict(
            dict_from_config, required_key_paths
        )
        if not required_key_path_check_passed:
            # Don't do the merge, config is introducing a new parameter which needs a
            # corresponding required parameter.
            continue
        union_key_path_check_passed = _validate_union_key_paths_in_a_dict(
            dict_from_config, union_key_paths
        )
        if not union_key_path_check_passed:
            # Don't do the merge, Union parameters are not obeyed.
            continue
        input_list[i] = dict_from_config

    _log_sagemaker_config_merge(
        source_value=inputs_copy,
        config_value=unmodified_inputs_from_config,
        merged_source_and_config_value=input_list,
        config_key_path=config_key_path,
    )


def _validate_required_paths_in_a_dict(source_dict, required_key_paths: List[str] = None) -> bool:
    """Placeholder docstring"""
    if not required_key_paths:
        return True
    for required_key_path in required_key_paths:
        if get_config_value(required_key_path, source_dict) is None:
            return False
    return True


def _validate_union_key_paths_in_a_dict(
    source_dict, union_key_paths: List[List[str]] = None
) -> bool:
    """Placeholder docstring"""
    if not union_key_paths:
        return True
    for union_key_path in union_key_paths:
        union_parameter_present = False
        for key_path in union_key_path:
            if get_config_value(key_path, source_dict):
                if union_parameter_present:
                    return False
                union_parameter_present = True
    return True


def update_nested_dictionary_with_values_from_config(
    source_dict, config_key_path, sagemaker_session=None
) -> dict:
    """Updates a nested dictionary with missing values that are present in Config.

    Args:
        source_dict: The input nested dictionary that was provided as method parameter.
        config_key_path: The Key Path in the Config file which corresponds to this
        source_dict parameter.
        sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
            SageMaker interactions (default: None).

    Returns:
        dict: The merged nested dictionary that is updated with missing values that are present
        in the Config file.
    """
    inferred_config_dict = get_sagemaker_config_value(sagemaker_session, config_key_path) or {}
    original_config_dict_value = copy.deepcopy(inferred_config_dict)
    merge_dicts(inferred_config_dict, source_dict or {})

    if original_config_dict_value == {}:
        # The config value is empty. That means either
        # (1) inferred_config_dict equals source_dict, or
        # (2) if source_dict was None, inferred_config_dict equals {}
        # We should return whatever source_dict was to be safe. Because if for example,
        # a VpcConfig is set to {} instead of None, some boto calls will fail due to
        # ParamValidationError (because a VpcConfig was specified but required parameters for
        # the VpcConfig were missing.)

        # Don't need to print because no config value was used or defined
        return source_dict

    _log_sagemaker_config_merge(
        source_value=source_dict,
        config_value=original_config_dict_value,
        merged_source_and_config_value=inferred_config_dict,
        config_key_path=config_key_path,
    )

    return inferred_config_dict


def stringify_object(obj: Any) -> str:
    """Returns string representation of object, returning only non-None fields."""
    non_none_atts = {key: value for key, value in obj.__dict__.items() if value is not None}
    return f"{type(obj).__name__}: {str(non_none_atts)}"


def volume_size_supported(instance_type: str) -> bool:
    """Returns True if SageMaker allows volume_size to be used for the instance type.

    Raises:
        ValueError: If the instance type is improperly formatted.
    """

    try:

        # local mode does not support volume size
        # instance type given as pipeline parameter does not support volume size
        # do not change the if statement order below.
        if is_pipeline_variable(instance_type) or instance_type.startswith("local"):
            return False

        parts: List[str] = instance_type.split(".")

        if len(parts) == 3 and parts[0] == "ml":
            parts = parts[1:]

        if len(parts) != 2:
            raise ValueError(f"Failed to parse instance type '{instance_type}'")

        # Any instance type with a "d" in the instance family (i.e. c5d, p4d, etc) + g5
        # does not support attaching an EBS volume.
        family = parts[0]
        return "d" not in family and not family.startswith("g5")
    except Exception as e:
        raise ValueError(f"Failed to parse instance type '{instance_type}': {str(e)}")


def instance_supports_kms(instance_type: str) -> bool:
    """Returns True if SageMaker allows KMS keys to be attached to the instance.

    Raises:
        ValueError: If the instance type is improperly formatted.
    """
    return volume_size_supported(instance_type)
