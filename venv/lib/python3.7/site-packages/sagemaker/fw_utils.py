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
"""Utility methods used by framework classes"""
from __future__ import absolute_import

import json
import logging
import os
import re
import time
import shutil
import tempfile
from collections import namedtuple
from typing import Optional, Union, Dict
from packaging import version

import sagemaker.image_uris
from sagemaker.s3_utils import s3_path_join
from sagemaker.session_settings import SessionSettings
import sagemaker.utils
from sagemaker.workflow import is_pipeline_variable

from sagemaker.deprecations import renamed_warning, renamed_kwargs
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.deprecations import deprecation_warn_base

logger = logging.getLogger(__name__)

_TAR_SOURCE_FILENAME = "source.tar.gz"

UploadedCode = namedtuple("UploadedCode", ["s3_prefix", "script_name"])
"""sagemaker.fw_utils.UploadedCode: An object containing the S3 prefix and script name.
This is for the source code used for the entry point with an ``Estimator``. It can be
instantiated with positional or keyword arguments.
"""

PYTHON_2_DEPRECATION_WARNING = (
    "{latest_supported_version} is the latest version of {framework} that supports "
    "Python 2. Newer versions of {framework} will only be available for Python 3."
    "Please set the argument \"py_version='py3'\" to use the Python 3 {framework} image."
)
PARAMETER_SERVER_MULTI_GPU_WARNING = (
    "If you have selected a multi-GPU training instance type "
    "and also enabled parameter server for distributed training, "
    "distributed training with the default parameter server configuration will not "
    "fully leverage all GPU cores; the parameter server will be configured to run "
    "only one worker per host regardless of the number of GPUs."
)

DEBUGGER_UNSUPPORTED_REGIONS = (
    "us-iso-east-1",
    "us-isob-east-1",
    "ap-southeast-3",
    "ap-southeast-4",
    "eu-south-2",
    "me-central-1",
    "ap-south-2",
    "eu-central-2",
    "us-gov-east-1",
)
PROFILER_UNSUPPORTED_REGIONS = (
    "us-iso-east-1",
    "us-isob-east-1",
    "ap-southeast-3",
    "ap-southeast-4",
    "eu-south-2",
    "me-central-1",
    "ap-south-2",
    "eu-central-2",
    "us-gov-east-1",
)

SINGLE_GPU_INSTANCE_TYPES = ("ml.p2.xlarge", "ml.p3.2xlarge")
SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES = (
    "ml.p3.16xlarge",
    "ml.p3dn.24xlarge",
    "ml.p4d.24xlarge",
    "ml.p4de.24xlarge",
    "local_gpu",
)
SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSIONS = {
    # tf 2.12 should not be supported: smdataparallel excludes support for tf 2.12.
    "tensorflow": [
        "2.3",
        "2.3.1",
        "2.3.2",
        "2.4",
        "2.4.1",
        "2.4.3",
        "2.5",
        "2.5.0",
        "2.5.1",
        "2.6",
        "2.6.0",
        "2.6.2",
        "2.6.3",
        "2.7",
        "2.7.1",
        "2.8",
        "2.8.0",
        "2.9",
        "2.9.1",
        "2.9.2",
        "2.10",
        "2.10.1",
        "2.11",
        "2.11.0",
    ],
    "pytorch": [
        "1.6",
        "1.6.0",
        "1.7",
        "1.7.1",
        "1.8",
        "1.8.0",
        "1.8.1",
        "1.9",
        "1.9.0",
        "1.9.1",
        "1.10",
        "1.10.0",
        "1.10.2",
        "1.11",
        "1.11.0",
        "1.12",
        "1.12.0",
        "1.12.1",
        "1.13.1",
        "2.0.0",
    ],
}

PYTORCHDDP_SUPPORTED_FRAMEWORK_VERSIONS = [
    "1.10",
    "1.10.0",
    "1.10.2",
    "1.11",
    "1.11.0",
    "1.12",
    "1.12.0",
    "1.12.1",
    "1.13.1",
    "2.0.0",
]


TORCH_DISTRIBUTED_GPU_SUPPORTED_FRAMEWORK_VERSIONS = ["1.13.1", "2.0.0"]

TRAINIUM_SUPPORTED_DISTRIBUTION_STRATEGIES = ["torch_distributed"]
TRAINIUM_SUPPORTED_TORCH_DISTRIBUTED_FRAMEWORK_VERSIONS = [
    "1.11",
    "1.11.0",
    "1.12",
    "1.12.0",
    "1.12.1",
    "1.13.1",
    "2.0.0",
]

SMDISTRIBUTED_SUPPORTED_STRATEGIES = ["dataparallel", "modelparallel"]


GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY = [
    "m6g",
    "m6gd",
    "c6g",
    "c6gd",
    "c6gn",
    "c7g",
    "r6g",
    "r6gd",
]


GRAVITON_ALLOWED_FRAMEWORKS = set(["tensorflow", "pytorch", "xgboost", "sklearn"])


def validate_source_dir(script, directory):
    """Validate that the source directory exists and it contains the user script.

    Args:
        script (str): Script filename.
        directory (str): Directory containing the source file.
    Raises:
        ValueError: If ``directory`` does not exist, is not a directory, or does
            not contain ``script``.
    """
    if directory:
        if not os.path.isfile(os.path.join(directory, script)):
            raise ValueError(
                'No file named "{}" was found in directory "{}".'.format(script, directory)
            )

    return True


def validate_source_code_input_against_pipeline_variables(
    entry_point: Optional[Union[str, PipelineVariable]] = None,
    source_dir: Optional[Union[str, PipelineVariable]] = None,
    git_config: Optional[Dict[str, str]] = None,
    enable_network_isolation: Union[bool, PipelineVariable] = False,
):
    """Validate source code input against pipeline variables

    Args:
        entry_point (str or PipelineVariable): The path to the local Python source file that
            should be executed as the entry point to training (default: None).
        source_dir (str or PipelineVariable): The Path to a directory with any other
            training source code dependencies aside from the entry point file (default: None).
        git_config (Dict[str, str]): Git configurations used for cloning files (default: None).
        enable_network_isolation (bool or PipelineVariable): Specifies whether container will run
            in network isolation mode (default: False).
    """
    if is_pipeline_variable(enable_network_isolation) or enable_network_isolation is True:
        if is_pipeline_variable(entry_point) or is_pipeline_variable(source_dir):
            raise TypeError(
                "entry_point, source_dir should not be pipeline variables "
                "when enable_network_isolation is a pipeline variable or it is set to True."
            )
    if git_config:
        if is_pipeline_variable(entry_point) or is_pipeline_variable(source_dir):
            raise TypeError(
                "entry_point, source_dir should not be pipeline variables when git_config is given."
            )
    if is_pipeline_variable(entry_point):
        if not source_dir:
            raise TypeError(
                "The entry_point should not be a pipeline variable when source_dir is missing."
            )
        if not is_pipeline_variable(source_dir) and not source_dir.lower().startswith("s3://"):
            raise TypeError(
                "The entry_point should not be a pipeline variable when source_dir is a local path."
            )
        logger.warning(
            "The entry_point is a pipeline variable: %s. During pipeline execution, "
            "the interpreted value of entry_point has to be a local path in the container "
            "pointing to a Python source file which is located at the root of source_dir.",
            type(entry_point),
        )
    if is_pipeline_variable(source_dir):
        logger.warning(
            "The source_dir is a pipeline variable: %s. During pipeline execution, "
            "the interpreted value of source_dir has to be an S3 URI and "
            "must point to a tar.gz file",
            type(source_dir),
        )


def parse_mp_parameters(params):
    """Parse the model parallelism parameters provided by the user.

    Args:
        params: a string representing path to an existing config, or
                a config dict.

    Returns:
        parsed: a dict of parsed config.

    Raises:
        ValueError: if params is not a string or a dict, or
                    the config file cannot be parsed as json.
    """
    parsed = None
    if isinstance(params, dict):
        parsed = params
    elif os.path.exists(params):
        try:
            with open(params, "r") as fp:
                parsed = json.load(fp)
        except json.decoder.JSONDecodeError:
            pass
    else:
        raise ValueError(
            f"Expected a string path to an existing modelparallel config, or a dictionary. "
            f"Received: {params}."
        )

    if parsed is None:
        raise ValueError(f"Cannot parse {params} as a json file.")

    return parsed


def get_mp_parameters(distribution):
    """Get the model parallelism parameters provided by the user.

    Args:
        distribution: distribution dictionary defined by the user.

    Returns:
        params: dictionary containing model parallelism parameters
        used for training.
    """
    try:
        mp_dict = distribution["smdistributed"]["modelparallel"]
    except KeyError:
        mp_dict = {}
    if mp_dict.get("enabled", False) is True:
        params = mp_dict.get("parameters", {})
        params = parse_mp_parameters(params)
        validate_mp_config(params)
        return params
    return None


def validate_mp_config(config):
    """Validate the configuration dictionary for model parallelism.

    Args:
       config (dict): Dictionary holding configuration keys and values.

    Raises:
        ValueError: If any of the keys have incorrect values.
    """

    if "partitions" not in config:
        raise ValueError("'partitions' is a required parameter.")

    def validate_positive(key):
        try:
            if not isinstance(config[key], int) or config[key] < 1:
                raise ValueError(f"The number of {key} must be a positive integer.")
        except KeyError:
            pass

    def validate_in(key, vals):
        try:
            if config[key] not in vals:
                raise ValueError(f"{key} must be a value in: {vals}.")
        except KeyError:
            pass

    def validate_bool(keys):
        validate_in(keys, [True, False])

    validate_in("pipeline", ["simple", "interleaved", "_only_forward"])
    validate_in("placement_strategy", ["spread", "cluster"])
    validate_in("optimize", ["speed", "memory"])

    for key in ["microbatches", "partitions", "active_microbatches"]:
        validate_positive(key)

    for key in [
        "auto_partition",
        "contiguous",
        "load_partition",
        "horovod",
        "ddp",
        "deterministic_server",
    ]:
        validate_bool(key)

    if "partition_file" in config and not isinstance(config.get("partition_file"), str):
        raise ValueError("'partition_file' must be a str.")

    if config.get("auto_partition") is False and "default_partition" not in config:
        raise ValueError("default_partition must be supplied if auto_partition is set to False!")

    if "default_partition" in config and config["default_partition"] >= config["partitions"]:
        raise ValueError("default_partition must be less than the number of partitions!")

    if "memory_weight" in config and (
        config["memory_weight"] > 1.0 or config["memory_weight"] < 0.0
    ):
        raise ValueError("memory_weight must be between 0.0 and 1.0!")

    if "ddp_port" in config and "ddp" not in config:
        raise ValueError("`ddp_port` needs `ddp` to be set as well")

    if "ddp_dist_backend" in config and "ddp" not in config:
        raise ValueError("`ddp_dist_backend` needs `ddp` to be set as well")

    if "ddp_port" in config:
        if not isinstance(config["ddp_port"], int) or config["ddp_port"] < 0:
            value = config["ddp_port"]
            raise ValueError(f"Invalid port number {value}.")

    if config.get("horovod", False) and config.get("ddp", False):
        raise ValueError("'ddp' and 'horovod' cannot be simultaneously enabled.")


def tar_and_upload_dir(
    session,
    bucket,
    s3_key_prefix,
    script,
    directory=None,
    dependencies=None,
    kms_key=None,
    s3_resource=None,
    settings: Optional[SessionSettings] = None,
) -> UploadedCode:
    """Package source files and upload a compress tar file to S3.

    The S3 location will be ``s3://<bucket>/s3_key_prefix/sourcedir.tar.gz``.
    If directory is an S3 URI, an UploadedCode object will be returned, but
    nothing will be uploaded to S3 (this allow reuse of code already in S3).
    If directory is None, the script will be added to the archive at
    ``./<basename of script>``. If directory is not None, the (recursive) contents
    of the directory will be added to the archive. directory is treated as the base
    path of the archive, and the script name is assumed to be a filename or relative path
    inside the directory.

    Args:
        session (boto3.Session): Boto session used to access S3.
        bucket (str): S3 bucket to which the compressed file is uploaded.
        s3_key_prefix (str): Prefix for the S3 key.
        script (str): Script filename or path.
        directory (str): Optional. Directory containing the source file. If it
            starts with "s3://", no action is taken.
        dependencies (List[str]): Optional. A list of paths to directories
            (absolute or relative) containing additional libraries that will be
            copied into /opt/ml/lib
        kms_key (str): Optional. KMS key ID used to upload objects to the bucket
            (default: None).
        s3_resource (boto3.resource("s3")): Optional. Pre-instantiated Boto3 Resource
            for S3 connections, can be used to customize the configuration,
            e.g. set the endpoint URL (default: None).
        settings (sagemaker.session_settings.SessionSettings): Optional. The settings
            of the SageMaker ``Session``, can be used to override the default encryption
            behavior (default: None).
    Returns:
        sagemaker.fw_utils.UploadedCode: An object with the S3 bucket and key (S3 prefix) and
            script name.
    """
    if directory and (is_pipeline_variable(directory) or directory.lower().startswith("s3://")):
        return UploadedCode(s3_prefix=directory, script_name=script)

    script_name = script if directory else os.path.basename(script)
    dependencies = dependencies or []
    key = "%s/sourcedir.tar.gz" % s3_key_prefix
    if (
        settings is not None
        and settings.local_download_dir is not None
        and not (
            os.path.exists(settings.local_download_dir)
            and os.path.isdir(settings.local_download_dir)
        )
    ):
        raise ValueError(
            "Inputted directory for storing newly generated temporary directory does "
            f"not exist: '{settings.local_download_dir}'"
        )
    local_download_dir = None if settings is None else settings.local_download_dir
    tmp = tempfile.mkdtemp(dir=local_download_dir)
    encrypt_artifact = True if settings is None else settings.encrypt_repacked_artifacts

    try:
        source_files = _list_files_to_compress(script, directory) + dependencies
        tar_file = sagemaker.utils.create_tar_file(
            source_files, os.path.join(tmp, _TAR_SOURCE_FILENAME)
        )

        if kms_key:
            extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key}
        elif encrypt_artifact:
            # encrypt the tarball at rest in S3 with the default AWS managed KMS key for S3
            # see https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html#API_PutObject_RequestSyntax
            extra_args = {"ServerSideEncryption": "aws:kms"}
        else:
            extra_args = None

        if s3_resource is None:
            s3_resource = session.resource("s3", region_name=session.region_name)
        else:
            print("Using provided s3_resource")

        s3_resource.Object(bucket, key).upload_file(tar_file, ExtraArgs=extra_args)
    finally:
        shutil.rmtree(tmp)

    return UploadedCode(s3_prefix="s3://%s/%s" % (bucket, key), script_name=script_name)


def _list_files_to_compress(script, directory):
    """Placeholder docstring"""
    if directory is None:
        return [script]

    basedir = directory if directory else os.path.dirname(script)
    return [os.path.join(basedir, name) for name in os.listdir(basedir)]


def framework_name_from_image(image_uri):
    # noinspection LongLine
    """Extract the framework and Python version from the image name.

    Args:
        image_uri (str): Image URI, which should be one of the following forms:
            legacy:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<container_version>'
            legacy:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<fw_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>:<fw_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-<fw>:<rl_toolkit><rl_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/<fw>-<image_scope>:<fw_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-xgboost:<fw_version>-<container_version>'

    Returns:
        tuple: A tuple containing:

            - str: The framework name
            - str: The Python version
            - str: The image tag
            - str: If the TensorFlow image is script mode
    """
    sagemaker_pattern = re.compile(sagemaker.utils.ECR_URI_PATTERN)
    sagemaker_match = sagemaker_pattern.match(image_uri)
    if sagemaker_match is None:
        return None, None, None, None

    # extract framework, python version and image tag
    # We must support both the legacy and current image name format.
    name_pattern = re.compile(
        r"""^(?:sagemaker(?:-rl)?-)?
        (tensorflow|mxnet|chainer|pytorch|pytorch-trcomp|scikit-learn|xgboost
        |huggingface-tensorflow|huggingface-pytorch
        |huggingface-tensorflow-trcomp|huggingface-pytorch-trcomp)(?:-)?
        (scriptmode|training)?
        :(.*)-(.*?)-(py2|py3\d*)(?:.*)$""",
        re.VERBOSE,
    )
    name_match = name_pattern.match(sagemaker_match.group(9))
    if name_match is not None:
        fw, scriptmode, ver, device, py = (
            name_match.group(1),
            name_match.group(2),
            name_match.group(3),
            name_match.group(4),
            name_match.group(5),
        )
        return fw, py, "{}-{}-{}".format(ver, device, py), scriptmode

    legacy_name_pattern = re.compile(r"^sagemaker-(tensorflow|mxnet)-(py2|py3)-(cpu|gpu):(.*)$")
    legacy_match = legacy_name_pattern.match(sagemaker_match.group(9))
    if legacy_match is not None:
        return (legacy_match.group(1), legacy_match.group(2), legacy_match.group(4), None)

    # sagemaker-xgboost images are tagged with two aliases, e.g.:
    # 1. Long tag: "315553699071.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3"
    # 2. Short tag: "315553699071.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
    # Note 1: Both tags point to the same image
    # Note 2: Both tags have full GPU capabilities, despite "cpu" delineation in the long tag
    short_xgboost_tag_pattern = re.compile(r"^sagemaker-(xgboost):(.*)$")
    short_xgboost_tag_match = short_xgboost_tag_pattern.match(sagemaker_match.group(9))
    if short_xgboost_tag_match is not None:
        return (short_xgboost_tag_match.group(1), "py3", short_xgboost_tag_match.group(2), None)
    return None, None, None, None


def framework_version_from_tag(image_tag):
    """Extract the framework version from the image tag.

    Args:
        image_tag (str): Image tag, which should take the form
            '<framework_version>-<device>-<py_version>'
            '<xgboost_version>-<container_version>'

    Returns:
        str: The framework version.
    """
    tag_pattern = re.compile(r"^(.*)-(cpu|gpu)-(py2|py3\d*)$")
    tag_match = tag_pattern.match(image_tag)
    if tag_match is None:
        short_xgboost_tag_pattern = re.compile(r"^(\d\.\d+\-\d)$")
        tag_match = short_xgboost_tag_pattern.match(image_tag)
    return None if tag_match is None else tag_match.group(1)


def model_code_key_prefix(code_location_key_prefix, model_name, image):
    """Returns the s3 key prefix for uploading code during model deployment.

    The location returned is a potential concatenation of 2 parts
        1. code_location_key_prefix if it exists
        2. model_name or a name derived from the image

    Args:
        code_location_key_prefix (str): the s3 key prefix from code_location
        model_name (str): the name of the model
        image (str): the image from which a default name can be extracted

    Returns:
        str: the key prefix to be used in uploading code
    """
    name_from_image = f"/model_code/{int(time.time())}"
    if not is_pipeline_variable(image):
        name_from_image = sagemaker.utils.name_from_image(image)
    return s3_path_join(code_location_key_prefix, model_name or name_from_image)


def warn_if_parameter_server_with_multi_gpu(training_instance_type, distribution):
    """Warn the user about training when it doesn't leverage all the GPU cores.

    Warn the user that training will not fully leverage all the GPU
    cores if parameter server is enabled and a multi-GPU instance is selected.
    Distributed training with the default parameter server setup doesn't
    support multi-GPU instances.

    Args:
        training_instance_type (str): A string representing the type of training instance selected.
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) For example:

            .. code:: python

                {
                    "parameter_server": {
                        "enabled": True
                    }
                }


    """
    if training_instance_type == "local" or distribution is None:
        return
    if is_pipeline_variable(training_instance_type):
        # The training_instance_type is not available in compile time.
        # Rather, it's given in Pipeline execution time
        return

    is_multi_gpu_instance = (
        training_instance_type == "local_gpu"
        or training_instance_type.split(".")[1].startswith("p")
    ) and training_instance_type not in SINGLE_GPU_INSTANCE_TYPES

    ps_enabled = "parameter_server" in distribution and distribution["parameter_server"].get(
        "enabled", False
    )

    if is_multi_gpu_instance and ps_enabled:
        logger.warning(PARAMETER_SERVER_MULTI_GPU_WARNING)


def profiler_config_deprecation_warning(
    profiler_config, image_uri, framework_name, framework_version
):
    """Put out a deprecation message for if framework profiling is specified TF >= 2.12 and PT >= 2.0"""
    if profiler_config is None or profiler_config.framework_profile_params is None:
        return

    if framework_name not in ("pytorch", "tensorflow"):
        return

    if framework_version is None:
        framework_name, _, image_tag, _ = framework_name_from_image(image_uri)

        if image_tag is not None:
            framework_version = framework_version_from_tag(image_tag)

    if framework_version is not None:
        framework_profile_thresh = (
            version.parse("2.0") if framework_name == "pytorch" else version.parse("2.12")
        )
        framework_profile = version.parse(framework_version)
        if framework_profile >= framework_profile_thresh:
            deprecation_warn_base(
                f"Framework profiling is deprecated from\
                 {framework_name} version {framework_version}.\
                 No framework metrics will be collected"
            )


def validate_smdistributed(
    instance_type, framework_name, framework_version, py_version, distribution, image_uri=None
):
    """Check if smdistributed strategy is correctly invoked by the user.

    Currently, two strategies are supported: `dataparallel` or `modelparallel`.
    Validate if the user requested strategy is supported.

    Currently, only one strategy can be specified at a time. Validate if the user has requested
    more than one strategy simultaneously.

    Validate if the smdistributed dict arg is syntactically correct.

    Additionally, perform strategy-specific validations.

    Args:
        instance_type (str): A string representing the type of training instance selected.
        framework_name (str): A string representing the name of framework selected.
        framework_version (str): A string representing the framework version selected.
        py_version (str): A string representing the python version selected.
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) For example:

            .. code:: python

                {
                    "smdistributed": {
                        "dataparallel": {
                            "enabled": True
                        }
                    }
                }
        image_uri (str): A string representing a Docker image URI.

    Raises:
        ValueError: if distribution dictionary isn't correctly formatted or
            multiple strategies are requested simultaneously or
            an unsupported strategy is requested or
            strategy-specific inputs are incorrect/unsupported
    """
    if "smdistributed" not in distribution:
        # Distribution strategy other than smdistributed is selected
        return
    if is_pipeline_variable(instance_type) or is_pipeline_variable(image_uri):
        # The instance_type is not available in compile time.
        # Rather, it's given in Pipeline execution time
        return

    # distribution contains smdistributed
    smdistributed = distribution["smdistributed"]
    if not isinstance(smdistributed, dict):
        raise ValueError("smdistributed strategy requires a dictionary")

    if len(smdistributed) > 1:
        # more than 1 smdistributed strategy requested by the user
        err_msg = (
            "Cannot use more than 1 smdistributed strategy. \n"
            "Choose one of the following supported strategies:"
            f"{SMDISTRIBUTED_SUPPORTED_STRATEGIES}"
        )
        raise ValueError(err_msg)

    # validate if smdistributed strategy is supported
    # currently this for loop essentially checks for only 1 key
    for strategy in smdistributed:
        if strategy not in SMDISTRIBUTED_SUPPORTED_STRATEGIES:
            err_msg = (
                f"Invalid smdistributed strategy provided: {strategy} \n"
                f"Supported strategies: {SMDISTRIBUTED_SUPPORTED_STRATEGIES}"
            )
            raise ValueError(err_msg)

    # smdataparallel-specific input validation
    if "dataparallel" in smdistributed:
        _validate_smdataparallel_args(
            instance_type, framework_name, framework_version, py_version, distribution, image_uri
        )


def _validate_smdataparallel_args(
    instance_type, framework_name, framework_version, py_version, distribution, image_uri=None
):
    """Check if request is using unsupported arguments.

    Validate if user specifies a supported instance type, framework version, and python
    version.

    Args:
        instance_type (str): A string representing the type of training instance selected. Ex: `ml.p3.16xlarge`
        framework_name (str): A string representing the name of framework selected. Ex: `tensorflow`
        framework_version (str): A string representing the framework version selected. Ex: `2.3.1`
        py_version (str): A string representing the python version selected. Ex: `py3`
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) Ex:

            .. code:: python

                {
                    "smdistributed": {
                        "dataparallel": {
                            "enabled": True
                        }
                    }
                }
        image_uri (str): A string representing a Docker image URI.

    Raises:
        ValueError: if
            (`instance_type` is not in SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES or
            `py_version` is not python3 or
            `framework_version` is not in SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSION
    """
    smdataparallel_enabled = (
        distribution.get("smdistributed").get("dataparallel").get("enabled", False)
    )

    if not smdataparallel_enabled:
        return

    is_instance_type_supported = instance_type in SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES

    err_msg = ""

    if not is_instance_type_supported:
        # instance_type is required
        err_msg += (
            f"Provided instance_type {instance_type} is not supported by smdataparallel.\n"
            "Please specify one of the supported instance types:"
            f"{SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES}\n"
        )

    if not image_uri:
        # ignore framework_version & py_version if image_uri is set
        # in case image_uri is not set, then both are mandatory
        supported = SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSIONS[framework_name]
        if framework_version not in supported:
            err_msg += (
                f"Provided framework_version {framework_version} is not supported by"
                " smdataparallel.\n"
                f"Please specify one of the supported framework versions: {supported} \n"
            )

        if "py3" not in py_version:
            err_msg += (
                f"Provided py_version {py_version} is not supported by smdataparallel.\n"
                "Please specify py_version>=py3"
            )

    if err_msg:
        raise ValueError(err_msg)


def validate_distribution(
    distribution,
    instance_groups,
    framework_name,
    framework_version,
    py_version,
    image_uri,
    kwargs,
):
    """Check if distribution strategy is correctly invoked by the user.

    Currently, check for `dataparallel`, `modelparallel` and heterogeneous cluster set up.
    Validate if the user requested strategy is supported.

    Args:
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) For example:

            .. code:: python

                {
                    "smdistributed": {
                        "dataparallel": {
                            "enabled": True
                        }
                    }
                }
        instance_groups ([InstanceGroup]): A list contains instance groups used for training.
        framework_name (str): A string representing the name of framework selected.
        framework_version (str): A string representing the framework version selected.
        py_version (str): A string representing the python version selected.
        image_uri (str): A string representing a Docker image URI.
        kwargs(dict): Additional kwargs passed to this function

    Returns:
        distribution(dict): updated dictionary with validated information
            to enable distributed training.

    Raises:
        ValueError: if distribution dictionary isn't correctly formatted or
            multiple strategies are requested simultaneously or
            an unsupported strategy is requested or
            strategy-specific inputs are incorrect/unsupported or
            heterogeneous cluster set up is incorrect
    """
    train_instance_groups = distribution.get("instance_groups", [])
    if instance_groups is None:
        if len(train_instance_groups) >= 1:
            # if estimator's instance_groups is not defined but
            # train_instance_groups are specified in distribution
            raise ValueError("Instance groups not specified in the estimator !")
    else:
        if len(train_instance_groups) > len(instance_groups):
            # if train_instance_groups in distribution are more than estimator's instance_groups
            raise ValueError("Train instance groups oversubscribed !")
        if len(instance_groups) == 1 and len(train_instance_groups) == 0:
            # if just one instance_group but it is not specified in distribution, we set it for user
            train_instance_groups = instance_groups
        elif len(instance_groups) > 1 and len(train_instance_groups) != 1:
            # currently we just support one train instance group
            raise ValueError("Distribution should only contain one instance group name !")

    if len(train_instance_groups) != 0:
        # in this case, we are handling a heterogeneous cluster training job
        instance_group_names = []
        for train_instance_group in train_instance_groups:
            # in future version we will support multiple train_instance_groups, so use loop here
            if train_instance_group not in instance_groups:
                # check if train instance groups belongs to what user defined in estimator set up
                raise ValueError(
                    f"Invalid training instance group {train_instance_group.instance_group_name} !"
                )
            instance_type = train_instance_group.instance_type
            validate_distribution_for_instance_type(
                instance_type=instance_type,
                distribution=distribution,
            )
            validate_smdistributed(
                instance_type=instance_type,
                framework_name=framework_name,
                framework_version=framework_version,
                py_version=py_version,
                distribution=distribution,
                image_uri=image_uri,
            )
            if framework_name and framework_name == "pytorch":
                # We need to validate only for PyTorch framework
                validate_pytorch_distribution(
                    distribution=distribution,
                    framework_name=framework_name,
                    framework_version=framework_version,
                    py_version=py_version,
                    image_uri=image_uri,
                )
                validate_torch_distributed_distribution(
                    instance_type=instance_type,
                    distribution=distribution,
                    framework_version=framework_version,
                    py_version=py_version,
                    image_uri=image_uri,
                    entry_point=kwargs["entry_point"],
                )
            warn_if_parameter_server_with_multi_gpu(
                training_instance_type=instance_type, distribution=distribution
            )
            # get instance group names
            instance_group_names.append(train_instance_group.instance_group_name)
        distribution["instance_groups"] = instance_group_names
    else:
        # in this case, we are handling a normal training job (without heterogeneous cluster)
        instance_type = renamed_kwargs(
            "train_instance_type", "instance_type", kwargs.get("instance_type"), kwargs
        )
        validate_distribution_for_instance_type(
            instance_type=instance_type,
            distribution=distribution,
        )
        validate_smdistributed(
            instance_type=instance_type,
            framework_name=framework_name,
            framework_version=framework_version,
            py_version=py_version,
            distribution=distribution,
            image_uri=image_uri,
        )
        if framework_name and framework_name == "pytorch":
            # We need to validate only for PyTorch framework
            validate_pytorch_distribution(
                distribution=distribution,
                framework_name=framework_name,
                framework_version=framework_version,
                py_version=py_version,
                image_uri=image_uri,
            )
            validate_torch_distributed_distribution(
                instance_type=instance_type,
                distribution=distribution,
                framework_version=framework_version,
                py_version=py_version,
                image_uri=image_uri,
                entry_point=kwargs["entry_point"],
            )
        warn_if_parameter_server_with_multi_gpu(
            training_instance_type=instance_type, distribution=distribution
        )
    return distribution


def validate_distribution_for_instance_type(instance_type, distribution):
    """Check if the provided distribution strategy is supported for the instance_type

    Args:
        instance_type (str): A string representing the type of training instance selected.
        distribution (dict): A dictionary with information to enable distributed training.
    """
    err_msg = ""
    if isinstance(instance_type, str):
        match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
        if match and match[1].startswith("trn"):
            keys = list(distribution.keys())
            if len(keys) == 0:
                return
            if len(keys) == 1:
                distribution_strategy = keys[0]
                if distribution_strategy != "torch_distributed":
                    err_msg += (
                        f"Provided distribution strategy {distribution_strategy} is not supported"
                        " for Trainium instances.\n"
                        "Please specify one of the following supported distribution strategies:"
                        f" {TRAINIUM_SUPPORTED_DISTRIBUTION_STRATEGIES} \n"
                    )
            elif len(keys) > 1:
                err_msg += (
                    "Multiple distribution strategies are not supported for Trainium instances.\n"
                    "Please specify one of the following supported distribution strategies:"
                    f" {TRAINIUM_SUPPORTED_DISTRIBUTION_STRATEGIES} "
                )

    if err_msg:
        raise ValueError(err_msg)


def validate_pytorch_distribution(
    distribution, framework_name, framework_version, py_version, image_uri
):
    """Check if pytorch distribution strategy is correctly invoked by the user.

    Args:
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) For example:

            .. code:: python

                {
                    "pytorchddp": {
                        "enabled": True
                    }
                }
        framework_name (str): A string representing the name of framework selected.
        framework_version (str): A string representing the framework version selected.
        py_version (str): A string representing the python version selected.
        image_uri (str): A string representing a Docker image URI.

    Raises:
        ValueError: if
            `py_version` is not python3 or
            `framework_version` is not in PYTORCHDDP_SUPPORTED_FRAMEWORK_VERSIONS
    """
    if framework_name and framework_name != "pytorch":
        # We need to validate only for PyTorch framework
        return

    pytorch_ddp_enabled = False
    if "pytorchddp" in distribution:
        pytorch_ddp_enabled = distribution.get("pytorchddp").get("enabled", False)
    if not pytorch_ddp_enabled:
        # Distribution strategy other than pytorchddp is selected
        return

    err_msg = ""
    if not image_uri:
        # ignore framework_version and py_version if image_uri is set
        # in case image_uri is not set, then both are mandatory
        if framework_version not in PYTORCHDDP_SUPPORTED_FRAMEWORK_VERSIONS:
            err_msg += (
                f"Provided framework_version {framework_version} is not supported by"
                " pytorchddp.\n"
                "Please specify one of the supported framework versions:"
                f" {PYTORCHDDP_SUPPORTED_FRAMEWORK_VERSIONS} \n"
            )
        if "py3" not in py_version:
            err_msg += (
                f"Provided py_version {py_version} is not supported by pytorchddp.\n"
                "Please specify py_version>=py3"
            )
    if err_msg:
        raise ValueError(err_msg)


def validate_torch_distributed_distribution(
    instance_type,
    distribution,
    framework_version,
    py_version,
    image_uri,
    entry_point,
):
    """Check if torch_distributed distribution strategy is correctly invoked by the user.

    Args:
        instance_type (str): A string representing the type of training instance selected.
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) For example:

            .. code:: python

                {
                    "torch_distributed": {
                        "enabled": True
                    }
                }
        framework_version (str): A string representing the framework version selected.
        py_version (str): A string representing the python version selected.
        image_uri (str): A string representing a Docker image URI.
        entry_point (str or PipelineVariable): The absolute or relative path to the local Python
            source file that should be executed as the entry point to
            training.

    Raises:
        ValueError: if
            `py_version` is not python3 or
            `framework_version` is not compatible with instance types
    """
    torch_distributed_enabled = False
    if "torch_distributed" in distribution:
        torch_distributed_enabled = distribution.get("torch_distributed").get("enabled", False)
    if not torch_distributed_enabled:
        # Distribution strategy other than torch_distributed is selected
        return

    err_msg = ""

    if not image_uri:
        # ignore framework_version and py_version if image_uri is set
        # in case image_uri is not set, then both are mandatory
        if "py3" not in py_version:
            err_msg += (
                f"Provided py_version {py_version} is not supported by torch_distributed.\n"
                "Please specify py_version>=py3\n"
            )

        # Check instance and framework_version compatibility
        if _is_gpu_instance(instance_type):
            if framework_version not in TORCH_DISTRIBUTED_GPU_SUPPORTED_FRAMEWORK_VERSIONS:
                err_msg += (
                    f"Provided framework_version {framework_version} is not supported by"
                    f" torch_distributed for instance {instance_type}.\n"
                    "Please specify one of the supported framework versions:"
                    f"{TORCH_DISTRIBUTED_GPU_SUPPORTED_FRAMEWORK_VERSIONS} \n"
                )
        elif _is_trainium_instance(instance_type):
            if framework_version not in TRAINIUM_SUPPORTED_TORCH_DISTRIBUTED_FRAMEWORK_VERSIONS:
                err_msg += (
                    f"Provided framework_version {framework_version} is not supported by"
                    f" torch_distributed for instance {instance_type}.\n"
                    "Please specify one of the supported framework versions:"
                    f"{TRAINIUM_SUPPORTED_TORCH_DISTRIBUTED_FRAMEWORK_VERSIONS} \n"
                )
        else:
            err_msg += (
                "Currently torch_distributed is supported only for GPU and Trainium instances.\n"
            )

    # Check entry point type
    if not entry_point.endswith(".py"):
        err_msg += (
            "Unsupported entry point type for the distribution torch_distributed.\n"
            "Only python programs (*.py) are supported."
        )

    if err_msg:
        raise ValueError(err_msg)


def _is_gpu_instance(instance_type):
    """Returns bool indicating whether instance_type supports GPU

    Args:
        instance_type (str): Name of the instance_type to check against.

    Returns:
        bool: Whether or not the instance_type supports GPU
    """
    if isinstance(instance_type, str):
        match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
        if match:
            if match[1].startswith("p") or match[1].startswith("g"):
                return True
        if instance_type == "local_gpu":
            return True
    return False


def _is_trainium_instance(instance_type):
    """Returns bool indicating whether instance_type is a Trainium instance

    Args:
        instance_type (str): Name of the instance_type to check against.

    Returns:
        bool: Whether or not the instance_type is a Trainium instance
    """
    if isinstance(instance_type, str):
        match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
        if match and match[1].startswith("trn"):
            return True
    return False


def python_deprecation_warning(framework, latest_supported_version):
    """Placeholder docstring"""
    return PYTHON_2_DEPRECATION_WARNING.format(
        framework=framework, latest_supported_version=latest_supported_version
    )


def _region_supports_debugger(region_name):
    """Returns boolean indicating whether the region supports Amazon SageMaker Debugger.

    Args:
        region_name (str): Name of the region to check against.

    Returns:
        bool: Whether or not the region supports Amazon SageMaker Debugger.

    """
    return region_name.lower() not in DEBUGGER_UNSUPPORTED_REGIONS


def _region_supports_profiler(region_name):
    """Returns bool indicating whether region supports Amazon SageMaker Debugger profiling feature.

    Args:
        region_name (str): Name of the region to check against.

    Returns:
        bool: Whether or not the region supports Amazon SageMaker Debugger profiling feature.

    """
    return region_name.lower() not in PROFILER_UNSUPPORTED_REGIONS


def _instance_type_supports_profiler(instance_type):
    """Returns bool indicating whether instance_type supports SageMaker Debugger profiling feature.

    Args:
        instance_type (str): Name of the instance_type to check against.

    Returns:
        bool: Whether or not the region supports Amazon SageMaker Debugger profiling feature.
    """
    if isinstance(instance_type, str):
        match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
        if match and match[1].startswith("trn"):
            return True
    return False


def validate_version_or_image_args(framework_version, py_version, image_uri):
    """Checks if version or image arguments are specified.

    Validates framework and model arguments to enforce version or image specification.

    Args:
        framework_version (str): The version of the framework.
        py_version (str): The version of Python.
        image_uri (str): The URI of the image.

    Raises:
        ValueError: if `image_uri` is None and either `framework_version` or `py_version` is
            None.
    """
    if (framework_version is None or py_version is None) and image_uri is None:
        raise ValueError(
            "framework_version or py_version was None, yet image_uri was also None. "
            "Either specify both framework_version and py_version, or specify image_uri."
        )


def create_image_uri(
    region,
    framework,
    instance_type,
    framework_version,
    py_version=None,
    account=None,  # pylint: disable=W0613
    accelerator_type=None,
    optimized_families=None,  # pylint: disable=W0613
):
    """Deprecated method. Please use sagemaker.image_uris.retrieve().

    Args:
        region (str): AWS region where the image is uploaded.
        framework (str): framework used by the image.
        instance_type (str): SageMaker instance type. Used to determine device
            type (cpu/gpu/family-specific optimized).
        framework_version (str): The version of the framework.
        py_version (str): Optional. Python version. If specified, should be one
            of 'py2' or 'py3'. If not specified, image uri will not include a
            python component.
        account (str): AWS account that contains the image. (default:
            '520713654638')
        accelerator_type (str): SageMaker Elastic Inference accelerator type.
        optimized_families (str): Deprecated. A no-op argument.

    Returns:
        the image uri
    """
    renamed_warning("The method create_image_uri")
    return sagemaker.image_uris.retrieve(
        framework=framework,
        region=region,
        version=framework_version,
        py_version=py_version,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
    )
