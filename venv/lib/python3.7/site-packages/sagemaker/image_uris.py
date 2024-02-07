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
"""Functions for generating ECR image URIs for pre-built SageMaker Docker images."""
from __future__ import absolute_import

import json
import logging
import os
import re
from typing import Optional
from packaging.version import Version

from sagemaker import utils
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.utils import is_jumpstart_model_input
from sagemaker.spark import defaults
from sagemaker.jumpstart import artifacts
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.utilities import override_pipeline_parameter_var
from sagemaker.fw_utils import GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY, GRAVITON_ALLOWED_FRAMEWORKS

logger = logging.getLogger(__name__)

ECR_URI_TEMPLATE = "{registry}.dkr.{hostname}/{repository}"
HUGGING_FACE_FRAMEWORK = "huggingface"
HUGGING_FACE_LLM_FRAMEWORK = "huggingface-llm"
XGBOOST_FRAMEWORK = "xgboost"
SKLEARN_FRAMEWORK = "sklearn"
TRAINIUM_ALLOWED_FRAMEWORKS = "pytorch"
INFERENCE_GRAVITON = "inference_graviton"
DATA_WRANGLER_FRAMEWORK = "data-wrangler"
STABILITYAI_FRAMEWORK = "stabilityai"


@override_pipeline_parameter_var
def retrieve(
    framework,
    region,
    version=None,
    py_version=None,
    instance_type=None,
    accelerator_type=None,
    image_scope=None,
    container_version=None,
    distribution=None,
    base_framework_version=None,
    training_compiler_config=None,
    model_id=None,
    model_version=None,
    tolerate_vulnerable_model=False,
    tolerate_deprecated_model=False,
    sdk_version=None,
    inference_tool=None,
    serverless_inference_config=None,
    sagemaker_session=DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Retrieves the ECR URI for the Docker image matching the given arguments.

    Ideally this function should not be called directly, rather it should be called from the
    fit() function inside framework estimator.

    Args:
        framework (str): The name of the framework or algorithm.
        region (str): The AWS region.
        version (str): The framework or algorithm version. This is required if there is
            more than one supported version for the given framework or algorithm.
        py_version (str): The Python version. This is required if there is
            more than one supported Python version for the given framework version.
        instance_type (str): The SageMaker instance type. For supported types, see
            https://aws.amazon.com/sagemaker/pricing. This is required if
            there are different images for different processor types.
        accelerator_type (str): Elastic Inference accelerator type. For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
        image_scope (str): The image type, i.e. what it is used for.
            Valid values: "training", "inference", "inference_graviton", "eia".
            If ``accelerator_type`` is set, ``image_scope`` is ignored.
        container_version (str): the version of docker image.
            Ideally the value of parameter should be created inside the framework.
            For custom use, see the list of supported container versions:
            https://github.com/aws/deep-learning-containers/blob/master/available_images.md
            (default: None).
        distribution (dict): A dictionary with information on how to run distributed training
        training_compiler_config (:class:`~sagemaker.training_compiler.TrainingCompilerConfig`):
            A configuration class for the SageMaker Training Compiler
            (default: None).
        model_id (str): The JumpStart model ID for which to retrieve the image URI
            (default: None).
        model_version (str): The version of the JumpStart model for which to retrieve the
            image URI (default: None).
        tolerate_vulnerable_model (bool): ``True`` if vulnerable versions of model specifications
            should be tolerated without an exception raised. If ``False``, raises an exception if
            the script used by this version of the model has dependencies with known security
            vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model specifications
            should be tolerated without an exception raised. If False, raises an exception
            if the version of the model is deprecated. (Default: False).
        sdk_version (str): the version of python-sdk that will be used in the image retrieval.
            (default: None).
        inference_tool (str): the tool that will be used to aid in the inference.
            Valid values: "neuron, neuronx, None"
            (default: None).
        serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
            Specifies configuration related to serverless endpoint. Instance type is
            not provided in serverless inference. So this is used to determine processor type.
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).

    Returns:
        str: The ECR URI for the corresponding SageMaker Docker image.

    Raises:
        NotImplementedError: If the scope is not supported.
        ValueError: If the combination of arguments specified is not supported or
            any PipelineVariable object is passed in.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """
    args = dict(locals())
    for name, val in args.items():
        if is_pipeline_variable(val):
            raise ValueError(
                "When retrieving the image_uri, the argument %s should not be a pipeline variable "
                "(%s) since pipeline variables are only interpreted in the pipeline execution time."
                % (name, type(val))
            )

    if is_jumpstart_model_input(model_id, model_version):
        return artifacts._retrieve_image_uri(
            model_id,
            model_version,
            image_scope,
            framework,
            region,
            version,
            py_version,
            instance_type,
            accelerator_type,
            container_version,
            distribution,
            base_framework_version,
            training_compiler_config,
            tolerate_vulnerable_model,
            tolerate_deprecated_model,
            sagemaker_session=sagemaker_session,
        )

    if training_compiler_config and (framework in [HUGGING_FACE_FRAMEWORK, "pytorch"]):
        final_image_scope = image_scope
        config = _config_for_framework_and_scope(
            framework + "-training-compiler", final_image_scope, accelerator_type
        )
    else:
        _framework = framework
        if framework == HUGGING_FACE_FRAMEWORK or framework in TRAINIUM_ALLOWED_FRAMEWORKS:
            inference_tool = _get_inference_tool(inference_tool, instance_type)
            if inference_tool in ["neuron", "neuronx"]:
                _framework = f"{framework}-{inference_tool}"
        final_image_scope = _get_final_image_scope(framework, instance_type, image_scope)
        _validate_for_suppported_frameworks_and_instance_type(framework, instance_type)
        config = _config_for_framework_and_scope(_framework, final_image_scope, accelerator_type)

    original_version = version
    version = _validate_version_and_set_if_needed(version, config, framework)
    version_config = config["versions"][_version_for_config(version, config)]

    if framework == HUGGING_FACE_FRAMEWORK:
        if version_config.get("version_aliases"):
            full_base_framework_version = version_config["version_aliases"].get(
                base_framework_version, base_framework_version
            )
        _validate_arg(full_base_framework_version, list(version_config.keys()), "base framework")
        version_config = version_config.get(full_base_framework_version)

    py_version = _validate_py_version_and_set_if_needed(py_version, version_config, framework)
    version_config = version_config.get(py_version) or version_config
    registry = _registry_from_region(region, version_config["registries"])
    endpoint_data = utils._botocore_resolver().construct_endpoint("ecr", region)
    if region == "il-central-1" and not endpoint_data:
        endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(region)}
    hostname = endpoint_data["hostname"]

    repo = version_config["repository"]

    processor = _processor(
        instance_type,
        config.get("processors") or version_config.get("processors"),
        serverless_inference_config,
    )

    # if container version is available in .json file, utilize that
    if version_config.get("container_version"):
        container_version = version_config["container_version"][processor]

    # Append sdk version in case of trainium instances
    if repo in ["pytorch-training-neuron"]:
        if not sdk_version:
            sdk_version = _get_latest_versions(version_config["sdk_versions"])
        container_version = sdk_version + "-" + container_version

    if framework == HUGGING_FACE_FRAMEWORK:
        pt_or_tf_version = (
            re.compile("^(pytorch|tensorflow)(.*)$").match(base_framework_version).group(2)
        )
        _version = original_version

        if repo in [
            "huggingface-pytorch-trcomp-training",
            "huggingface-tensorflow-trcomp-training",
        ]:
            _version = version
        if repo in [
            "huggingface-pytorch-inference-neuron",
            "huggingface-pytorch-inference-neuronx",
        ]:
            if not sdk_version:
                sdk_version = _get_latest_versions(version_config["sdk_versions"])
            container_version = sdk_version + "-" + container_version
            if config.get("version_aliases").get(original_version):
                _version = config.get("version_aliases")[original_version]
            if (
                config.get("versions", {})
                .get(_version, {})
                .get("version_aliases", {})
                .get(base_framework_version, {})
            ):
                _base_framework_version = config.get("versions")[_version]["version_aliases"][
                    base_framework_version
                ]
                pt_or_tf_version = (
                    re.compile("^(pytorch|tensorflow)(.*)$").match(_base_framework_version).group(2)
                )

        tag_prefix = f"{pt_or_tf_version}-transformers{_version}"
    else:
        tag_prefix = version_config.get("tag_prefix", version)

    if repo == f"{framework}-inference-graviton":
        container_version = f"{container_version}-sagemaker"
    _validate_instance_deprecation(framework, instance_type, version)

    tag = _get_image_tag(
        container_version,
        distribution,
        final_image_scope,
        framework,
        inference_tool,
        instance_type,
        processor,
        py_version,
        tag_prefix,
        version,
    )

    if tag:
        repo += ":{}".format(tag)

    return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo)


def _get_instance_type_family(instance_type):
    """Return the family of the instance type.

    Regex matches either "ml.<family>.<size>" or "ml_<family>. If input is None
    or there is no match, return an empty string.
    """
    instance_type_family = ""
    if isinstance(instance_type, str):
        match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
        if match is not None:
            instance_type_family = match[1]
    return instance_type_family


def _get_image_tag(
    container_version,
    distribution,
    final_image_scope,
    framework,
    inference_tool,
    instance_type,
    processor,
    py_version,
    tag_prefix,
    version,
):
    """Return image tag based on framework, container, and compute configuration(s)."""
    instance_type_family = _get_instance_type_family(instance_type)
    if framework in (XGBOOST_FRAMEWORK, SKLEARN_FRAMEWORK):
        if instance_type_family and final_image_scope == INFERENCE_GRAVITON:
            _validate_arg(
                instance_type_family,
                GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY,
                "instance type",
            )
        if (
            instance_type_family in GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY
            or final_image_scope == INFERENCE_GRAVITON
        ):
            version_to_arm64_tag_mapping = {
                "xgboost": {
                    "1.5-1": "1.5-1-arm64",
                    "1.3-1": "1.3-1-arm64",
                },
                "sklearn": {
                    "1.0-1": "1.0-1-arm64-cpu-py3",
                },
            }
            tag = version_to_arm64_tag_mapping[framework][version]
        else:
            tag = _format_tag(tag_prefix, processor, py_version, container_version, inference_tool)
    else:
        tag = _format_tag(tag_prefix, processor, py_version, container_version, inference_tool)

        if instance_type is not None and _should_auto_select_container_version(
            instance_type, distribution
        ):
            container_versions = {
                "tensorflow-2.3-gpu-py37": "cu110-ubuntu18.04-v3",
                "tensorflow-2.3.1-gpu-py37": "cu110-ubuntu18.04",
                "tensorflow-2.3.2-gpu-py37": "cu110-ubuntu18.04",
                "tensorflow-1.15-gpu-py37": "cu110-ubuntu18.04-v8",
                "tensorflow-1.15.4-gpu-py37": "cu110-ubuntu18.04",
                "tensorflow-1.15.5-gpu-py37": "cu110-ubuntu18.04",
                "mxnet-1.8-gpu-py37": "cu110-ubuntu16.04-v1",
                "mxnet-1.8.0-gpu-py37": "cu110-ubuntu16.04",
                "pytorch-1.6-gpu-py36": "cu110-ubuntu18.04-v3",
                "pytorch-1.6.0-gpu-py36": "cu110-ubuntu18.04",
                "pytorch-1.6-gpu-py3": "cu110-ubuntu18.04-v3",
                "pytorch-1.6.0-gpu-py3": "cu110-ubuntu18.04",
            }
            key = "-".join([framework, tag])
            if key in container_versions:
                tag = "-".join([tag, container_versions[key]])

    return tag


def _config_for_framework_and_scope(framework, image_scope, accelerator_type=None):
    """Loads the JSON config for the given framework and image scope."""
    config = config_for_framework(framework)

    if accelerator_type:
        _validate_accelerator_type(accelerator_type)

        if image_scope not in ("eia", "inference"):
            logger.warning(
                "Elastic inference is for inference only. Ignoring image scope: %s.", image_scope
            )
        image_scope = "eia"

    available_scopes = config.get("scope", list(config.keys()))

    if len(available_scopes) == 1:
        if image_scope and image_scope != available_scopes[0]:
            logger.warning(
                "Defaulting to only supported image scope: %s. Ignoring image scope: %s.",
                available_scopes[0],
                image_scope,
            )
        image_scope = available_scopes[0]

    if not image_scope and "scope" in config and set(available_scopes) == {"training", "inference"}:
        logger.info(
            "Same images used for training and inference. Defaulting to image scope: %s.",
            available_scopes[0],
        )
        image_scope = available_scopes[0]

    _validate_arg(image_scope, available_scopes, "image scope")
    return config if "scope" in config else config[image_scope]


def _validate_instance_deprecation(framework, instance_type, version):
    """Check if instance type is deprecated for a certain framework with a certain version"""
    if _get_instance_type_family(instance_type) == "p2":
        if (framework == "pytorch" and Version(version) >= Version("1.13")) or (
            framework == "tensorflow" and Version(version) >= Version("2.12")
        ):
            raise ValueError(
                "P2 instances have been deprecated for sagemaker jobs starting PyTorch 1.13 and TensorFlow 2.12"
                "For information about supported instance types please refer to "
                "https://aws.amazon.com/sagemaker/pricing/"
            )


def _validate_for_suppported_frameworks_and_instance_type(framework, instance_type):
    """Validate if framework is supported for the instance_type"""
    # Validate for Trainium allowed frameworks
    if (
        instance_type is not None
        and "trn" in instance_type
        and framework not in TRAINIUM_ALLOWED_FRAMEWORKS
    ):
        _validate_framework(framework, TRAINIUM_ALLOWED_FRAMEWORKS, "framework", "Trainium")

    # Validate for Graviton allowed frameowrks
    if (
        instance_type is not None
        and _get_instance_type_family(instance_type) in GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY
        and framework not in GRAVITON_ALLOWED_FRAMEWORKS
    ):
        _validate_framework(framework, GRAVITON_ALLOWED_FRAMEWORKS, "framework", "Graviton")


def config_for_framework(framework):
    """Loads the JSON config for the given framework."""
    fname = os.path.join(os.path.dirname(__file__), "image_uri_config", "{}.json".format(framework))
    with open(fname) as f:
        return json.load(f)


def _get_final_image_scope(framework, instance_type, image_scope):
    """Return final image scope based on provided framework and instance type."""
    if (
        framework in GRAVITON_ALLOWED_FRAMEWORKS
        and _get_instance_type_family(instance_type) in GRAVITON_ALLOWED_TARGET_INSTANCE_FAMILY
    ):
        return INFERENCE_GRAVITON
    if image_scope is None and framework in (XGBOOST_FRAMEWORK, SKLEARN_FRAMEWORK):
        # Preserves backwards compatibility with XGB/SKLearn configs which no
        # longer define top-level "scope" keys after introducing support for
        # Graviton inference. Training and inference configs for XGB/SKLearn are
        # identical, so default to training.
        return "training"
    return image_scope


def _get_inference_tool(inference_tool, instance_type):
    """Extract the inference tool name from instance type."""
    if not inference_tool:
        instance_type_family = _get_instance_type_family(instance_type)
        if instance_type_family.startswith("inf") or instance_type_family.startswith("trn"):
            return "neuron"
    return inference_tool


def _get_latest_versions(list_of_versions):
    """Extract the latest version from the input list of available versions."""
    return sorted(list_of_versions, reverse=True)[0]


def _validate_accelerator_type(accelerator_type):
    """Raises a ``ValueError`` if ``accelerator_type`` is invalid."""
    if not accelerator_type.startswith("ml.eia") and accelerator_type != "local_sagemaker_notebook":
        raise ValueError(
            "Invalid SageMaker Elastic Inference accelerator type: {}. "
            "See https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html".format(accelerator_type)
        )


def _validate_version_and_set_if_needed(version, config, framework):
    """Checks if the framework/algorithm version is one of the supported versions."""
    available_versions = list(config["versions"].keys())
    aliased_versions = list(config.get("version_aliases", {}).keys())

    if len(available_versions) == 1 and version not in aliased_versions:
        log_message = "Defaulting to the only supported framework/algorithm version: {}.".format(
            available_versions[0]
        )
        if version and version != available_versions[0]:
            logger.warning("%s Ignoring framework/algorithm version: %s.", log_message, version)
        elif not version:
            logger.info(log_message)

        return available_versions[0]

    if version is None and framework in [
        DATA_WRANGLER_FRAMEWORK,
        HUGGING_FACE_LLM_FRAMEWORK,
        STABILITYAI_FRAMEWORK,
    ]:
        version = _get_latest_versions(available_versions)

    _validate_arg(version, available_versions + aliased_versions, "{} version".format(framework))
    return version


def _version_for_config(version, config):
    """Returns the version string for retrieving a framework version's specific config."""
    if "version_aliases" in config:
        if version in config["version_aliases"].keys():
            return config["version_aliases"][version]

    return version


def _registry_from_region(region, registry_dict):
    """Returns the ECR registry (AWS account number) for the given region."""
    _validate_arg(region, registry_dict.keys(), "region")
    return registry_dict[region]


def _processor(instance_type, available_processors, serverless_inference_config=None):
    """Returns the processor type for the given instance type."""
    if not available_processors:
        logger.info("Ignoring unnecessary instance type: %s.", instance_type)
        return None

    if len(available_processors) == 1 and not instance_type:
        logger.info("Defaulting to only supported image scope: %s.", available_processors[0])
        return available_processors[0]

    if serverless_inference_config is not None:
        logger.info("Defaulting to CPU type when using serverless inference")
        return "cpu"

    if not instance_type:
        raise ValueError(
            "Empty SageMaker instance type. For options, see: "
            "https://aws.amazon.com/sagemaker/pricing/instance-types"
        )

    if instance_type.startswith("local"):
        processor = "cpu" if instance_type == "local" else "gpu"
    elif instance_type.startswith("neuron"):
        processor = "neuron"
    else:
        # looks for either "ml.<family>.<size>" or "ml_<family>"
        family = _get_instance_type_family(instance_type)
        if family:
            # For some frameworks, we have optimized images for specific families, e.g c5 or p3.
            # In those cases, we use the family name in the image tag. In other cases, we use
            # 'cpu' or 'gpu'.
            if family in available_processors:
                processor = family
            elif family.startswith("inf"):
                processor = "inf"
            elif family.startswith("trn"):
                processor = "trn"
            elif family[0] in ("g", "p"):
                processor = "gpu"
            else:
                processor = "cpu"
        else:
            raise ValueError(
                "Invalid SageMaker instance type: {}. For options, see: "
                "https://aws.amazon.com/sagemaker/pricing/instance-types".format(instance_type)
            )

    _validate_arg(processor, available_processors, "processor")
    return processor


def _should_auto_select_container_version(instance_type, distribution):
    """Returns a boolean that indicates whether to use an auto-selected container version."""
    p4d = False
    if instance_type:
        # looks for either "ml.<family>.<size>" or "ml_<family>"
        family = _get_instance_type_family(instance_type)
        if family:
            p4d = family == "p4d"

    smdistributed = False
    if distribution:
        smdistributed = "smdistributed" in distribution

    return p4d or smdistributed


def _validate_py_version_and_set_if_needed(py_version, version_config, framework):
    """Checks if the Python version is one of the supported versions."""
    if "repository" in version_config:
        available_versions = version_config.get("py_versions")
    else:
        available_versions = list(version_config.keys())

    if not available_versions:
        if py_version:
            logger.info("Ignoring unnecessary Python version: %s.", py_version)
        return None

    if py_version is None and defaults.SPARK_NAME == framework:
        return None

    if py_version is None and len(available_versions) == 1:
        logger.info("Defaulting to only available Python version: %s", available_versions[0])
        return available_versions[0]

    _validate_arg(py_version, available_versions, "Python version")
    return py_version


def _validate_arg(arg, available_options, arg_name):
    """Checks if the arg is in the available options, and raises a ``ValueError`` if not."""
    if arg not in available_options:
        raise ValueError(
            "Unsupported {arg_name}: {arg}. You may need to upgrade your SDK version "
            "(pip install -U sagemaker) for newer {arg_name}s. Supported {arg_name}(s): "
            "{options}.".format(arg_name=arg_name, arg=arg, options=", ".join(available_options))
        )


def _validate_framework(framework, allowed_frameworks, arg_name, hardware_name):
    """Checks if the framework is in the allowed frameworks, and raises a ``ValueError`` if not."""
    if framework not in allowed_frameworks:
        raise ValueError(
            f"Unsupported {arg_name}: {framework}. "
            f"Supported {arg_name}(s) for {hardware_name} instances: {allowed_frameworks}."
        )


def _format_tag(tag_prefix, processor, py_version, container_version, inference_tool=None):
    """Creates a tag for the image URI."""
    if inference_tool:
        return "-".join(x for x in (tag_prefix, inference_tool, py_version, container_version) if x)
    return "-".join(x for x in (tag_prefix, processor, py_version, container_version) if x)


@override_pipeline_parameter_var
def get_training_image_uri(
    region,
    framework,
    framework_version=None,
    py_version=None,
    image_uri=None,
    distribution=None,
    compiler_config=None,
    tensorflow_version=None,
    pytorch_version=None,
    instance_type=None,
) -> str:
    """Retrieves the image URI for training.

    Args:
        region (str): The AWS region to use for image URI.
        framework (str): The framework for which to retrieve an image URI.
        framework_version (str): The framework version for which to retrieve an
            image URI (default: None).
        py_version (str): The python version to use for the image (default: None).
        image_uri (str): If an image URI is supplied, it is returned (default: None).
        distribution (dict): A dictionary with information on how to run distributed
            training (default: None).
        compiler_config (:class:`~sagemaker.training_compiler.TrainingCompilerConfig`):
            A configuration class for the SageMaker Training Compiler
            (default: None).
        tensorflow_version (str): The version of TensorFlow to use. (default: None)
        pytorch_version (str): The version of PyTorch to use. (default: None)
        instance_type (str): The instance type to use. (default: None)

    Returns:
        str: The image URI string.
    """

    if image_uri:
        return image_uri

    logger.info(
        "image_uri is not presented, retrieving image_uri based on instance_type, framework etc."
    )
    base_framework_version: Optional[str] = None

    if tensorflow_version is not None or pytorch_version is not None:
        processor = _processor(instance_type, ["cpu", "gpu"])
        is_native_huggingface_gpu = processor == "gpu" and not compiler_config
        container_version = "cu110-ubuntu18.04" if is_native_huggingface_gpu else None
        if tensorflow_version is not None:
            base_framework_version = f"tensorflow{tensorflow_version}"
        else:
            base_framework_version = f"pytorch{pytorch_version}"
    else:
        container_version = None
        base_framework_version = None

    return retrieve(
        framework,
        region,
        instance_type=instance_type,
        version=framework_version,
        py_version=py_version,
        image_scope="training",
        distribution=distribution,
        base_framework_version=base_framework_version,
        container_version=container_version,
        training_compiler_config=compiler_config,
    )


def get_base_python_image_uri(region, py_version="310") -> str:
    """Retrieves the image URI for base python image.

    Args:
        region (str): The AWS region to use for image URI.
        py_version (str): The python version to use for the image. Can be 310 or 38
        Default to 310

    Returns:
        str: The image URI string.
    """

    framework = "sagemaker-base-python"
    version = "1.0"
    endpoint_data = utils._botocore_resolver().construct_endpoint("ecr", region)
    if region == "il-central-1" and not endpoint_data:
        endpoint_data = {"hostname": "ecr.{}.amazonaws.com".format(region)}
    hostname = endpoint_data["hostname"]
    config = config_for_framework(framework)
    version_config = config["versions"][_version_for_config(version, config)]

    registry = _registry_from_region(region, version_config["registries"])

    repo = version_config["repository"] + "-" + py_version
    repo_and_tag = repo + ":" + version

    return ECR_URI_TEMPLATE.format(registry=registry, hostname=hostname, repository=repo_and_tag)
