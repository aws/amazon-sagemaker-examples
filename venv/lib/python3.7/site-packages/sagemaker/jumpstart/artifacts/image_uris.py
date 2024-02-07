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
"""This module contains functions for obtaining JumpStart image uris."""
from __future__ import absolute_import

from typing import Optional
from sagemaker import image_uris
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    ModelFramework,
)
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


def _retrieve_image_uri(
    model_id: str,
    model_version: str,
    image_scope: str,
    framework: Optional[str] = None,
    region: Optional[str] = None,
    version: Optional[str] = None,
    py_version: Optional[str] = None,
    instance_type: Optional[str] = None,
    accelerator_type: Optional[str] = None,
    container_version: Optional[str] = None,
    distribution: Optional[str] = None,
    base_framework_version: Optional[str] = None,
    training_compiler_config: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
):
    """Retrieves the container image URI for JumpStart models.

    Only `model_id`, `model_version`, and `image_scope` are required;
    the rest of the fields are auto-populated.


    Args:
        model_id (str): JumpStart model ID for which to retrieve image URI.
        model_version (str): Version of the JumpStart model for which to retrieve
            the image URI.
        image_scope (str): The image type, i.e. what it is used for.
            Valid values: "training", "inference", "eia". If ``accelerator_type`` is set,
            ``image_scope`` is ignored.
        framework (str): The name of the framework or algorithm.
        region (str): The AWS region. (Default: None).
        version (str): The framework or algorithm version. This is required if there is
            more than one supported version for the given framework or algorithm.
            (Default: None).
        py_version (str): The Python version. This is required if there is
            more than one supported Python version for the given framework version.
        instance_type (str): The SageMaker instance type. For supported types, see
            https://aws.amazon.com/sagemaker/pricing/instance-types. This is required if
            there are different images for different processor types.
            (Default: None).
        accelerator_type (str): Elastic Inference accelerator type. For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
            (Default: None).
        container_version (str): the version of docker image.
            Ideally the value of parameter should be created inside the framework.
            For custom use, see the list of supported container versions:
            https://github.com/aws/deep-learning-containers/blob/master/available_images.md.
            (Default: None).
        distribution (dict): A dictionary with information on how to run distributed training.
            (Default: None).
        training_compiler_config (:class:`~sagemaker.training_compiler.TrainingCompilerConfig`):
            A configuration class for the SageMaker Training Compiler.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    Returns:
        str: the ECR URI for the corresponding SageMaker Docker image.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """
    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=image_scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    if image_scope == JumpStartScriptScope.INFERENCE:
        ecr_specs = model_specs.hosting_ecr_specs
    elif image_scope == JumpStartScriptScope.TRAINING:
        ecr_specs = model_specs.training_ecr_specs

    if framework is not None and framework != ecr_specs.framework:
        raise ValueError(
            f"Incorrect container framework '{framework}' for JumpStart model ID '{model_id}' "
            f"and version '{model_version}'."
        )

    if version is not None and version != ecr_specs.framework_version:
        raise ValueError(
            f"Incorrect container framework version '{version}' for JumpStart model ID "
            f"'{model_id}' and version '{model_version}'."
        )

    if py_version is not None and py_version != ecr_specs.py_version:
        raise ValueError(
            f"Incorrect python version '{py_version}' for JumpStart model ID '{model_id}' "
            f"and version '{model_version}'."
        )

    base_framework_version_override: Optional[str] = None
    version_override: Optional[str] = None
    if ecr_specs.framework == ModelFramework.HUGGINGFACE:
        base_framework_version_override = ecr_specs.framework_version
        version_override = ecr_specs.huggingface_transformers_version

    if image_scope == JumpStartScriptScope.TRAINING:
        return image_uris.get_training_image_uri(
            region=region,
            framework=ecr_specs.framework,
            framework_version=version_override or ecr_specs.framework_version,
            py_version=ecr_specs.py_version,
            image_uri=None,
            distribution=None,
            compiler_config=None,
            tensorflow_version=None,
            pytorch_version=base_framework_version_override or base_framework_version,
            instance_type=instance_type,
        )
    if base_framework_version_override is not None:
        base_framework_version_override = f"pytorch{base_framework_version_override}"

    return image_uris.retrieve(
        framework=ecr_specs.framework,
        region=region,
        version=version_override or ecr_specs.framework_version,
        py_version=ecr_specs.py_version,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        image_scope=image_scope,
        container_version=container_version,
        distribution=distribution,
        base_framework_version=base_framework_version_override or base_framework_version,
        training_compiler_config=training_compiler_config,
    )
