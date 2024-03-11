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
"""This module contains functions for obtaining JumpStart kwargs."""
from __future__ import absolute_import
from copy import deepcopy
from typing import Optional
from sagemaker.session import Session
from sagemaker.utils import volume_size_supported
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs,
)


def _retrieve_model_init_kwargs(
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> dict:
    """Retrieves kwargs for `Model`.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the kwargs.
        model_version (str): Version of the JumpStart model for which to retrieve the
            kwargs.
        region (Optional[str]): Region for which to retrieve kwargs.
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
        dict: the kwargs to use for the use case.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    kwargs = deepcopy(model_specs.model_kwargs)

    if model_specs.inference_enable_network_isolation is not None:
        kwargs.update({"enable_network_isolation": model_specs.inference_enable_network_isolation})

    return kwargs


def _retrieve_model_deploy_kwargs(
    model_id: str,
    model_version: str,
    instance_type: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> dict:
    """Retrieves kwargs for `Model.deploy`.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the kwargs.
        model_version (str): Version of the JumpStart model for which to retrieve the
            kwargs.
        instance_type (str): Instance type of the hosting endpoint, to determine if volume size
            is supported.
        region (Optional[str]): Region for which to retrieve kwargs.
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
        dict: the kwargs to use for the use case.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    if volume_size_supported(instance_type) and model_specs.inference_volume_size is not None:
        return {**model_specs.deploy_kwargs, **{"volume_size": model_specs.inference_volume_size}}

    return model_specs.deploy_kwargs


def _retrieve_estimator_init_kwargs(
    model_id: str,
    model_version: str,
    instance_type: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> dict:
    """Retrieves kwargs for `Estimator`.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the kwargs.
        model_version (str): Version of the JumpStart model for which to retrieve the
            kwargs.
        instance_type (str): Instance type of the training job, to determine if volume size is
            supported.
        region (Optional[str]): Region for which to retrieve kwargs.
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
        dict: the kwargs to use for the use case.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=JumpStartScriptScope.TRAINING,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    kwargs = deepcopy(model_specs.estimator_kwargs)

    if model_specs.training_enable_network_isolation is not None:
        kwargs.update({"enable_network_isolation": model_specs.training_enable_network_isolation})

    if volume_size_supported(instance_type) and model_specs.training_volume_size is not None:
        kwargs.update({"volume_size": model_specs.training_volume_size})

    return kwargs


def _retrieve_estimator_fit_kwargs(
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> dict:
    """Retrieves kwargs for `Estimator.fit`.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the kwargs.
        model_version (str): Version of the JumpStart model for which to retrieve the
            kwargs.
        region (Optional[str]): Region for which to retrieve kwargs.
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
        dict: the kwargs to use for the use case.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=JumpStartScriptScope.TRAINING,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    return model_specs.fit_kwargs
