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
"""This module contains functions for obtaining JumpStart model packages."""
from __future__ import absolute_import
from typing import Optional
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.session import Session


def _retrieve_model_package_arn(
    model_id: str,
    model_version: str,
    region: Optional[str],
    scope: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Optional[str]:
    """Retrieves associated model pacakge arn for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the model package arn.
        model_version (str): Version of the JumpStart model for which to retrieve the
            model package arn.
        region (Optional[str]): Region for which to retrieve the model package arn.
        scope (Optional[str]): Scope for which to retrieve the model package arn.
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
        str: the model package arn to use for the model or None.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    if scope == JumpStartScriptScope.INFERENCE:

        if model_specs.hosting_model_package_arns is None:
            return None

        regional_arn = model_specs.hosting_model_package_arns.get(region)

        if regional_arn is None:
            raise ValueError(
                f"Model package arn for '{model_id}' not supported in {region}. "
                "Please try one of the following regions: "
                f"{', '.join(model_specs.hosting_model_package_arns.keys())}."
            )

        return regional_arn

    raise NotImplementedError(f"Model Package ARN not supported for scope: '{scope}'")


def _retrieve_model_package_model_artifact_s3_uri(
    model_id: str,
    model_version: str,
    region: Optional[str],
    scope: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Optional[str]:
    """Retrieves s3 artifact uri associated with model package.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the model package artifact.
        model_version (str): Version of the JumpStart model for which to retrieve the
            model package artifact.
        region (Optional[str]): Region for which to retrieve the model package artifact.
            (Default: None).
        scope (Optional[str]): Scope for which to retrieve the model package artifact.
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
        str: the model package artifact uri to use for the model or None.

    Raises:
        NotImplementedError: If an unsupported script is used.
    """

    if scope == JumpStartScriptScope.TRAINING:

        if region is None:
            region = JUMPSTART_DEFAULT_REGION_NAME

        model_specs = verify_model_region_and_return_specs(
            model_id=model_id,
            version=model_version,
            scope=scope,
            region=region,
            tolerate_vulnerable_model=tolerate_vulnerable_model,
            tolerate_deprecated_model=tolerate_deprecated_model,
            sagemaker_session=sagemaker_session,
        )

        if model_specs.training_model_package_artifact_uris is None:
            return None

        model_s3_uri = model_specs.training_model_package_artifact_uris.get(region)

        if model_s3_uri is None:
            raise ValueError(
                f"Model package artifact s3 uri for '{model_id}' not supported in {region}. "
                "Please try one of the following regions: "
                f"{', '.join(model_specs.training_model_package_artifact_uris.keys())}."
            )

        return model_s3_uri

    raise NotImplementedError(f"Model Package Artifact URI not supported for scope: '{scope}'")
