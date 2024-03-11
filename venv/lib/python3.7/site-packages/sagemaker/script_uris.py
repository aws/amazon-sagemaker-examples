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
"""Accessors to retrieve the script Amazon S3 URI to run pretrained machine learning models."""

from __future__ import absolute_import

import logging
from typing import Optional

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import artifacts
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.session import Session

logger = logging.getLogger(__name__)


def retrieve(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    script_scope: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Retrieves the script S3 URI associated with the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the model script S3 URI.
        model_id (str): The model ID of the JumpStart model for which to
            retrieve the script S3 URI.
        model_version (str): The version of the JumpStart model for which to retrieve the
            model script S3 URI.
        script_scope (str): The script type.
            Valid values: "training" and "inference".
        tolerate_vulnerable_model (bool): ``True`` if vulnerable versions of model
            specifications should be tolerated without raising an exception. If ``False``, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): ``True`` if deprecated models should be tolerated
            without raising an exception. ``False`` if these models should raise an exception.
            (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    Returns:
        str: The model script URI for the corresponding model.

    Raises:
        NotImplementedError: If the scope is not supported.
        ValueError: If the combination of arguments specified is not supported.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` when retrieving script URIs."
        )

    return artifacts._retrieve_script_uri(
        model_id,
        model_version,
        script_scope,
        region,
        tolerate_vulnerable_model,
        tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )
