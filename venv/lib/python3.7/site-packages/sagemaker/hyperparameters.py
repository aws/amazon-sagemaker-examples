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
"""Accessors to retrieve hyperparameters for training jobs."""

from __future__ import absolute_import

import logging
from typing import Dict, Optional

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import artifacts
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.enums import HyperparameterValidationMode
from sagemaker.jumpstart.validators import validate_hyperparameters
from sagemaker.session import Session

logger = logging.getLogger(__name__)


def retrieve_default(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    include_container_hyperparameters: bool = False,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Dict[str, str]:
    """Retrieves the default training hyperparameters for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the default hyperparameters.
            Defaults to ``None``.
        model_id (str): The model ID of the model for which to
            retrieve the default hyperparameters. (Default: None).
        model_version (str): The version of the model for which to retrieve the
            default hyperparameters. (Default: None).
        include_container_hyperparameters (bool): ``True`` if the container hyperparameters
            should be returned. Container hyperparameters are not used to tune
            the specific algorithm. They are used by SageMaker Training jobs to set up
            the training container environment. For example, there is a container hyperparameter
            that indicates the entrypoint script to use. These hyperparameters may be required
            when creating a training job with boto3, however the ``Estimator`` classes
            add required container hyperparameters to the job. (Default: False).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated models should be tolerated
            (exception not raised). False if these models should raise an exception.
            (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    Returns:
        dict: The hyperparameters to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` when retrieving hyperparameters."
        )

    return artifacts._retrieve_default_hyperparameters(
        model_id,
        model_version,
        region,
        include_container_hyperparameters,
        tolerate_vulnerable_model,
        tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )


def validate(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
    validation_mode: HyperparameterValidationMode = HyperparameterValidationMode.VALIDATE_PROVIDED,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> None:
    """Validates hyperparameters for models.

    Args:
        region (str): The AWS Region for which to validate hyperparameters. (Default: None).
        model_id (str): The model ID of the model for which to validate hyperparameters.
            (Default: None).
        model_version (str): The version of the model for which to validate hyperparameters.
            (Default: None).
        hyperparameters (dict): Hyperparameters to validate.
            (Default: None).
        validation_mode (HyperparameterValidationMode): Method of validation to use with
          hyperparameters. If set to ``VALIDATE_PROVIDED``, only hyperparameters provided
          to this function will be validated, the missing hyperparameters will be ignored.
          If set to``VALIDATE_ALGORITHM``, all algorithm hyperparameters will be validated.
          If set to ``VALIDATE_ALL``, all hyperparameters for the model will be validated.
          (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated models should be tolerated
            (exception not raised). False if these models should raise an exception.
            (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).

    Raises:
        JumpStartHyperparametersError: If the hyperparameter is not formatted correctly,
            according to its specs in the model metadata.
        ValueError: If the combination of arguments specified is not supported.

    """

    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` when validating hyperparameters."
        )

    if model_id is None or model_version is None:
        raise RuntimeError("Model ID and version must both be non-None")

    if hyperparameters is None:
        raise ValueError("Must specify hyperparameters.")

    return validate_hyperparameters(
        model_id=model_id,
        model_version=model_version,
        hyperparameters=hyperparameters,
        validation_mode=validation_mode,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )
