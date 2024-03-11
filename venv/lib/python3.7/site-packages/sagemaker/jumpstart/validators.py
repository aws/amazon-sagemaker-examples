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
"""This module contains validators related to SageMaker JumpStart."""
from __future__ import absolute_import
from typing import Any, Dict, List, Optional
from sagemaker import session
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME

from sagemaker.jumpstart.enums import (
    HyperparameterValidationMode,
    JumpStartScriptScope,
    VariableScope,
    VariableTypes,
)
from sagemaker.jumpstart.exceptions import JumpStartHyperparametersError
from sagemaker.jumpstart.types import JumpStartHyperparameter
from sagemaker.jumpstart.utils import verify_model_region_and_return_specs


def _validate_hyperparameter(
    hyperparameter_name: str,
    hyperparameter_value: Any,
    hyperparameter_specs: List[JumpStartHyperparameter],
) -> None:
    """Perform low-level hyperparameter validation on single parameter.

    Args:
      hyperparameter_name (str): The name of the hyperparameter to validate.
      hyperparameter_value (Any): The value of the hyperparemter to validate.
      hyperparameter_specs (List[JumpStartHyperparameter]): List of ``JumpStartHyperparameter`` to
        use when validating the hyperparameter.

    Raises:
        JumpStartHyperparametersError: If the hyperparameter is not formatted correctly,
            according to its specs in the model metadata.
    """
    hyperparameter_spec = [
        spec for spec in hyperparameter_specs if spec.name == hyperparameter_name
    ]
    if len(hyperparameter_spec) == 0:
        raise JumpStartHyperparametersError(
            f"Unable to perform validation -- cannot find hyperparameter '{hyperparameter_name}' "
            "in model specs."
        )

    if len(hyperparameter_spec) > 1:
        raise JumpStartHyperparametersError(
            "Unable to perform validation -- found multiple hyperparameter "
            f"'{hyperparameter_name}' in model specs."
        )

    hyperparameter_spec = hyperparameter_spec[0]

    if hyperparameter_spec.type == VariableTypes.BOOL.value:
        if isinstance(hyperparameter_value, bool):
            return
        if not isinstance(hyperparameter_value, str):
            raise JumpStartHyperparametersError(
                f"Expecting boolean valued hyperparameter, but got '{str(hyperparameter_value)}'."
            )
        if str(hyperparameter_value).lower() not in ["true", "false"]:
            raise JumpStartHyperparametersError(
                f"Expecting boolean valued hyperparameter, but got '{str(hyperparameter_value)}'."
            )
    elif hyperparameter_spec.type == VariableTypes.TEXT.value:
        if not isinstance(hyperparameter_value, str):
            raise JumpStartHyperparametersError(
                "Expecting text valued hyperparameter to have string type."
            )

        if hasattr(hyperparameter_spec, "options"):
            if hyperparameter_value not in hyperparameter_spec.options:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' must have one of the following "
                    f"values: {', '.join(hyperparameter_spec.options)}."
                )

        if hasattr(hyperparameter_spec, "min"):
            if len(hyperparameter_value) < hyperparameter_spec.min:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' must have length no less than "
                    f"{hyperparameter_spec.min}."
                )

        if hasattr(hyperparameter_spec, "exclusive_min"):
            if len(hyperparameter_value) <= hyperparameter_spec.exclusive_min:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' must have length greater than "
                    f"{hyperparameter_spec.exclusive_min}."
                )

        if hasattr(hyperparameter_spec, "max"):
            if len(hyperparameter_value) > hyperparameter_spec.max:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' must have length no greater than "
                    f"{hyperparameter_spec.max}."
                )

        if hasattr(hyperparameter_spec, "exclusive_max"):
            if len(hyperparameter_value) >= hyperparameter_spec.exclusive_max:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' must have length less than "
                    f"{hyperparameter_spec.exclusive_max}."
                )

    # validate numeric types
    elif hyperparameter_spec.type in [VariableTypes.INT.value, VariableTypes.FLOAT.value]:
        try:
            numeric_hyperparam_value = float(hyperparameter_value)
        except ValueError:
            raise JumpStartHyperparametersError(
                f"Hyperparameter '{hyperparameter_name}' must be numeric type "
                f"('{hyperparameter_value}')."
            )

        if hyperparameter_spec.type == VariableTypes.INT.value:
            hyperparameter_value_str = str(hyperparameter_value)
            start_index = 0
            if hyperparameter_value_str[0] in ["+", "-"]:
                start_index = 1
            if not hyperparameter_value_str[start_index:].isdigit():
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' must be integer type "
                    f"('{hyperparameter_value}')."
                )

        if hasattr(hyperparameter_spec, "min"):
            if numeric_hyperparam_value < hyperparameter_spec.min:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' can be no less than "
                    f"{hyperparameter_spec.min}."
                )

        if hasattr(hyperparameter_spec, "max"):
            if numeric_hyperparam_value > hyperparameter_spec.max:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' can be no greater than "
                    f"{hyperparameter_spec.max}."
                )

        if hasattr(hyperparameter_spec, "exclusive_min"):
            if numeric_hyperparam_value <= hyperparameter_spec.exclusive_min:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' must be greater than "
                    f"{hyperparameter_spec.exclusive_min}."
                )

        if hasattr(hyperparameter_spec, "exclusive_max"):
            if numeric_hyperparam_value >= hyperparameter_spec.exclusive_max:
                raise JumpStartHyperparametersError(
                    f"Hyperparameter '{hyperparameter_name}' must be less than "
                    f"{hyperparameter_spec.exclusive_max}."
                )


def validate_hyperparameters(
    model_id: str,
    model_version: str,
    hyperparameters: Dict[str, Any],
    validation_mode: HyperparameterValidationMode = HyperparameterValidationMode.VALIDATE_PROVIDED,
    region: Optional[str] = JUMPSTART_DEFAULT_REGION_NAME,
    sagemaker_session: Optional[session.Session] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> None:
    """Validate hyperparameters for JumpStart models.

    Args:
        model_id (str): Model ID of the model for which to validate hyperparameters.
        model_version (str): Version of the model for which to validate hyperparameters.
        hyperparameters (dict): Hyperparameters to validate.
        validation_mode (HyperparameterValidationMode): Method of validation to use with
          hyperparameters. If set to ``VALIDATE_PROVIDED``, only hyperparameters provided
          to this function will be validated, the missing hyperparameters will be ignored.
          If set to``VALIDATE_ALGORITHM``, all algorithm hyperparameters will be validated.
          If set to ``VALIDATE_ALL``, all hyperparameters for the model will be validated.
        region (str): Region for which to validate hyperparameters. (Default: JumpStart
          default region).
        sagemaker_session (Optional[Session]): Custom SageMaker Session to use.
          (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
           specifications should be tolerated (exception not raised). If False, raises an
           exception if the script used by this version of the model has dependencies with known
           security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated models should be tolerated
           (exception not raised). False if these models should raise an exception.
           (Default: False).

    Raises:
        JumpStartHyperparametersError: If the hyperparameters are not formatted correctly,
            according to their metadata specs.

    """

    if validation_mode is None:
        validation_mode = HyperparameterValidationMode.VALIDATE_PROVIDED

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        region=region,
        scope=JumpStartScriptScope.TRAINING,
        sagemaker_session=sagemaker_session,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
    )
    hyperparameters_specs = model_specs.hyperparameters

    if validation_mode == HyperparameterValidationMode.VALIDATE_PROVIDED:
        for hyperparam_name, hyperparam_value in hyperparameters.items():
            _validate_hyperparameter(hyperparam_name, hyperparam_value, hyperparameters_specs)

    elif validation_mode == HyperparameterValidationMode.VALIDATE_ALGORITHM:
        for hyperparam in hyperparameters_specs:
            if hyperparam.scope == VariableScope.ALGORITHM:
                if hyperparam.name not in hyperparameters:
                    raise JumpStartHyperparametersError(
                        f"Cannot find algorithm hyperparameter for '{hyperparam.name}'."
                    )
                _validate_hyperparameter(
                    hyperparam.name, hyperparameters[hyperparam.name], hyperparameters_specs
                )

    elif validation_mode == HyperparameterValidationMode.VALIDATE_ALL:
        for hyperparam in hyperparameters_specs:
            if hyperparam.name not in hyperparameters:
                raise JumpStartHyperparametersError(
                    f"Cannot find hyperparameter for '{hyperparam.name}'."
                )
            _validate_hyperparameter(
                hyperparam.name, hyperparameters[hyperparam.name], hyperparameters_specs
            )

    else:
        raise NotImplementedError(
            f"Unable to handle validation for the mode '{validation_mode.value}'."
        )
