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
from __future__ import print_function, absolute_import

from typing import Optional
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION

from sagemaker.jumpstart.factory.model import get_default_predictor
from sagemaker.jumpstart.utils import is_jumpstart_model_input

from sagemaker.session import Session


# base_predictor was refactored from predictor.
# this import ensures backward compatibility.
from sagemaker.base_predictor import (  # noqa: F401 # pylint: disable=W0611
    Predictor,
    PredictorBase,
    RealTimePredictor,
)


def retrieve_default(
    endpoint_name: str,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> Predictor:
    """Retrieves the default predictor for the model matching the given arguments.

    Args:
        endpoint_name (str): Endpoint name for which to create a predictor.
        sagemaker_session (Session): The SageMaker Session to attach to the Predictor.
            (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        region (str): The AWS Region for which to retrieve the default predictor.
            (Default: None).
        model_id (str): The model ID of the model for which to
            retrieve the default predictor. (Default: None).
        model_version (str): The version of the model for which to retrieve the
            default predictor. (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated models should be tolerated
            (exception not raised). False if these models should raise an exception.
            (Default: False).
    Returns:
        Predictor: The default predictor to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """

    if not is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` "
            "when retrieving default predictor."
        )

    predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)

    return get_default_predictor(
        predictor=predictor,
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
    )
