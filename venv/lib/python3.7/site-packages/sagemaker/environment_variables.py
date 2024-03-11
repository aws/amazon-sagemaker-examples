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
"""Accessors to retrieve environment variables for hosting containers."""

from __future__ import absolute_import

import logging
from typing import Dict, Optional

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import artifacts
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.session import Session

logger = logging.getLogger(__name__)


def retrieve_default(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    include_aws_sdk_env_vars: bool = True,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Dict[str, str]:
    """Retrieves the default container environment variables for the model matching the arguments.

    Args:
        region (str): Optional. The AWS Region for which to retrieve the default environment
             variables. (Default: None).
        model_id (str): Optional. The model ID of the model for which to
            retrieve the default environment variables. (Default: None).
        model_version (str): Optional. The version of the model for which to retrieve the
            default environment variables. (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated models should be tolerated
            (exception not raised). False if these models should raise an exception.
            (Default: False).
        include_aws_sdk_env_vars (bool): True if environment variables for low-level AWS API call
            should be included. The `Model` class of the SageMaker Python SDK inserts environment
            variables that would be required when making the low-level AWS API call.
            (Default: True).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    Returns:
        dict: The variables to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` "
            "when retrieving environment variables."
        )

    return artifacts._retrieve_default_environment_variables(
        model_id,
        model_version,
        region,
        tolerate_vulnerable_model,
        tolerate_deprecated_model,
        include_aws_sdk_env_vars,
        sagemaker_session=sagemaker_session,
    )
