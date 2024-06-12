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
"""Implements methods for deserializing data returned from an inference endpoint."""
from __future__ import absolute_import


from typing import List, Optional

# base_deserializers was refactored from deserializers.
# this import ensures backward compatibility.
from sagemaker.base_deserializers import (  # noqa: F401 # pylint: disable=W0611
    BaseDeserializer,
    BytesDeserializer,
    CSVDeserializer,
    DeferredError,
    JSONDeserializer,
    JSONLinesDeserializer,
    NumpyDeserializer,
    PandasDeserializer,
    SimpleBaseDeserializer,
    StreamDeserializer,
    StringDeserializer,
)

from sagemaker.jumpstart import artifacts, utils as jumpstart_utils
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.session import Session


def retrieve_options(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> List[BaseDeserializer]:
    """Retrieves the supported deserializers for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the supported deserializers.
            Defaults to ``None``.
        model_id (str): The model ID of the model for which to
            retrieve the supported deserializers. (Default: None).
        model_version (str): The version of the model for which to retrieve the
            supported deserializers. (Default: None).
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
        List[BaseDeserializer]: The supported deserializers to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """

    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` when retrieving deserializers."
        )

    return artifacts._retrieve_deserializer_options(
        model_id,
        model_version,
        region,
        tolerate_vulnerable_model,
        tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )


def retrieve_default(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> BaseDeserializer:
    """Retrieves the default deserializer for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the default deserializer.
            Defaults to ``None``.
        model_id (str): The model ID of the model for which to
            retrieve the default deserializer. (Default: None).
        model_version (str): The version of the model for which to retrieve the
            default deserializer. (Default: None).
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
        BaseDeserializer: The default deserializer to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """

    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` when retrieving deserializers."
        )

    return artifacts._retrieve_default_deserializer(
        model_id,
        model_version,
        region,
        tolerate_vulnerable_model,
        tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )
