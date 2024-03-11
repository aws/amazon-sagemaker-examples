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
"""This module stores exceptions related to SageMaker JumpStart."""
from __future__ import absolute_import
from typing import List, Optional

from sagemaker.jumpstart.constants import MODEL_ID_LIST_WEB_URL, JumpStartScriptScope

NO_AVAILABLE_INSTANCES_ERROR_MSG = (
    "No instances available in {region} that can support model ID '{model_id}'. "
    "Please try another region."
)

INVALID_MODEL_ID_ERROR_MSG = (
    "Invalid model ID: '{model_id}'. Please visit "
    f"{MODEL_ID_LIST_WEB_URL} for list of supported model IDs. "
    "The module `sagemaker.jumpstart.notebook_utils` contains utilities for "
    "fetching model IDs. We recommend upgrading to the latest version of sagemaker "
    "to get access to the most models."
)


class JumpStartHyperparametersError(ValueError):
    """Exception raised for bad hyperparameters of a JumpStart model."""

    def __init__(
        self,
        message: Optional[str] = None,
    ):
        self.message = message

        super().__init__(self.message)


class VulnerableJumpStartModelError(ValueError):
    """Exception raised when trying to access a JumpStart model specs flagged as vulnerable.

    Raise this exception only if the scope of attributes accessed in the specifications have
    vulnerabilities. For example, a model training script may have vulnerabilities, but not
    the hosting scripts. In such a case, raise a ``VulnerableJumpStartModelError`` only when
    accessing the training specifications.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        vulnerabilities: Optional[List[str]] = None,
        scope: Optional[JumpStartScriptScope] = None,
        message: Optional[str] = None,
    ):
        """Instantiates VulnerableJumpStartModelError exception.

        Args:
            model_id (Optional[str]): model ID of vulnerable JumpStart model.
                (Default: None).
            version (Optional[str]): version of vulnerable JumpStart model.
                (Default: None).
            vulnerabilities (Optional[List[str]]): vulnerabilities associated with
                model. (Default: None).

        """
        if message:
            self.message = message
        else:
            if None in [model_id, version, vulnerabilities, scope]:
                raise RuntimeError(
                    "Must specify `model_id`, `version`, `vulnerabilities`, " "and scope arguments."
                )
            if scope == JumpStartScriptScope.INFERENCE:
                self.message = (
                    f"Version '{version}' of JumpStart model '{model_id}' "  # type: ignore
                    "has at least 1 vulnerable dependency in the inference script. "
                    "Please try targeting a higher version of the model or using a "
                    "different model. List of vulnerabilities: "
                    f"{', '.join(vulnerabilities)}"  # type: ignore
                )
            elif scope == JumpStartScriptScope.TRAINING:
                self.message = (
                    f"Version '{version}' of JumpStart model '{model_id}' "  # type: ignore
                    "has at least 1 vulnerable dependency in the training script. "
                    "Please try targeting a higher version of the model or using a "
                    "different model. List of vulnerabilities: "
                    f"{', '.join(vulnerabilities)}"  # type: ignore
                )
            else:
                raise NotImplementedError(
                    "Unsupported scope for VulnerableJumpStartModelError: "  # type: ignore
                    f"'{scope.value}'"
                )

        super().__init__(self.message)


class DeprecatedJumpStartModelError(ValueError):
    """Exception raised when trying to access a JumpStart model deprecated specifications.

    A deprecated specification for a JumpStart model does not mean the whole model is
    deprecated. There may be more recent specifications available for this model. For
    example, all specification before version ``2.0.0`` may be deprecated, in such a
    case, the SDK would raise this exception only when specifications ``1.*`` are
    accessed.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        message: Optional[str] = None,
    ):
        if message:
            self.message = message
        else:
            if None in [model_id, version]:
                raise RuntimeError("Must specify `model_id` and `version` arguments.")
            self.message = (
                f"Version '{version}' of JumpStart model '{model_id}' is deprecated. "
                "Please try targeting a higher version of the model or using a "
                "different model."
            )

        super().__init__(self.message)
