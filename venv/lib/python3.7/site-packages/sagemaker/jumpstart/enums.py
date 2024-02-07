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
"""This module stores enums related to SageMaker JumpStart."""

from __future__ import absolute_import

from enum import Enum


class ModelFramework(str, Enum):
    """Enum class for JumpStart model framework.

    The ML framework as referenced in the prefix of the model ID.
    This value does not necessarily correspond to the container name.
    """

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    MXNET = "mxnet"
    HUGGINGFACE = "huggingface"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"


class VariableScope(str, Enum):
    """Possible value of the ``scope`` attribute for a hyperparameter or environment variable.

    Used for hosting environment variables and training hyperparameters.
    """

    CONTAINER = "container"
    ALGORITHM = "algorithm"


class JumpStartScriptScope(str, Enum):
    """Enum class for JumpStart script scopes."""

    INFERENCE = "inference"
    TRAINING = "training"


class HyperparameterValidationMode(str, Enum):
    """Possible modes for validating hyperparameters."""

    VALIDATE_PROVIDED = "validate_provided"
    VALIDATE_ALGORITHM = "validate_algorithm"
    VALIDATE_ALL = "validate_all"


class VariableTypes(str, Enum):
    """Possible types for hyperparameters and environment variables."""

    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


class JumpStartTag(str, Enum):
    """Enum class for tag keys to apply to JumpStart models."""

    INFERENCE_MODEL_URI = "aws-jumpstart-inference-model-uri"
    INFERENCE_SCRIPT_URI = "aws-jumpstart-inference-script-uri"
    TRAINING_MODEL_URI = "aws-jumpstart-training-model-uri"
    TRAINING_SCRIPT_URI = "aws-jumpstart-training-script-uri"


class SerializerType(str, Enum):
    """Enum class for serializers associated with JumpStart models."""

    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    RAW_BYTES = "raw_bytes"


class DeserializerType(str, Enum):
    """Enum class for deserializers associated with JumpStart models."""

    JSON = "json"


class MIMEType(str, Enum):
    """Enum class for MIME types associated with JumpStart models."""

    X_IMAGE = "application/x-image"
    LIST_TEXT = "application/list-text"
    X_TEXT = "application/x-text"
    JSON = "application/json"
    CSV = "text/csv"
    WAV = "audio/wav"

    @staticmethod
    def from_suffixed_type(mime_type_with_suffix: str) -> "MIMEType":
        """Removes suffix from type and instantiates enum."""
        base_type, _, _ = mime_type_with_suffix.partition(";")
        return MIMEType(base_type)
