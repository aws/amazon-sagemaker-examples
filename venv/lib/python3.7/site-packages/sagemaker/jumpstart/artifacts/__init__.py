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
"""This module imports all JumpStart artifact functions from the respective sub-module."""
from sagemaker.jumpstart.artifacts.resource_names import (  # noqa: F401
    _retrieve_resource_name_base,
)
from sagemaker.jumpstart.artifacts.incremental_training import (  # noqa: F401
    _model_supports_incremental_training,
)
from sagemaker.jumpstart.artifacts.image_uris import _retrieve_image_uri  # noqa: F401
from sagemaker.jumpstart.artifacts.script_uris import (  # noqa: F401
    _retrieve_script_uri,
    _model_supports_inference_script_uri,
)
from sagemaker.jumpstart.artifacts.model_uris import (  # noqa: F401
    _retrieve_model_uri,
    _model_supports_training_model_uri,
)
from sagemaker.jumpstart.artifacts.hyperparameters import (  # noqa: F401
    _retrieve_default_hyperparameters,
)
from sagemaker.jumpstart.artifacts.environment_variables import (  # noqa: F401
    _retrieve_default_environment_variables,
)
from sagemaker.jumpstart.artifacts.kwargs import (  # noqa: F401
    _retrieve_model_init_kwargs,
    _retrieve_model_deploy_kwargs,
    _retrieve_estimator_init_kwargs,
    _retrieve_estimator_fit_kwargs,
)
from sagemaker.jumpstart.artifacts.instance_types import (  # noqa: F401
    _retrieve_default_instance_type,
    _retrieve_instance_types,
)
from sagemaker.jumpstart.artifacts.metric_definitions import (  # noqa: F401
    _retrieve_default_training_metric_definitions,
)
from sagemaker.jumpstart.artifacts.predictors import (  # noqa: F401
    _retrieve_serializer_from_content_type,
    _retrieve_deserializer_from_accept_type,
    _retrieve_default_deserializer,
    _retrieve_default_serializer,
    _retrieve_deserializer_options,
    _retrieve_serializer_options,
    _retrieve_default_content_type,
    _retrieve_default_accept_type,
    _retrieve_supported_accept_types,
    _retrieve_supported_content_types,
)
from sagemaker.jumpstart.artifacts.model_packages import (  # noqa: F401
    _retrieve_model_package_arn,
    _retrieve_model_package_model_artifact_s3_uri,
)
