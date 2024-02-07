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
"""This module stores types related to SageMaker JumpStart."""
from __future__ import absolute_import
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from sagemaker.session import Session


class JumpStartDataHolderType:
    """Base class for many JumpStart types.

    Allows objects to be added to dicts and sets,
    and improves string representation. This class overrides the ``__eq__``
    and ``__hash__`` methods so that different objects with the same attributes/types
    can be compared.
    """

    __slots__: List[str] = []

    def __eq__(self, other: Any) -> bool:
        """Returns True if ``other`` is of the same type and has all attributes equal.

        Args:
            other (Any): Other object to which to compare this object.
        """

        if not isinstance(other, type(self)):
            return False
        if getattr(other, "__slots__", None) is None:
            return False
        if self.__slots__ != other.__slots__:
            return False
        for attribute in self.__slots__:
            if (hasattr(self, attribute) and not hasattr(other, attribute)) or (
                hasattr(other, attribute) and not hasattr(self, attribute)
            ):
                return False
            if hasattr(self, attribute) and hasattr(other, attribute):
                if getattr(self, attribute) != getattr(other, attribute):
                    return False
        return True

    def __hash__(self) -> int:
        """Makes hash of object.

        Maps object to unique tuple, which then gets hashed.
        """

        return hash((type(self),) + tuple([getattr(self, att) for att in self.__slots__]))

    def __str__(self) -> str:
        """Returns string representation of object. Example:

        "JumpStartLaunchedRegionInfo:
        {'content_bucket': 'bucket', 'region_name': 'us-west-2'}"
        """

        att_dict = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return f"{type(self).__name__}: {str(att_dict)}"

    def __repr__(self) -> str:
        """Returns ``__repr__`` string of object. Example:

        "JumpStartLaunchedRegionInfo at 0x7f664529efa0:
        {'content_bucket': 'bucket', 'region_name': 'us-west-2'}"
        """

        att_dict = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return f"{type(self).__name__} at {hex(id(self))}: {str(att_dict)}"


class JumpStartS3FileType(str, Enum):
    """Type of files published in JumpStart S3 distribution buckets."""

    MANIFEST = "manifest"
    SPECS = "specs"


class JumpStartLaunchedRegionInfo(JumpStartDataHolderType):
    """Data class for launched region info."""

    __slots__ = ["content_bucket", "region_name"]

    def __init__(self, content_bucket: str, region_name: str):
        """Instantiates JumpStartLaunchedRegionInfo object.

        Args:
            content_bucket (str): Name of JumpStart s3 content bucket associated with region.
            region_name (str): Name of JumpStart launched region.
        """
        self.content_bucket = content_bucket
        self.region_name = region_name


class JumpStartModelHeader(JumpStartDataHolderType):
    """Data class JumpStart model header."""

    __slots__ = ["model_id", "version", "min_version", "spec_key"]

    def __init__(self, header: Dict[str, str]):
        """Initializes a JumpStartModelHeader object from its json representation.

        Args:
            header (Dict[str, str]): Dictionary representation of header.
        """
        self.from_json(header)

    def to_json(self) -> Dict[str, str]:
        """Returns json representation of JumpStartModelHeader object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj

    def from_json(self, json_obj: Dict[str, str]) -> None:
        """Sets fields in object based on json of header.

        Args:
            json_obj (Dict[str, str]): Dictionary representation of header.
        """
        self.model_id: str = json_obj["model_id"]
        self.version: str = json_obj["version"]
        self.min_version: str = json_obj["min_version"]
        self.spec_key: str = json_obj["spec_key"]


class JumpStartECRSpecs(JumpStartDataHolderType):
    """Data class for JumpStart ECR specs."""

    __slots__ = [
        "framework",
        "framework_version",
        "py_version",
        "huggingface_transformers_version",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartECRSpecs object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of spec.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of spec.
        """

        self.framework = json_obj["framework"]
        self.framework_version = json_obj["framework_version"]
        self.py_version = json_obj["py_version"]
        huggingface_transformers_version = json_obj.get("huggingface_transformers_version")
        if huggingface_transformers_version is not None:
            self.huggingface_transformers_version = huggingface_transformers_version

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartECRSpecs object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartHyperparameter(JumpStartDataHolderType):
    """Data class for JumpStart hyperparameter definition in the training container."""

    __slots__ = [
        "name",
        "type",
        "options",
        "default",
        "scope",
        "min",
        "max",
        "exclusive_min",
        "exclusive_max",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartHyperparameter object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of hyperparameter.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hyperparameter.
        """

        self.name = json_obj["name"]
        self.type = json_obj["type"]
        self.default = json_obj["default"]
        self.scope = json_obj["scope"]

        options = json_obj.get("options")
        if options is not None:
            self.options = options

        min_val = json_obj.get("min")
        if min_val is not None:
            self.min = min_val

        max_val = json_obj.get("max")
        if max_val is not None:
            self.max = max_val

        exclusive_min_val = json_obj.get("exclusive_min")
        if exclusive_min_val is not None:
            self.exclusive_min = exclusive_min_val

        exclusive_max_val = json_obj.get("exclusive_max")
        if exclusive_max_val is not None:
            self.exclusive_max = exclusive_max_val

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartHyperparameter object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartEnvironmentVariable(JumpStartDataHolderType):
    """Data class for JumpStart environment variable definitions in the hosting container."""

    __slots__ = [
        "name",
        "type",
        "default",
        "scope",
        "required_for_model_class",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartEnvironmentVariable object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of environment variable.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of environment variable.
        """

        self.name = json_obj["name"]
        self.type = json_obj["type"]
        self.default = json_obj["default"]
        self.scope = json_obj["scope"]
        self.required_for_model_class: bool = json_obj.get("required_for_model_class", False)

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartEnvironmentVariable object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartPredictorSpecs(JumpStartDataHolderType):
    """Data class for JumpStart Predictor specs."""

    __slots__ = [
        "default_content_type",
        "supported_content_types",
        "default_accept_type",
        "supported_accept_types",
    ]

    def __init__(self, spec: Optional[Dict[str, Any]]):
        """Initializes a JumpStartPredictorSpecs object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of predictor specs.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Optional[Dict[str, Any]]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of predictor specs.
        """

        if json_obj is None:
            return

        self.default_content_type = json_obj["default_content_type"]
        self.supported_content_types = json_obj["supported_content_types"]
        self.default_accept_type = json_obj["default_accept_type"]
        self.supported_accept_types = json_obj["supported_accept_types"]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartPredictorSpecs object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartModelSpecs(JumpStartDataHolderType):
    """Data class JumpStart model specs."""

    __slots__ = [
        "model_id",
        "url",
        "version",
        "min_sdk_version",
        "incremental_training_supported",
        "hosting_ecr_specs",
        "hosting_artifact_key",
        "hosting_script_key",
        "training_supported",
        "training_ecr_specs",
        "training_artifact_key",
        "training_script_key",
        "hyperparameters",
        "inference_environment_variables",
        "inference_vulnerable",
        "inference_dependencies",
        "inference_vulnerabilities",
        "training_vulnerable",
        "training_dependencies",
        "training_vulnerabilities",
        "deprecated",
        "deprecated_message",
        "deprecate_warn_message",
        "default_inference_instance_type",
        "supported_inference_instance_types",
        "default_training_instance_type",
        "supported_training_instance_types",
        "metrics",
        "training_prepacked_script_key",
        "hosting_prepacked_artifact_key",
        "model_kwargs",
        "deploy_kwargs",
        "estimator_kwargs",
        "fit_kwargs",
        "predictor_specs",
        "inference_volume_size",
        "training_volume_size",
        "inference_enable_network_isolation",
        "training_enable_network_isolation",
        "resource_name_base",
        "hosting_eula_key",
        "hosting_model_package_arns",
        "training_model_package_artifact_uris",
        "hosting_use_script_uri",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartModelSpecs object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of spec.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json of header.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of spec.
        """
        self.model_id: str = json_obj["model_id"]
        self.url: str = json_obj["url"]
        self.version: str = json_obj["version"]
        self.min_sdk_version: str = json_obj["min_sdk_version"]
        self.incremental_training_supported: bool = bool(json_obj["incremental_training_supported"])
        self.hosting_ecr_specs: JumpStartECRSpecs = JumpStartECRSpecs(json_obj["hosting_ecr_specs"])
        self.hosting_artifact_key: str = json_obj["hosting_artifact_key"]
        self.hosting_script_key: str = json_obj["hosting_script_key"]
        self.training_supported: bool = bool(json_obj["training_supported"])
        self.inference_environment_variables = [
            JumpStartEnvironmentVariable(env_variable)
            for env_variable in json_obj["inference_environment_variables"]
        ]
        self.inference_vulnerable: bool = bool(json_obj["inference_vulnerable"])
        self.inference_dependencies: List[str] = json_obj["inference_dependencies"]
        self.inference_vulnerabilities: List[str] = json_obj["inference_vulnerabilities"]
        self.training_vulnerable: bool = bool(json_obj["training_vulnerable"])
        self.training_dependencies: List[str] = json_obj["training_dependencies"]
        self.training_vulnerabilities: List[str] = json_obj["training_vulnerabilities"]
        self.deprecated: bool = bool(json_obj["deprecated"])
        self.deprecated_message: Optional[str] = json_obj.get("deprecated_message")
        self.deprecate_warn_message: Optional[str] = json_obj.get("deprecate_warn_message")
        self.default_inference_instance_type: Optional[str] = json_obj.get(
            "default_inference_instance_type"
        )
        self.default_training_instance_type: Optional[str] = json_obj.get(
            "default_training_instance_type"
        )
        self.supported_inference_instance_types: Optional[List[str]] = json_obj.get(
            "supported_inference_instance_types"
        )
        self.supported_training_instance_types: Optional[List[str]] = json_obj.get(
            "supported_training_instance_types"
        )
        self.metrics: Optional[List[Dict[str, str]]] = json_obj.get("metrics", None)
        self.training_prepacked_script_key: Optional[str] = json_obj.get(
            "training_prepacked_script_key", None
        )
        self.hosting_prepacked_artifact_key: Optional[str] = json_obj.get(
            "hosting_prepacked_artifact_key", None
        )
        self.model_kwargs = deepcopy(json_obj.get("model_kwargs", {}))
        self.deploy_kwargs = deepcopy(json_obj.get("deploy_kwargs", {}))
        self.predictor_specs: Optional[JumpStartPredictorSpecs] = (
            JumpStartPredictorSpecs(json_obj["predictor_specs"])
            if "predictor_specs" in json_obj
            else None
        )
        self.inference_volume_size: Optional[int] = json_obj.get("inference_volume_size")
        self.inference_enable_network_isolation: bool = json_obj.get(
            "inference_enable_network_isolation", False
        )
        self.resource_name_base: bool = json_obj.get("resource_name_base")

        self.hosting_eula_key: Optional[str] = json_obj.get("hosting_eula_key")

        self.hosting_model_package_arns: Optional[Dict] = json_obj.get("hosting_model_package_arns")
        self.hosting_use_script_uri: bool = json_obj.get("hosting_use_script_uri", True)

        if self.training_supported:
            self.training_ecr_specs: JumpStartECRSpecs = JumpStartECRSpecs(
                json_obj["training_ecr_specs"]
            )
            self.training_artifact_key: str = json_obj["training_artifact_key"]
            self.training_script_key: str = json_obj["training_script_key"]
            hyperparameters: Any = json_obj.get("hyperparameters")
            self.hyperparameters: List[JumpStartHyperparameter] = []
            if hyperparameters is not None:
                self.hyperparameters.extend(
                    [JumpStartHyperparameter(hyperparameter) for hyperparameter in hyperparameters]
                )
            self.estimator_kwargs = deepcopy(json_obj.get("estimator_kwargs", {}))
            self.fit_kwargs = deepcopy(json_obj.get("fit_kwargs", {}))
            self.training_volume_size: Optional[int] = json_obj.get("training_volume_size")
            self.training_enable_network_isolation: bool = json_obj.get(
                "training_enable_network_isolation", False
            )
            self.training_model_package_artifact_uris: Optional[Dict] = json_obj.get(
                "training_model_package_artifact_uris"
            )

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartModelSpecs object."""
        json_obj = {}
        for att in self.__slots__:
            if hasattr(self, att):
                cur_val = getattr(self, att)
                if issubclass(type(cur_val), JumpStartDataHolderType):
                    json_obj[att] = cur_val.to_json()
                elif isinstance(cur_val, list):
                    json_obj[att] = []
                    for obj in cur_val:
                        if issubclass(type(obj), JumpStartDataHolderType):
                            json_obj[att].append(obj.to_json())
                        else:
                            json_obj[att].append(obj)
                else:
                    json_obj[att] = cur_val
        return json_obj

    def supports_prepacked_inference(self) -> bool:
        """Returns True if the model has a prepacked inference artifact."""
        return getattr(self, "hosting_prepacked_artifact_key", None) is not None

    def use_inference_script_uri(self) -> bool:
        """Returns True if the model should use a script uri when deploying inference model."""
        if self.supports_prepacked_inference():
            return False
        return self.hosting_use_script_uri

    def use_training_model_artifact(self) -> bool:
        """Returns True if the model should use a model uri when kicking off training job."""
        return (
            self.training_model_package_artifact_uris is None
            or len(self.training_model_package_artifact_uris) == 0
        )

    def supports_incremental_training(self) -> bool:
        """Returns True if the model supports incremental training."""
        return self.incremental_training_supported


class JumpStartVersionedModelId(JumpStartDataHolderType):
    """Data class for versioned model IDs."""

    __slots__ = ["model_id", "version"]

    def __init__(
        self,
        model_id: str,
        version: str,
    ) -> None:
        """Instantiates JumpStartVersionedModelId object.

        Args:
            model_id (str): JumpStart model ID.
            version (str): JumpStart model version.
        """
        self.model_id = model_id
        self.version = version


class JumpStartCachedS3ContentKey(JumpStartDataHolderType):
    """Data class for the s3 cached content keys."""

    __slots__ = ["file_type", "s3_key"]

    def __init__(
        self,
        file_type: JumpStartS3FileType,
        s3_key: str,
    ) -> None:
        """Instantiates JumpStartCachedS3ContentKey object.

        Args:
            file_type (JumpStartS3FileType): JumpStart file type.
            s3_key (str): object key in s3.
        """
        self.file_type = file_type
        self.s3_key = s3_key


class JumpStartCachedS3ContentValue(JumpStartDataHolderType):
    """Data class for the s3 cached content values."""

    __slots__ = ["formatted_content", "md5_hash"]

    def __init__(
        self,
        formatted_content: Union[
            Dict[JumpStartVersionedModelId, JumpStartModelHeader],
            JumpStartModelSpecs,
        ],
        md5_hash: Optional[str] = None,
    ) -> None:
        """Instantiates JumpStartCachedS3ContentValue object.

        Args:
            formatted_content (Union[Dict[JumpStartVersionedModelId, JumpStartModelHeader],
            JumpStartModelSpecs]):
                Formatted content for model specs and mappings from
                versioned model IDs to specs.
            md5_hash (str): md5_hash for stored file content from s3.
        """
        self.formatted_content = formatted_content
        self.md5_hash = md5_hash


class JumpStartKwargs(JumpStartDataHolderType):
    """Data class for JumpStart object kwargs."""

    SERIALIZATION_EXCLUSION_SET: Set[str] = set()

    def to_kwargs_dict(self):
        """Serializes object to dictionary to be used for kwargs for method arguments."""
        kwargs_dict = {}
        for field in self.__slots__:
            if field not in self.SERIALIZATION_EXCLUSION_SET:
                att_value = getattr(self, field)
                if att_value is not None:
                    kwargs_dict[field] = getattr(self, field)
        return kwargs_dict


class JumpStartModelInitKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartModel.__init__` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "instance_type",
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "region",
        "image_uri",
        "model_data",
        "source_dir",
        "entry_point",
        "env",
        "predictor_cls",
        "role",
        "name",
        "vpc_config",
        "sagemaker_session",
        "enable_network_isolation",
        "model_kms_key",
        "image_config",
        "code_location",
        "container_log_level",
        "dependencies",
        "git_config",
        "model_package_arn",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "instance_type",
        "model_id",
        "model_version",
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "region",
        "model_package_arn",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        region: Optional[str] = None,
        instance_type: Optional[str] = None,
        image_uri: Optional[Union[str, Any]] = None,
        model_data: Optional[Union[str, Any]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, Any]]] = None,
        name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, Any]]]] = None,
        sagemaker_session: Optional[Any] = None,
        enable_network_isolation: Union[bool, Any] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, Any]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Optional[Union[int, Any]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        model_package_arn: Optional[str] = None,
    ) -> None:
        """Instantiates JumpStartModelInitKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.instance_type = instance_type
        self.region = region
        self.image_uri = image_uri
        self.model_data = model_data
        self.source_dir = source_dir
        self.entry_point = entry_point
        self.env = env
        self.predictor_cls = predictor_cls
        self.role = role
        self.name = name
        self.vpc_config = vpc_config
        self.sagemaker_session = sagemaker_session
        self.enable_network_isolation = enable_network_isolation
        self.model_kms_key = model_kms_key
        self.image_config = image_config
        self.code_location = code_location
        self.container_log_level = container_log_level
        self.dependencies = dependencies
        self.git_config = git_config
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.model_package_arn = model_package_arn


class JumpStartModelDeployKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartModel.deploy` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "initial_instance_count",
        "instance_type",
        "region",
        "serializer",
        "deserializer",
        "accelerator_type",
        "endpoint_name",
        "tags",
        "kms_key",
        "wait",
        "data_capture_config",
        "async_inference_config",
        "serverless_inference_config",
        "volume_size",
        "model_data_download_timeout",
        "container_startup_health_check_timeout",
        "inference_recommendation_id",
        "explainer_config",
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "sagemaker_session",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "model_id",
        "model_version",
        "region",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "sagemaker_session",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        region: Optional[str] = None,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        serializer: Optional[Any] = None,
        deserializer: Optional[Any] = None,
        accelerator_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        tags: List[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
        wait: Optional[bool] = None,
        data_capture_config: Optional[Any] = None,
        async_inference_config: Optional[Any] = None,
        serverless_inference_config: Optional[Any] = None,
        volume_size: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        inference_recommendation_id: Optional[str] = None,
        explainer_config: Optional[Any] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> None:
        """Instantiates JumpStartModelDeployKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.initial_instance_count = initial_instance_count
        self.instance_type = instance_type
        self.region = region
        self.serializer = serializer
        self.deserializer = deserializer
        self.accelerator_type = accelerator_type
        self.endpoint_name = endpoint_name
        self.tags = tags
        self.kms_key = kms_key
        self.wait = wait
        self.data_capture_config = data_capture_config
        self.async_inference_config = async_inference_config
        self.serverless_inference_config = serverless_inference_config
        self.volume_size = volume_size
        self.model_data_download_timeout = model_data_download_timeout
        self.container_startup_health_check_timeout = container_startup_health_check_timeout
        self.inference_recommendation_id = inference_recommendation_id
        self.explainer_config = explainer_config
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.sagemaker_session = sagemaker_session


class JumpStartEstimatorInitKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartEstimator.__init__` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "instance_type",
        "instance_count",
        "region",
        "image_uri",
        "model_uri",
        "source_dir",
        "entry_point",
        "hyperparameters",
        "metric_definitions",
        "role",
        "keep_alive_period_in_seconds",
        "volume_size",
        "volume_kms_key",
        "max_run",
        "input_mode",
        "output_path",
        "output_kms_key",
        "base_job_name",
        "sagemaker_session",
        "tags",
        "subnets",
        "security_group_ids",
        "model_channel_name",
        "encrypt_inter_container_traffic",
        "use_spot_instances",
        "max_wait",
        "checkpoint_s3_uri",
        "checkpoint_local_path",
        "enable_network_isolation",
        "rules",
        "debugger_hook_config",
        "tensorboard_output_config",
        "enable_sagemaker_metrics",
        "profiler_config",
        "disable_profiler",
        "environment",
        "max_retry_attempts",
        "git_config",
        "container_log_level",
        "code_location",
        "dependencies",
        "instance_groups",
        "training_repository_access_mode",
        "training_repository_credentials_provider_arn",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "container_entry_point",
        "container_arguments",
        "disable_output_compression",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "region",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "model_id",
        "model_version",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        region: Optional[str] = None,
        image_uri: Optional[Union[str, Any]] = None,
        role: Optional[str] = None,
        instance_count: Optional[Union[int, Any]] = None,
        instance_type: Optional[Union[str, Any]] = None,
        keep_alive_period_in_seconds: Optional[Union[int, Any]] = None,
        volume_size: Optional[Union[int, Any]] = None,
        volume_kms_key: Optional[Union[str, Any]] = None,
        max_run: Optional[Union[int, Any]] = None,
        input_mode: Optional[Union[str, Any]] = None,
        output_path: Optional[Union[str, Any]] = None,
        output_kms_key: Optional[Union[str, Any]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Any] = None,
        hyperparameters: Optional[Dict[str, Union[str, Any]]] = None,
        tags: Optional[List[Dict[str, Union[str, Any]]]] = None,
        subnets: Optional[List[Union[str, Any]]] = None,
        security_group_ids: Optional[List[Union[str, Any]]] = None,
        model_uri: Optional[str] = None,
        model_channel_name: Optional[Union[str, Any]] = None,
        metric_definitions: Optional[List[Dict[str, Union[str, Any]]]] = None,
        encrypt_inter_container_traffic: Union[bool, Any] = None,
        use_spot_instances: Optional[Union[bool, Any]] = None,
        max_wait: Optional[Union[int, Any]] = None,
        checkpoint_s3_uri: Optional[Union[str, Any]] = None,
        checkpoint_local_path: Optional[Union[str, Any]] = None,
        enable_network_isolation: Union[bool, Any] = None,
        rules: Optional[List[Any]] = None,
        debugger_hook_config: Optional[Union[Any, bool]] = None,
        tensorboard_output_config: Optional[Any] = None,
        enable_sagemaker_metrics: Optional[Union[bool, Any]] = None,
        profiler_config: Optional[Any] = None,
        disable_profiler: Optional[bool] = None,
        environment: Optional[Dict[str, Union[str, Any]]] = None,
        max_retry_attempts: Optional[Union[int, Any]] = None,
        source_dir: Optional[Union[str, Any]] = None,
        git_config: Optional[Dict[str, str]] = None,
        container_log_level: Optional[Union[int, Any]] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[Union[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        instance_groups: Optional[List[Any]] = None,
        training_repository_access_mode: Optional[Union[str, Any]] = None,
        training_repository_credentials_provider_arn: Optional[Union[str, Any]] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        container_entry_point: Optional[List[str]] = None,
        container_arguments: Optional[List[str]] = None,
        disable_output_compression: Optional[bool] = None,
    ) -> None:
        """Instantiates JumpStartEstimatorInitKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.region = region
        self.image_uri = image_uri
        self.model_uri = model_uri
        self.source_dir = source_dir
        self.entry_point = entry_point
        self.hyperparameters = hyperparameters
        self.metric_definitions = metric_definitions
        self.role = role
        self.keep_alive_period_in_seconds = keep_alive_period_in_seconds
        self.volume_size = volume_size
        self.volume_kms_key = volume_kms_key
        self.max_run = max_run
        self.input_mode = input_mode
        self.output_path = output_path
        self.output_kms_key = output_kms_key
        self.base_job_name = base_job_name
        self.sagemaker_session = sagemaker_session
        self.tags = tags
        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.model_channel_name = model_channel_name
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        self.use_spot_instances = use_spot_instances
        self.max_wait = max_wait
        self.checkpoint_s3_uri = checkpoint_s3_uri
        self.checkpoint_local_path = checkpoint_local_path
        self.enable_network_isolation = enable_network_isolation
        self.rules = rules
        self.debugger_hook_config = debugger_hook_config
        self.tensorboard_output_config = tensorboard_output_config
        self.enable_sagemaker_metrics = enable_sagemaker_metrics
        self.profiler_config = profiler_config
        self.disable_profiler = disable_profiler
        self.environment = environment
        self.max_retry_attempts = max_retry_attempts
        self.git_config = git_config
        self.container_log_level = container_log_level
        self.code_location = code_location
        self.dependencies = dependencies
        self.instance_groups = instance_groups
        self.training_repository_access_mode = training_repository_access_mode
        self.training_repository_credentials_provider_arn = (
            training_repository_credentials_provider_arn
        )
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.container_entry_point = container_entry_point
        self.container_arguments = container_arguments
        self.disable_output_compression = disable_output_compression


class JumpStartEstimatorFitKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartEstimator.fit` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "region",
        "inputs",
        "wait",
        "logs",
        "job_name",
        "experiment_config",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "sagemaker_session",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "model_id",
        "model_version",
        "region",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "sagemaker_session",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        region: Optional[str] = None,
        inputs: Optional[Union[str, Dict, Any, Any]] = None,
        wait: Optional[bool] = None,
        logs: Optional[str] = None,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> None:
        """Instantiates JumpStartEstimatorInitKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.region = region
        self.inputs = inputs
        self.wait = wait
        self.logs = logs
        self.job_name = job_name
        self.experiment_config = experiment_config
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.sagemaker_session = sagemaker_session


class JumpStartEstimatorDeployKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartEstimator.deploy` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "instance_type",
        "initial_instance_count",
        "region",
        "image_uri",
        "source_dir",
        "entry_point",
        "env",
        "predictor_cls",
        "serializer",
        "deserializer",
        "accelerator_type",
        "endpoint_name",
        "tags",
        "kms_key",
        "wait",
        "data_capture_config",
        "async_inference_config",
        "serverless_inference_config",
        "volume_size",
        "model_data_download_timeout",
        "container_startup_health_check_timeout",
        "inference_recommendation_id",
        "explainer_config",
        "role",
        "vpc_config",
        "sagemaker_session",
        "enable_network_isolation",
        "model_kms_key",
        "image_config",
        "code_location",
        "container_log_level",
        "dependencies",
        "git_config",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "model_name",
        "use_compiled_model",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "region",
        "model_id",
        "model_version",
        "sagemaker_session",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        region: Optional[str] = None,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        serializer: Optional[Any] = None,
        deserializer: Optional[Any] = None,
        accelerator_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        tags: List[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
        wait: Optional[bool] = None,
        data_capture_config: Optional[Any] = None,
        async_inference_config: Optional[Any] = None,
        serverless_inference_config: Optional[Any] = None,
        volume_size: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        inference_recommendation_id: Optional[str] = None,
        explainer_config: Optional[Any] = None,
        image_uri: Optional[Union[str, Any]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, Any]]] = None,
        model_name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, Any]]]] = None,
        sagemaker_session: Optional[Any] = None,
        enable_network_isolation: Union[bool, Any] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, Any]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Optional[Union[int, Any]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        use_compiled_model: bool = False,
    ) -> None:
        """Instantiates JumpStartEstimatorInitKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.instance_type = instance_type
        self.initial_instance_count = initial_instance_count
        self.region = region
        self.image_uri = image_uri
        self.source_dir = source_dir
        self.entry_point = entry_point
        self.env = env
        self.predictor_cls = predictor_cls
        self.serializer = serializer
        self.deserializer = deserializer
        self.accelerator_type = accelerator_type
        self.endpoint_name = endpoint_name
        self.tags = tags
        self.kms_key = kms_key
        self.wait = wait
        self.data_capture_config = data_capture_config
        self.async_inference_config = async_inference_config
        self.serverless_inference_config = serverless_inference_config
        self.volume_size = volume_size
        self.model_data_download_timeout = model_data_download_timeout
        self.container_startup_health_check_timeout = container_startup_health_check_timeout
        self.inference_recommendation_id = inference_recommendation_id
        self.explainer_config = explainer_config
        self.role = role
        self.model_name = model_name
        self.vpc_config = vpc_config
        self.sagemaker_session = sagemaker_session
        self.enable_network_isolation = enable_network_isolation
        self.model_kms_key = model_kms_key
        self.image_config = image_config
        self.code_location = code_location
        self.container_log_level = container_log_level
        self.dependencies = dependencies
        self.git_config = git_config
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.use_compiled_model = use_compiled_model
