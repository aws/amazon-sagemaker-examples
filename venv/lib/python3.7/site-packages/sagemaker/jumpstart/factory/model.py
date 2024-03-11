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
"""This module stores JumpStart Model factory methods."""
from __future__ import absolute_import


from typing import Any, Dict, List, Optional, Union
from sagemaker import environment_variables, image_uris, instance_types, model_uris, script_uris
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.jumpstart.artifacts import (
    _model_supports_inference_script_uri,
    _retrieve_model_init_kwargs,
    _retrieve_model_deploy_kwargs,
    _retrieve_model_package_arn,
)
from sagemaker.jumpstart.artifacts.resource_names import _retrieve_resource_name_base
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    INFERENCE_ENTRY_POINT_SCRIPT_NAME,
    JUMPSTART_DEFAULT_REGION_NAME,
    JUMPSTART_LOGGER,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.types import (
    JumpStartModelDeployKwargs,
    JumpStartModelInitKwargs,
)
from sagemaker.jumpstart.utils import (
    update_dict_if_key_not_present,
    resolve_model_sagemaker_config_field,
)

from sagemaker.model_monitor.data_capture_config import DataCaptureConfig
from sagemaker.base_predictor import Predictor
from sagemaker import accept_types, content_types, serializers, deserializers

from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.session import Session
from sagemaker.utils import name_from_base
from sagemaker.workflow.entities import PipelineVariable


def get_default_predictor(
    predictor: Predictor,
    model_id: str,
    model_version: str,
    region: str,
    tolerate_vulnerable_model: bool,
    tolerate_deprecated_model: bool,
    sagemaker_session: Session,
) -> Predictor:
    """Converts predictor returned from ``Model.deploy()`` into a JumpStart-specific one.

    Raises:
        RuntimeError: If a base-class predictor is not used.
    """

    # if there's a non-default predictor, do not mutate -- return as is
    if type(predictor) != Predictor:  # pylint: disable=C0123
        raise RuntimeError(
            "Can only get default predictor from base Predictor class. "
            f"Using Predictor class '{type(predictor).__name__}'."
        )

    predictor.serializer = serializers.retrieve_default(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
    )
    predictor.deserializer = deserializers.retrieve_default(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
    )
    predictor.accept = accept_types.retrieve_default(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
    )
    predictor.content_type = content_types.retrieve_default(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
    )

    return predictor


def _add_region_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets region kwargs based on default or override, returns full kwargs."""

    kwargs.region = kwargs.region or JUMPSTART_DEFAULT_REGION_NAME

    return kwargs


def _add_sagemaker_session_to_kwargs(
    kwargs: Union[JumpStartModelInitKwargs, JumpStartModelDeployKwargs]
) -> JumpStartModelInitKwargs:
    """Sets session in kwargs based on default or override, returns full kwargs."""
    kwargs.sagemaker_session = kwargs.sagemaker_session or DEFAULT_JUMPSTART_SAGEMAKER_SESSION
    return kwargs


def _add_role_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets role based on default or override, returns full kwargs."""

    kwargs.role = resolve_model_sagemaker_config_field(
        field_name="role",
        field_val=kwargs.role,
        sagemaker_session=kwargs.sagemaker_session,
        default_value=kwargs.role,
    )

    return kwargs


def _add_model_version_to_kwargs(
    kwargs: JumpStartModelInitKwargs,
) -> JumpStartModelInitKwargs:
    """Sets model version based on default or override, returns full kwargs."""

    kwargs.model_version = kwargs.model_version or "*"

    return kwargs


def _add_vulnerable_and_deprecated_status_to_kwargs(
    kwargs: JumpStartModelInitKwargs,
) -> JumpStartModelInitKwargs:
    """Sets deprecated and vulnerability check status, returns full kwargs."""

    kwargs.tolerate_deprecated_model = kwargs.tolerate_deprecated_model or False
    kwargs.tolerate_vulnerable_model = kwargs.tolerate_vulnerable_model or False

    return kwargs


def _add_instance_type_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets instance type based on default or override, returns full kwargs."""

    orig_instance_type = kwargs.instance_type

    kwargs.instance_type = kwargs.instance_type or instance_types.retrieve_default(
        region=kwargs.region,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        scope=JumpStartScriptScope.INFERENCE,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    if orig_instance_type is None:
        JUMPSTART_LOGGER.info(
            "No instance type selected for inference hosting endpoint. Defaulting to %s.",
            kwargs.instance_type,
        )

    return kwargs


def _add_image_uri_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets image uri based on default or override, returns full kwargs."""

    kwargs.image_uri = kwargs.image_uri or image_uris.retrieve(
        region=kwargs.region,
        framework=None,
        image_scope=JumpStartScriptScope.INFERENCE,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        instance_type=kwargs.instance_type,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    return kwargs


def _add_model_data_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets model data based on default or override, returns full kwargs."""

    model_data = kwargs.model_data

    kwargs.model_data = model_data or model_uris.retrieve(
        model_scope=JumpStartScriptScope.INFERENCE,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    return kwargs


def _add_source_dir_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets source dir based on default or override, returns full kwargs."""

    source_dir = kwargs.source_dir

    if _model_supports_inference_script_uri(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    ):
        source_dir = source_dir or script_uris.retrieve(
            script_scope=JumpStartScriptScope.INFERENCE,
            model_id=kwargs.model_id,
            model_version=kwargs.model_version,
            region=kwargs.region,
            tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
            tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
            sagemaker_session=kwargs.sagemaker_session,
        )

    kwargs.source_dir = source_dir

    return kwargs


def _add_entry_point_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets entry point based on default or override, returns full kwargs."""

    entry_point = kwargs.entry_point

    if _model_supports_inference_script_uri(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    ):

        entry_point = entry_point or INFERENCE_ENTRY_POINT_SCRIPT_NAME

    kwargs.entry_point = entry_point

    return kwargs


def _add_env_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets env based on default or override, returns full kwargs."""

    env = kwargs.env

    if env is None:
        env = {}

    extra_env_vars = environment_variables.retrieve_default(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        include_aws_sdk_env_vars=False,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    for key, value in extra_env_vars.items():
        update_dict_if_key_not_present(
            env,
            key,
            value,
        )

    if env == {}:
        env = None

    kwargs.env = env

    return kwargs


def _add_model_package_arn_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets model package arn based on default or override, returns full kwargs."""

    model_package_arn = kwargs.model_package_arn or _retrieve_model_package_arn(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        scope=JumpStartScriptScope.INFERENCE,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    kwargs.model_package_arn = model_package_arn
    return kwargs


def _add_extra_model_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    model_kwargs_to_add = _retrieve_model_init_kwargs(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    for key, value in model_kwargs_to_add.items():
        if getattr(kwargs, key) is None:
            resolved_value = resolve_model_sagemaker_config_field(
                field_name=key,
                field_val=value,
                sagemaker_session=kwargs.sagemaker_session,
            )
            setattr(kwargs, key, resolved_value)

    return kwargs


def _add_predictor_cls_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets predictor class based on default or override, returns full kwargs."""

    predictor_cls = kwargs.predictor_cls or Predictor

    kwargs.predictor_cls = predictor_cls
    return kwargs


def _add_endpoint_name_to_kwargs(
    kwargs: Optional[JumpStartModelDeployKwargs],
) -> JumpStartModelDeployKwargs:
    """Sets resource name based on default or override, returns full kwargs."""

    default_endpoint_name = _retrieve_resource_name_base(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    kwargs.endpoint_name = kwargs.endpoint_name or (
        name_from_base(default_endpoint_name) if default_endpoint_name is not None else None
    )

    return kwargs


def _add_model_name_to_kwargs(
    kwargs: Optional[JumpStartModelInitKwargs],
) -> JumpStartModelInitKwargs:
    """Sets resource name based on default or override, returns full kwargs."""

    default_model_name = _retrieve_resource_name_base(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    kwargs.name = kwargs.name or (
        name_from_base(default_model_name) if default_model_name is not None else None
    )

    return kwargs


def _add_deploy_extra_kwargs(kwargs: JumpStartModelInitKwargs) -> Dict[str, Any]:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    deploy_kwargs_to_add = _retrieve_model_deploy_kwargs(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        instance_type=kwargs.instance_type,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    for key, value in deploy_kwargs_to_add.items():
        if getattr(kwargs, key) is None:
            setattr(kwargs, key, value)

    return kwargs


def get_deploy_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    region: Optional[str] = None,
    initial_instance_count: Optional[int] = None,
    instance_type: Optional[str] = None,
    serializer: Optional[BaseSerializer] = None,
    deserializer: Optional[BaseDeserializer] = None,
    accelerator_type: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    tags: List[Dict[str, str]] = None,
    kms_key: Optional[str] = None,
    wait: Optional[bool] = None,
    data_capture_config: Optional[DataCaptureConfig] = None,
    async_inference_config: Optional[AsyncInferenceConfig] = None,
    serverless_inference_config: Optional[ServerlessInferenceConfig] = None,
    volume_size: Optional[int] = None,
    model_data_download_timeout: Optional[int] = None,
    container_startup_health_check_timeout: Optional[int] = None,
    inference_recommendation_id: Optional[str] = None,
    explainer_config: Optional[ExplainerConfig] = None,
    tolerate_vulnerable_model: Optional[bool] = None,
    tolerate_deprecated_model: Optional[bool] = None,
    sagemaker_session: Optional[Session] = None,
) -> JumpStartModelDeployKwargs:
    """Returns kwargs required to call `deploy` on `sagemaker.estimator.Model` object."""

    deploy_kwargs: JumpStartModelDeployKwargs = JumpStartModelDeployKwargs(
        model_id=model_id,
        model_version=model_version,
        region=region,
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        serializer=serializer,
        deserializer=deserializer,
        accelerator_type=accelerator_type,
        endpoint_name=endpoint_name,
        tags=tags,
        kms_key=kms_key,
        wait=wait,
        data_capture_config=data_capture_config,
        async_inference_config=async_inference_config,
        serverless_inference_config=serverless_inference_config,
        volume_size=volume_size,
        model_data_download_timeout=model_data_download_timeout,
        container_startup_health_check_timeout=container_startup_health_check_timeout,
        inference_recommendation_id=inference_recommendation_id,
        explainer_config=explainer_config,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
    )

    deploy_kwargs = _add_sagemaker_session_to_kwargs(kwargs=deploy_kwargs)

    deploy_kwargs = _add_model_version_to_kwargs(kwargs=deploy_kwargs)

    deploy_kwargs = _add_endpoint_name_to_kwargs(kwargs=deploy_kwargs)

    deploy_kwargs = _add_instance_type_to_kwargs(
        kwargs=deploy_kwargs,
    )

    deploy_kwargs.initial_instance_count = initial_instance_count or 1

    deploy_kwargs = _add_deploy_extra_kwargs(kwargs=deploy_kwargs)

    return deploy_kwargs


def get_init_kwargs(
    model_id: str,
    model_from_estimator: bool = False,
    model_version: Optional[str] = None,
    tolerate_vulnerable_model: Optional[bool] = None,
    tolerate_deprecated_model: Optional[bool] = None,
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    image_uri: Optional[Union[str, PipelineVariable]] = None,
    model_data: Optional[Union[str, PipelineVariable]] = None,
    role: Optional[str] = None,
    predictor_cls: Optional[callable] = None,
    env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    name: Optional[str] = None,
    vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None,
    sagemaker_session: Optional[Session] = None,
    enable_network_isolation: Union[bool, PipelineVariable] = None,
    model_kms_key: Optional[str] = None,
    image_config: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    source_dir: Optional[str] = None,
    code_location: Optional[str] = None,
    entry_point: Optional[str] = None,
    container_log_level: Optional[Union[int, PipelineVariable]] = None,
    dependencies: Optional[List[str]] = None,
    git_config: Optional[Dict[str, str]] = None,
    model_package_arn: Optional[str] = None,
) -> JumpStartModelInitKwargs:
    """Returns kwargs required to instantiate `sagemaker.estimator.Model` object."""

    model_init_kwargs: JumpStartModelInitKwargs = JumpStartModelInitKwargs(
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
        region=region,
        image_uri=image_uri,
        model_data=model_data,
        source_dir=source_dir,
        entry_point=entry_point,
        env=env,
        predictor_cls=predictor_cls,
        role=role,
        name=name,
        vpc_config=vpc_config,
        sagemaker_session=sagemaker_session,
        enable_network_isolation=enable_network_isolation,
        model_kms_key=model_kms_key,
        image_config=image_config,
        code_location=code_location,
        container_log_level=container_log_level,
        dependencies=dependencies,
        git_config=git_config,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        model_package_arn=model_package_arn,
    )

    model_init_kwargs = _add_model_version_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_vulnerable_and_deprecated_status_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_region_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_sagemaker_session_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_model_name_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_instance_type_to_kwargs(
        kwargs=model_init_kwargs,
    )

    model_init_kwargs = _add_image_uri_to_kwargs(kwargs=model_init_kwargs)

    # we use the model artifact from the training job output
    if not model_from_estimator:
        model_init_kwargs = _add_model_data_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_source_dir_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_entry_point_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_env_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_predictor_cls_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_extra_model_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_role_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_model_package_arn_to_kwargs(kwargs=model_init_kwargs)

    return model_init_kwargs
