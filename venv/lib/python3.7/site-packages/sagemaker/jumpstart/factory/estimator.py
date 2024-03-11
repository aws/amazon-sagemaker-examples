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
"""This module stores JumpStart Estimator factory methods."""
from __future__ import absolute_import


from typing import Dict, List, Optional, Union
from sagemaker import (
    hyperparameters as hyperparameters_utils,
    image_uris,
    instance_types,
    metric_definitions as metric_definitions_utils,
    model_uris,
    script_uris,
)
from sagemaker.jumpstart.artifacts import (
    _model_supports_incremental_training,
    _retrieve_model_package_model_artifact_s3_uri,
)
from sagemaker.jumpstart.artifacts.resource_names import _retrieve_resource_name_base
from sagemaker.session import Session
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.debugger.debugger import DebuggerHookConfig, RuleBase, TensorBoardOutputConfig
from sagemaker.debugger.profiler_config import ProfilerConfig
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.inputs import FileSystemInput, TrainingInput
from sagemaker.instance_group import InstanceGroup
from sagemaker.jumpstart.artifacts import (
    _retrieve_estimator_init_kwargs,
    _retrieve_estimator_fit_kwargs,
    _model_supports_training_model_uri,
)
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_REGION_NAME,
    JUMPSTART_LOGGER,
    TRAINING_ENTRY_POINT_SCRIPT_NAME,
    SAGEMAKER_GATED_MODEL_S3_URI_TRAINING_ENV_VAR_KEY,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.factory import model
from sagemaker.jumpstart.types import (
    JumpStartEstimatorDeployKwargs,
    JumpStartEstimatorFitKwargs,
    JumpStartEstimatorInitKwargs,
    JumpStartKwargs,
    JumpStartModelDeployKwargs,
    JumpStartModelInitKwargs,
)
from sagemaker.jumpstart.utils import (
    update_dict_if_key_not_present,
    resolve_estimator_sagemaker_config_field,
)


from sagemaker.model_monitor.data_capture_config import DataCaptureConfig
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.utils import name_from_base
from sagemaker.workflow.entities import PipelineVariable


def get_init_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    tolerate_vulnerable_model: Optional[bool] = None,
    tolerate_deprecated_model: Optional[bool] = None,
    region: Optional[str] = None,
    image_uri: Optional[Union[str, PipelineVariable]] = None,
    role: Optional[str] = None,
    instance_count: Optional[Union[int, PipelineVariable]] = None,
    instance_type: Optional[Union[str, PipelineVariable]] = None,
    keep_alive_period_in_seconds: Optional[Union[int, PipelineVariable]] = None,
    volume_size: Optional[Union[int, PipelineVariable]] = None,
    volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
    max_run: Optional[Union[int, PipelineVariable]] = None,
    input_mode: Optional[Union[str, PipelineVariable]] = None,
    output_path: Optional[Union[str, PipelineVariable]] = None,
    output_kms_key: Optional[Union[str, PipelineVariable]] = None,
    base_job_name: Optional[str] = None,
    sagemaker_session: Optional[Session] = None,
    hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
    subnets: Optional[List[Union[str, PipelineVariable]]] = None,
    security_group_ids: Optional[List[Union[str, PipelineVariable]]] = None,
    model_uri: Optional[str] = None,
    model_channel_name: Optional[Union[str, PipelineVariable]] = None,
    metric_definitions: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
    encrypt_inter_container_traffic: Union[bool, PipelineVariable] = None,
    use_spot_instances: Optional[Union[bool, PipelineVariable]] = None,
    max_wait: Optional[Union[int, PipelineVariable]] = None,
    checkpoint_s3_uri: Optional[Union[str, PipelineVariable]] = None,
    checkpoint_local_path: Optional[Union[str, PipelineVariable]] = None,
    enable_network_isolation: Union[bool, PipelineVariable] = None,
    rules: Optional[List[RuleBase]] = None,
    debugger_hook_config: Optional[Union[DebuggerHookConfig, bool]] = None,
    tensorboard_output_config: Optional[TensorBoardOutputConfig] = None,
    enable_sagemaker_metrics: Optional[Union[bool, PipelineVariable]] = None,
    profiler_config: Optional[ProfilerConfig] = None,
    disable_profiler: Optional[bool] = None,
    environment: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    max_retry_attempts: Optional[Union[int, PipelineVariable]] = None,
    source_dir: Optional[Union[str, PipelineVariable]] = None,
    git_config: Optional[Dict[str, str]] = None,
    container_log_level: Optional[Union[int, PipelineVariable]] = None,
    code_location: Optional[str] = None,
    entry_point: Optional[Union[str, PipelineVariable]] = None,
    dependencies: Optional[List[str]] = None,
    instance_groups: Optional[List[InstanceGroup]] = None,
    training_repository_access_mode: Optional[Union[str, PipelineVariable]] = None,
    training_repository_credentials_provider_arn: Optional[Union[str, PipelineVariable]] = None,
    container_entry_point: Optional[List[str]] = None,
    container_arguments: Optional[List[str]] = None,
    disable_output_compression: Optional[bool] = None,
) -> JumpStartEstimatorInitKwargs:
    """Returns kwargs required to instantiate `sagemaker.estimator.Estimator` object."""

    estimator_init_kwargs: JumpStartEstimatorInitKwargs = JumpStartEstimatorInitKwargs(
        model_id=model_id,
        model_version=model_version,
        role=role,
        region=region,
        instance_count=instance_count,
        instance_type=instance_type,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        volume_size=volume_size,
        volume_kms_key=volume_kms_key,
        max_run=max_run,
        input_mode=input_mode,
        output_path=output_path,
        output_kms_key=output_kms_key,
        base_job_name=base_job_name,
        sagemaker_session=sagemaker_session,
        tags=tags,
        subnets=subnets,
        security_group_ids=security_group_ids,
        model_uri=model_uri,
        model_channel_name=model_channel_name,
        metric_definitions=metric_definitions,
        encrypt_inter_container_traffic=encrypt_inter_container_traffic,
        use_spot_instances=use_spot_instances,
        max_wait=max_wait,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        rules=rules,
        debugger_hook_config=debugger_hook_config,
        tensorboard_output_config=tensorboard_output_config,
        enable_sagemaker_metrics=enable_sagemaker_metrics,
        enable_network_isolation=enable_network_isolation,
        profiler_config=profiler_config,
        disable_profiler=disable_profiler,
        environment=environment,
        max_retry_attempts=max_retry_attempts,
        source_dir=source_dir,
        git_config=git_config,
        hyperparameters=hyperparameters,
        container_log_level=container_log_level,
        code_location=code_location,
        entry_point=entry_point,
        dependencies=dependencies,
        instance_groups=instance_groups,
        training_repository_access_mode=training_repository_access_mode,
        training_repository_credentials_provider_arn=training_repository_credentials_provider_arn,
        image_uri=image_uri,
        container_entry_point=container_entry_point,
        container_arguments=container_arguments,
        disable_output_compression=disable_output_compression,
    )

    estimator_init_kwargs = _add_model_version_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_vulnerable_and_deprecated_status_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_region_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_sagemaker_session_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_instance_type_and_count_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_image_uri_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_model_uri_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_source_dir_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_entry_point_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_hyperparameters_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_metric_definitions_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_estimator_extra_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_role_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_env_to_kwargs(estimator_init_kwargs)

    return estimator_init_kwargs


def get_fit_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    region: Optional[str] = None,
    inputs: Optional[Union[str, Dict, TrainingInput, FileSystemInput]] = None,
    wait: Optional[bool] = None,
    logs: Optional[str] = None,
    job_name: Optional[str] = None,
    experiment_config: Optional[Dict[str, str]] = None,
    tolerate_vulnerable_model: Optional[bool] = None,
    tolerate_deprecated_model: Optional[bool] = None,
    sagemaker_session: Optional[Session] = None,
) -> JumpStartEstimatorFitKwargs:
    """Returns kwargs required call `fit` on `sagemaker.estimator.Estimator` object."""

    estimator_fit_kwargs: JumpStartEstimatorFitKwargs = JumpStartEstimatorFitKwargs(
        model_id=model_id,
        model_version=model_version,
        region=region,
        inputs=inputs,
        wait=wait,
        logs=logs,
        job_name=job_name,
        experiment_config=experiment_config,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
    )

    estimator_fit_kwargs = _add_model_version_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_region_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_training_job_name_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_fit_extra_kwargs(estimator_fit_kwargs)

    return estimator_fit_kwargs


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
    image_uri: Optional[Union[str, PipelineVariable]] = None,
    role: Optional[str] = None,
    predictor_cls: Optional[callable] = None,
    env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
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
    tolerate_deprecated_model: Optional[bool] = None,
    tolerate_vulnerable_model: Optional[bool] = None,
    use_compiled_model: Optional[bool] = None,
    model_name: Optional[str] = None,
) -> JumpStartEstimatorDeployKwargs:
    """Returns kwargs required to call `deploy` on `sagemaker.estimator.Estimator` object."""

    model_deploy_kwargs: JumpStartModelDeployKwargs = model.get_deploy_kwargs(
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
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    model_init_kwargs: JumpStartModelInitKwargs = model.get_init_kwargs(
        model_id=model_id,
        model_from_estimator=True,
        model_version=model_version,
        instance_type=model_deploy_kwargs.instance_type,  # prevent excess logging
        region=region,
        image_uri=image_uri,
        source_dir=source_dir,
        entry_point=entry_point,
        env=env,
        predictor_cls=predictor_cls,
        role=role,
        name=model_name,
        vpc_config=vpc_config,
        sagemaker_session=model_deploy_kwargs.sagemaker_session,
        enable_network_isolation=enable_network_isolation,
        model_kms_key=model_kms_key,
        image_config=image_config,
        code_location=code_location,
        container_log_level=container_log_level,
        dependencies=dependencies,
        git_config=git_config,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    estimator_deploy_kwargs: JumpStartEstimatorDeployKwargs = JumpStartEstimatorDeployKwargs(
        model_id=model_init_kwargs.model_id,
        model_version=model_init_kwargs.model_version,
        instance_type=model_init_kwargs.instance_type,
        initial_instance_count=model_deploy_kwargs.initial_instance_count,
        region=model_init_kwargs.region,
        image_uri=model_init_kwargs.image_uri,
        source_dir=model_init_kwargs.source_dir,
        entry_point=model_init_kwargs.entry_point,
        env=model_init_kwargs.env,
        predictor_cls=model_init_kwargs.predictor_cls,
        serializer=model_deploy_kwargs.serializer,
        deserializer=model_deploy_kwargs.deserializer,
        accelerator_type=model_deploy_kwargs.accelerator_type,
        endpoint_name=model_deploy_kwargs.endpoint_name,
        tags=model_deploy_kwargs.tags,
        kms_key=model_deploy_kwargs.kms_key,
        wait=model_deploy_kwargs.wait,
        data_capture_config=model_deploy_kwargs.data_capture_config,
        async_inference_config=model_deploy_kwargs.async_inference_config,
        serverless_inference_config=model_deploy_kwargs.serverless_inference_config,
        volume_size=model_deploy_kwargs.volume_size,
        model_data_download_timeout=model_deploy_kwargs.model_data_download_timeout,
        container_startup_health_check_timeout=(
            model_deploy_kwargs.container_startup_health_check_timeout
        ),
        inference_recommendation_id=model_deploy_kwargs.inference_recommendation_id,
        explainer_config=model_deploy_kwargs.explainer_config,
        role=model_init_kwargs.role,
        model_name=model_init_kwargs.name,
        vpc_config=model_init_kwargs.vpc_config,
        sagemaker_session=model_init_kwargs.sagemaker_session,
        enable_network_isolation=model_init_kwargs.enable_network_isolation,
        model_kms_key=model_init_kwargs.model_kms_key,
        image_config=model_init_kwargs.image_config,
        code_location=model_init_kwargs.code_location,
        container_log_level=model_init_kwargs.container_log_level,
        dependencies=model_init_kwargs.dependencies,
        git_config=model_init_kwargs.git_config,
        tolerate_vulnerable_model=model_init_kwargs.tolerate_vulnerable_model,
        tolerate_deprecated_model=model_init_kwargs.tolerate_deprecated_model,
        use_compiled_model=use_compiled_model,
    )

    return estimator_deploy_kwargs


def _add_region_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets region in kwargs based on default or override, returns full kwargs."""
    kwargs.region = kwargs.region or JUMPSTART_DEFAULT_REGION_NAME
    return kwargs


def _add_sagemaker_session_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets session in kwargs based on default or override, returns full kwargs."""
    kwargs.sagemaker_session = kwargs.sagemaker_session or DEFAULT_JUMPSTART_SAGEMAKER_SESSION
    return kwargs


def _add_model_version_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets model version in kwargs based on default or override, returns full kwargs."""

    kwargs.model_version = kwargs.model_version or "*"

    return kwargs


def _add_role_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets role based on default or override, returns full kwargs."""

    kwargs.role = resolve_estimator_sagemaker_config_field(
        field_name="role",
        field_val=kwargs.role,
        sagemaker_session=kwargs.sagemaker_session,
        default_value=kwargs.role,
    )

    return kwargs


def _add_instance_type_and_count_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets instance type and count in kwargs based on default or override, returns full kwargs."""

    orig_instance_type = kwargs.instance_type

    kwargs.instance_type = kwargs.instance_type or instance_types.retrieve_default(
        region=kwargs.region,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        scope=JumpStartScriptScope.TRAINING,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    kwargs.instance_count = kwargs.instance_count or 1

    if orig_instance_type is None:
        JUMPSTART_LOGGER.info(
            "No instance type selected for training job. Defaulting to %s.", kwargs.instance_type
        )

    return kwargs


def _add_image_uri_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets image uri in kwargs based on default or override, returns full kwargs."""

    kwargs.image_uri = kwargs.image_uri or image_uris.retrieve(
        region=kwargs.region,
        framework=None,
        image_scope=JumpStartScriptScope.TRAINING,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        instance_type=kwargs.instance_type,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    return kwargs


def _add_model_uri_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets model uri in kwargs based on default or override, returns full kwargs."""

    if _model_supports_training_model_uri(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    ):
        default_model_uri = model_uris.retrieve(
            model_scope=JumpStartScriptScope.TRAINING,
            model_id=kwargs.model_id,
            model_version=kwargs.model_version,
            tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
            tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
            sagemaker_session=kwargs.sagemaker_session,
        )

        if (
            kwargs.model_uri is not None
            and kwargs.model_uri != default_model_uri
            and not _model_supports_incremental_training(
                model_id=kwargs.model_id,
                model_version=kwargs.model_version,
                region=kwargs.region,
                tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
                tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
                sagemaker_session=kwargs.sagemaker_session,
            )
        ):
            JUMPSTART_LOGGER.warning(
                "'%s' does not support incremental training but is being trained with"
                " non-default model artifact.",
                kwargs.model_id,
            )

        kwargs.model_uri = kwargs.model_uri or default_model_uri

    return kwargs


def _add_vulnerable_and_deprecated_status_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets deprecated and vulnerability check status, returns full kwargs."""

    kwargs.tolerate_deprecated_model = kwargs.tolerate_deprecated_model or False
    kwargs.tolerate_vulnerable_model = kwargs.tolerate_vulnerable_model or False

    return kwargs


def _add_source_dir_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets source dir in kwargs based on default or override, returns full kwargs."""

    kwargs.source_dir = kwargs.source_dir or script_uris.retrieve(
        script_scope=JumpStartScriptScope.TRAINING,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    return kwargs


def _add_env_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets environment in kwargs based on default or override, returns full kwargs."""

    model_package_artifact_uri = _retrieve_model_package_model_artifact_s3_uri(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        scope=JumpStartScriptScope.TRAINING,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    if model_package_artifact_uri:
        if kwargs.environment is None:
            kwargs.environment = {}
        kwargs.environment = {
            **{SAGEMAKER_GATED_MODEL_S3_URI_TRAINING_ENV_VAR_KEY: model_package_artifact_uri},
            **kwargs.environment,
        }

    return kwargs


def _add_entry_point_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets entry point in kwargs based on default or override, returns full kwargs."""

    kwargs.entry_point = kwargs.entry_point or TRAINING_ENTRY_POINT_SCRIPT_NAME

    return kwargs


def _add_training_job_name_to_kwargs(
    kwargs: Optional[JumpStartEstimatorFitKwargs],
) -> JumpStartEstimatorFitKwargs:
    """Sets resource name based on default or override, returns full kwargs."""

    default_training_job_name = _retrieve_resource_name_base(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    kwargs.job_name = kwargs.job_name or (
        name_from_base(default_training_job_name) if default_training_job_name is not None else None
    )

    return kwargs


def _add_hyperparameters_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets hyperparameters in kwargs based on default or override, returns full kwargs."""

    kwargs.hyperparameters = (
        kwargs.hyperparameters.copy() if kwargs.hyperparameters is not None else {}
    )

    default_hyperparameters = hyperparameters_utils.retrieve_default(
        region=kwargs.region,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    for key, value in default_hyperparameters.items():
        kwargs.hyperparameters = update_dict_if_key_not_present(
            kwargs.hyperparameters,
            key,
            value,
        )

    if kwargs.hyperparameters == {}:
        kwargs.hyperparameters = None

    return kwargs


def _add_metric_definitions_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets metric definitions in kwargs based on default or override, returns full kwargs."""

    kwargs.metric_definitions = (
        kwargs.metric_definitions.copy() if kwargs.metric_definitions is not None else []
    )

    default_metric_definitions = (
        metric_definitions_utils.retrieve_default(
            region=kwargs.region,
            model_id=kwargs.model_id,
            model_version=kwargs.model_version,
            tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
            tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
            sagemaker_session=kwargs.sagemaker_session,
        )
        or []
    )

    for metric_definition in default_metric_definitions:
        if metric_definition["Name"] not in {
            definition["Name"] for definition in kwargs.metric_definitions
        }:
            kwargs.metric_definitions.append(metric_definition)

    if kwargs.metric_definitions == []:
        kwargs.metric_definitions = None

    return kwargs


def _add_estimator_extra_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    estimator_kwargs_to_add = _retrieve_estimator_init_kwargs(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        instance_type=kwargs.instance_type,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    for key, value in estimator_kwargs_to_add.items():
        if getattr(kwargs, key) is None:
            resolved_value = resolve_estimator_sagemaker_config_field(
                field_name=key,
                field_val=value,
                sagemaker_session=kwargs.sagemaker_session,
            )
            setattr(kwargs, key, resolved_value)

    return kwargs


def _add_fit_extra_kwargs(kwargs: JumpStartEstimatorFitKwargs) -> JumpStartEstimatorFitKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    fit_kwargs_to_add = _retrieve_estimator_fit_kwargs(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        tolerate_deprecated_model=kwargs.tolerate_deprecated_model,
        tolerate_vulnerable_model=kwargs.tolerate_vulnerable_model,
        sagemaker_session=kwargs.sagemaker_session,
    )

    for key, value in fit_kwargs_to_add.items():
        if getattr(kwargs, key) is None:
            setattr(kwargs, key, value)

    return kwargs
