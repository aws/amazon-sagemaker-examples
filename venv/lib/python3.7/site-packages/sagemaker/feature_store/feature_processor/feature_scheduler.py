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
"""Feature Processor schedule APIs."""
from __future__ import absolute_import
import logging
import json
import re
from datetime import datetime
from typing import Callable, List, Optional, Dict, Sequence, Union, Any, Tuple

import pytz
from botocore.exceptions import ClientError

from sagemaker.feature_store.feature_processor._config_uploader import ConfigUploader
from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode

# pylint: disable=C0301
from sagemaker.feature_store.feature_processor.lineage._feature_processor_lineage_name_helper import (
    _get_feature_group_lineage_context_name,
    _get_feature_group_pipeline_lineage_context_name,
    _get_feature_group_pipeline_version_lineage_context_name,
    _get_feature_processor_pipeline_lineage_context_name,
    _get_feature_processor_pipeline_version_lineage_context_name,
)
from sagemaker.lineage import context
from sagemaker.lineage._utils import get_resource_name_from_arn
from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentManager,
)

from sagemaker.remote_function.spark_config import SparkConfig
from sagemaker.vpc_utils import SUBNETS_KEY, SECURITY_GROUP_IDS_KEY

from sagemaker.feature_store.feature_processor._constants import (
    EXECUTION_TIME_PIPELINE_PARAMETER,
    EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT,
    RESOURCE_NOT_FOUND_EXCEPTION,
    SPARK_JAR_FILES_PATH,
    SPARK_PY_FILES_PATH,
    SPARK_FILES_PATH,
    FEATURE_PROCESSOR_TAG_KEY,
    FEATURE_PROCESSOR_TAG_VALUE,
    PIPELINE_CONTEXT_TYPE,
    DEFAULT_SCHEDULE_STATE,
    SCHEDULED_TIME_PIPELINE_PARAMETER,
    PIPELINE_CONTEXT_NAME_TAG_KEY,
    PIPELINE_VERSION_CONTEXT_NAME_TAG_KEY,
    PIPELINE_NAME_MAXIMUM_LENGTH,
    RESOURCE_NOT_FOUND,
    FEATURE_GROUP_ARN_REGEX_PATTERN,
    TO_PIPELINE_RESERVED_TAG_KEYS,
)
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)

from sagemaker.s3 import s3_path_join

from sagemaker import Session, get_execution_role
from sagemaker.feature_store.feature_processor._event_bridge_scheduler_helper import (
    EventBridgeSchedulerHelper,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.retry import (
    StepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
    SageMakerJobExceptionTypeEnum,
)

from sagemaker.workflow.steps import TrainingStep

from sagemaker.estimator import Estimator

from sagemaker.feature_store.feature_processor.lineage._feature_processor_lineage import (
    FeatureProcessorLineageHandler,
    TransformationCode,
)

from sagemaker.remote_function.job import (
    _JobSettings,
    JOBS_CONTAINER_ENTRYPOINT,
    SPARK_APP_SCRIPT_PATH,
)

from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
)

logger = logging.getLogger("sagemaker")


def to_pipeline(
    pipeline_name: str,
    step: Callable,
    role: Optional[str] = None,
    transformation_code: Optional[TransformationCode] = None,
    max_retries: Optional[int] = None,
    tags: Optional[List[Tuple[str, str]]] = None,
    sagemaker_session: Optional[Session] = None,
) -> str:
    """Creates a sagemaker pipeline that takes in a callable as a training step.

    To configure training step used in sagemaker pipeline, input argument step needs to be wrapped
    by remote decorator in module sagemaker.remote_function. If not wrapped by remote decorator,
    default configurations in sagemaker.remote_function.job._JobSettings will be used to create
    training step.

    Args:
        pipeline_name (str): The name of the pipeline.
        step (Callable): A user provided function wrapped by feature_processor and optionally
            wrapped by remote_decorator.
        role (Optional[str]): The Amazon Resource Name (ARN) of the role used by the pipeline to
            access and create resources. If not specified, it will default to the credentials
            provided by the AWS configuration chain.
        transformation_code (Optional[str]): The data source for a reference to the transformation
            code for Lineage tracking. This code is not used for actual transformation.
        max_retries (Optional[int]): The number of times to retry sagemaker pipeline step.
            If not specified, sagemaker pipline step will not retry.
        tags (List[Tuple[str, str]): A list of tags attached to the pipeline and all corresponding
            lineage resources that support tags. If not specified, no custom tags will be attached.
        sagemaker_session (Optional[Session]): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    Returns:
        str: SageMaker Pipeline ARN.
    """

    _validate_input_for_to_pipeline_api(pipeline_name, step)
    if tags:
        _validate_tags_for_to_pipeline_api(tags)

    _sagemaker_session = sagemaker_session or Session()

    _validate_lineage_resources_for_to_pipeline_api(
        step.feature_processor_config, _sagemaker_session
    )

    remote_decorator_config = _get_remote_decorator_config_from_input(
        wrapped_func=step, sagemaker_session=_sagemaker_session
    )
    _role = role or get_execution_role(_sagemaker_session)

    runtime_env_manager = RuntimeEnvironmentManager()
    client_python_version = runtime_env_manager._current_python_version()
    config_uploader = ConfigUploader(remote_decorator_config, runtime_env_manager)

    s3_base_uri = s3_path_join(remote_decorator_config.s3_root_uri, pipeline_name)

    (
        input_data_config,
        spark_dependency_paths,
    ) = config_uploader.prepare_step_input_channel_for_spark_mode(
        func=getattr(step, "wrapped_func", step),
        s3_base_uri=s3_base_uri,
        sagemaker_session=_sagemaker_session,
    )

    estimator_request_dict = _prepare_estimator_request_from_remote_decorator_config(
        remote_decorator_config=remote_decorator_config,
        s3_base_uri=s3_base_uri,
        client_python_version=client_python_version,
        spark_dependency_paths=spark_dependency_paths,
    )

    training_step_request_dict = dict(
        name="-".join([pipeline_name, "feature-processor"]),
        estimator=Estimator(**estimator_request_dict),
        inputs=input_data_config,
    )

    if max_retries:
        training_step_request_dict["retry_policies"] = [
            StepRetryPolicy(
                exception_types=[
                    StepExceptionTypeEnum.SERVICE_FAULT,
                    StepExceptionTypeEnum.THROTTLING,
                ],
                max_attempts=max_retries,
            ),
            SageMakerJobStepRetryPolicy(
                exception_types=[
                    SageMakerJobExceptionTypeEnum.INTERNAL_ERROR,
                    SageMakerJobExceptionTypeEnum.CAPACITY_ERROR,
                    SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT,
                ],
                max_attempts=max_retries,
            ),
        ]

    pipeline_request_dict = dict(
        name=pipeline_name,
        steps=[TrainingStep(**training_step_request_dict)],
        sagemaker_session=_sagemaker_session,
        parameters=[SCHEDULED_TIME_PIPELINE_PARAMETER],
    )
    pipeline_tags = [dict(Key=FEATURE_PROCESSOR_TAG_KEY, Value=FEATURE_PROCESSOR_TAG_VALUE)]
    if tags:
        pipeline_tags.extend([dict(Key=k, Value=v) for k, v in tags])

    pipeline = Pipeline(**pipeline_request_dict)
    logger.info("Creating/Updating sagemaker pipeline %s", pipeline_name)
    pipeline.upsert(
        role_arn=_role,
        tags=pipeline_tags,
    )
    logger.info("Created sagemaker pipeline %s", pipeline_name)

    describe_pipeline_response = pipeline.describe()
    pipeline_arn = describe_pipeline_response["PipelineArn"]
    tags_propagate_to_lineage_resources = _get_tags_from_pipeline_to_propagate_to_lineage_resources(
        pipeline_arn, _sagemaker_session
    )

    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=pipeline_name,
        pipeline_arn=pipeline_arn,
        pipeline=describe_pipeline_response,
        inputs=_get_feature_processor_inputs(wrapped_func=step),
        output=_get_feature_processor_outputs(wrapped_func=step),
        transformation_code=transformation_code,
        sagemaker_session=_sagemaker_session,
    )
    lineage_handler.create_lineage(tags_propagate_to_lineage_resources)
    lineage_handler.upsert_tags_for_lineage_resources(tags_propagate_to_lineage_resources)

    pipeline_lineage_names: Dict[str, str] = lineage_handler.get_pipeline_lineage_names()

    if pipeline_lineage_names is None:
        raise RuntimeError("Failed to retrieve pipeline lineage. Pipeline Lineage does not exist")

    pipeline.upsert(
        role_arn=_role,
        tags=[
            {
                "Key": PIPELINE_CONTEXT_NAME_TAG_KEY,
                "Value": pipeline_lineage_names["pipeline_context_name"],
            },
            {
                "Key": PIPELINE_VERSION_CONTEXT_NAME_TAG_KEY,
                "Value": pipeline_lineage_names["pipeline_version_context_name"],
            },
        ],
    )
    return pipeline_arn


def schedule(
    pipeline_name: str,
    schedule_expression: str,
    role_arn: Optional[str] = None,
    state: Optional[str] = DEFAULT_SCHEDULE_STATE,
    start_date: Optional[datetime] = None,
    sagemaker_session: Optional[Session] = None,
) -> str:
    """Creates an EventBridge Schedule that schedules executions of a sagemaker pipeline.

    The pipeline created will also have a pipeline parameter `scheduled-time` indicating when the
    pipeline is scheduled to run.

    Args:
        pipeline_name (str): The SageMaker Pipeline name that will be scheduled.
        schedule_expression (str): The expression that defines when the schedule runs. It supports
            at expression, rate expression and cron expression. See https://docs.aws.amazon.com/
            scheduler/latest/APIReference/API_CreateSchedule.html#scheduler-CreateSchedule-request
            -ScheduleExpression for more details.
        state (str): Specifies whether the schedule is enabled or disabled. Valid values are
            ENABLED and DISABLED. See https://docs.aws.amazon.com/scheduler/latest/APIReference/
            API_CreateSchedule.html#scheduler-CreateSchedule-request-State for more details.
            If not specified, it will default to ENABLED.
        start_date (Optional[datetime]): The date, in UTC, after which the schedule can begin
            invoking its target. Depending on the schedule’s recurrence expression, invocations
            might occur on, or after, the StartDate you specify.
        role_arn (Optional[str]): The Amazon Resource Name (ARN) of the IAM role that EventBridge
            Scheduler will assume for this target when the schedule is invoked.
        sagemaker_session (Optional[Session]): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    Returns:
        str: The EventBridge Schedule ARN.
    """

    _sagemaker_session = sagemaker_session or Session()
    _validate_pipeline_lineage_resources(pipeline_name, _sagemaker_session)
    _start_date = start_date or datetime.now(tz=pytz.utc)
    _role_arn = role_arn or get_execution_role(_sagemaker_session)
    event_bridge_scheduler_helper = EventBridgeSchedulerHelper(
        _sagemaker_session,
        _sagemaker_session.boto_session.client("scheduler"),
    )
    describe_pipeline_response = _sagemaker_session.sagemaker_client.describe_pipeline(
        PipelineName=pipeline_name
    )
    pipeline_arn = describe_pipeline_response["PipelineArn"]
    tags_propagate_to_lineage_resources = _get_tags_from_pipeline_to_propagate_to_lineage_resources(
        pipeline_arn, _sagemaker_session
    )

    logger.info("Creating/Updating EventBridge Schedule for pipeline %s.", pipeline_name)
    event_bridge_schedule_arn = event_bridge_scheduler_helper.upsert_schedule(
        schedule_name=pipeline_name,
        pipeline_arn=pipeline_arn,
        schedule_expression=schedule_expression,
        state=state,
        start_date=_start_date,
        role=_role_arn,
    )
    logger.info("Created/Updated EventBridge Schedule for pipeline %s.", pipeline_name)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=pipeline_name,
        pipeline_arn=describe_pipeline_response["PipelineArn"],
        pipeline=describe_pipeline_response,
        sagemaker_session=_sagemaker_session,
    )
    lineage_handler.create_schedule_lineage(
        pipeline_name=pipeline_name,
        schedule_arn=event_bridge_schedule_arn["ScheduleArn"],
        schedule_expression=schedule_expression,
        state=state,
        start_date=_start_date,
        tags=tags_propagate_to_lineage_resources,
    )
    return event_bridge_schedule_arn["ScheduleArn"]


def execute(
    pipeline_name: str,
    execution_time: Optional[datetime] = None,
    sagemaker_session: Optional[Session] = None,
) -> str:
    """Starts an execution of a SageMaker Pipeline created by feature_processor

    Args:
        pipeline_name (str): The SageMaker Pipeline name that will be executed.
        execution_time (datetime): The date, in UTC, will be used as a sagemaker pipeline parameter
            indicating the time which at which the execution is scheduled to execute. If not
            specified, it will default to the current timestamp.
        sagemaker_session (Optional[Session]): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    Returns:
        str: The pipeline execution ARN.
    """
    _sagemaker_session = sagemaker_session or Session()
    _validate_pipeline_lineage_resources(pipeline_name, _sagemaker_session)
    _execution_time = execution_time or datetime.now()
    start_pipeline_execution_request = dict(
        PipelineName=pipeline_name,
        PipelineParameters=[
            dict(
                Name=EXECUTION_TIME_PIPELINE_PARAMETER,
                Value=_execution_time.strftime(EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT),
            )
        ],
    )
    logger.info("Starting an execution for pipline %s", pipeline_name)
    execution_response = _sagemaker_session.sagemaker_client.start_pipeline_execution(
        **start_pipeline_execution_request
    )
    execution_arn = execution_response["PipelineExecutionArn"]
    logger.info(
        "Execution %s for pipeline %s is successfully started.",
        execution_arn,
        pipeline_name,
    )
    return execution_arn


def delete_schedule(pipeline_name: str, sagemaker_session: Optional[Session] = None) -> None:
    """Delete EventBridge Schedule corresponding to a SageMaker Pipeline if there is one.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline that needs to be deleted
        sagemaker_session: (Optional[Session], optional): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    """
    _sagemaker_session = sagemaker_session or Session()
    event_bridge_scheduler_helper = EventBridgeSchedulerHelper(
        _sagemaker_session, _sagemaker_session.boto_session.client("scheduler")
    )
    try:
        event_bridge_scheduler_helper.delete_schedule(pipeline_name)
        logger.info("Deleted EventBridge Schedule for pipeline %s.", pipeline_name)
    except ClientError as e:
        if RESOURCE_NOT_FOUND_EXCEPTION != e.response["Error"]["Code"]:
            raise e


def describe(
    pipeline_name: str, sagemaker_session: Optional[Session] = None
) -> Dict[str, Union[int, str]]:
    """Describe feature processor and other related resources.

    This API will include details related to the feature processor including SageMaker Pipeline and
    EventBridge Schedule.

    Args:
        pipeline_name (str): Name of the pipeline.
        sagemaker_session (Optional[Session]): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    Returns:
        Dict[str, Union[int, str]]: Return information for resources related to feature processor.
    """

    _sagemaker_session = sagemaker_session or Session()
    describe_response_dict = {}

    try:
        describe_pipeline_response = _sagemaker_session.sagemaker_client.describe_pipeline(
            PipelineName=pipeline_name
        )
        pipeline_definition = json.loads(describe_pipeline_response["PipelineDefinition"])
        pipeline_step = pipeline_definition["Steps"][0]
        describe_response_dict = dict(
            pipeline_arn=describe_pipeline_response["PipelineArn"],
            pipeline_execution_role_arn=describe_pipeline_response["RoleArn"],
        )

        if "RetryPolicies" in pipeline_step:
            describe_response_dict["max_retries"] = pipeline_step["RetryPolicies"][0]["MaxAttempts"]
    except ClientError as e:
        if RESOURCE_NOT_FOUND_EXCEPTION == e.response["Error"]["Code"]:
            logger.info("Pipeline %s does not exist.", pipeline_name)

    event_bridge_scheduler_helper = EventBridgeSchedulerHelper(
        _sagemaker_session,
        _sagemaker_session.boto_session.client("scheduler"),
    )

    event_bridge_schedule = event_bridge_scheduler_helper.describe_schedule(pipeline_name)
    if event_bridge_schedule:
        describe_response_dict.update(
            dict(
                schedule_arn=event_bridge_schedule["Arn"],
                schedule_expression=event_bridge_schedule["ScheduleExpression"],
                schedule_state=event_bridge_schedule["State"],
                schedule_start_date=event_bridge_schedule["StartDate"].strftime(
                    EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT
                ),
                schedule_role=event_bridge_schedule["Target"]["RoleArn"],
            )
        )

    return describe_response_dict


def list_pipelines(sagemaker_session: Optional[Session] = None) -> List[Dict[str, Any]]:
    """Lists all SageMaker Pipelines created by Feature Processor SDK.

    Args:
        sagemaker_session (Optional[Session]): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    Returns:
        List[Dict[str, Any]]: Return list of SageMaker Pipeline metadata created for
            feature_processor.
    """

    _sagemaker_session = sagemaker_session or Session()
    next_token = None
    list_response = []
    pipeline_names_so_far = set([])
    while True:
        list_contexts_request = dict(ContextType=PIPELINE_CONTEXT_TYPE)
        if next_token:
            list_contexts_request["NextToken"] = next_token
        list_contexts_response = _sagemaker_session.sagemaker_client.list_contexts(
            **list_contexts_request
        )
        for _context in list_contexts_response["ContextSummaries"]:
            pipeline_name = get_resource_name_from_arn(_context["Source"]["SourceUri"])
            if pipeline_name not in pipeline_names_so_far:
                list_response.append(dict(pipeline_name=pipeline_name))
                pipeline_names_so_far.add(pipeline_name)
        next_token = list_contexts_response.get("NextToken")
        if not next_token:
            break

    return list_response


def _validate_input_for_to_pipeline_api(pipeline_name: str, step: Callable) -> None:
    """Validate input to to_pipeline API.

    The provided callable is considered valid if it's wrapped by feature_processor decorator
    and uses pyspark mode.

    Args:
        pipeline_name (str): The name of the pipeline.
        step (Callable): A user provided function wrapped by feature_processor and optionally
            wrapped by remote_decorator.

    Raises (ValueError): raises ValueError when any of the following scenario happen:
           1. pipeline name is longer than 80 characters.
           2. function is not annotated with either feature_processor or remote decorator.
           3. provides a mode other than pyspark.
    """
    if len(pipeline_name) > PIPELINE_NAME_MAXIMUM_LENGTH:
        raise ValueError(
            "Pipeline name used by feature processor should be less than 80 "
            "characters. Please choose another pipeline name."
        )

    if not hasattr(step, "feature_processor_config") or not step.feature_processor_config:
        raise ValueError(
            "Please wrap step parameter with feature_processor decorator"
            " in order to use to_pipeline API."
        )

    if not hasattr(step, "job_settings") or not step.job_settings:
        raise ValueError(
            "Please wrap step parameter with remote decorator in order to use to_pipeline API."
        )

    if FeatureProcessorMode.PYSPARK != step.feature_processor_config.mode:
        raise ValueError(
            f"Mode {step.feature_processor_config.mode} is not supported by to_pipeline API."
        )


def _validate_tags_for_to_pipeline_api(tags: List[Tuple[str, str]]) -> None:
    """Validate tags provided to to_pipeline API.

    Args:
        tags (List[Tuple[str, str]]): A list of tags attached to the pipeline.

    Raises (ValueError): raises ValueError when any of the following scenario happen:
           1. reserved tag keys are provided to API.
    """
    provided_tag_keys = [tag_key_value_pair[0] for tag_key_value_pair in tags]
    for reserved_tag_key in TO_PIPELINE_RESERVED_TAG_KEYS:
        if reserved_tag_key in provided_tag_keys:
            raise ValueError(
                f"{reserved_tag_key} is a reserved tag key for to_pipeline API. Please choose another tag."
            )


def _validate_lineage_resources_for_to_pipeline_api(
    feature_processor_config: FeatureProcessorConfig, sagemaker_session: Session
) -> None:
    """Validate existence of feature group lineage resources for to_pipeline API.

    Args:
        feature_processor_config (FeatureProcessorConfig): The configuration values for the
            feature_processor decorator.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed.
    """
    inputs = feature_processor_config.inputs
    output = feature_processor_config.output
    for ds in inputs:
        if isinstance(ds, FeatureGroupDataSource):
            fg_name = _parse_name_from_arn(ds.name)
            _validate_fg_lineage_resources(fg_name, sagemaker_session)
    output_fg_name = _parse_name_from_arn(output)
    _validate_fg_lineage_resources(output_fg_name, sagemaker_session)


def _validate_fg_lineage_resources(feature_group_name: str, sagemaker_session: Session) -> None:
    """Validate existence of feature group lineage resources.

    Args:
        feature_group_name (str): The name or arn of the feature group.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed.

    Raises (ValueError): raises ValueError when lineage resources are not created for feature
        groups.
    """

    feature_group = sagemaker_session.describe_feature_group(feature_group_name=feature_group_name)
    feature_group_creation_time = feature_group["CreationTime"].strftime("%s")
    feature_group_context = _get_feature_group_lineage_context_name(
        feature_group_name=feature_group_name,
        feature_group_creation_time=feature_group_creation_time,
    )
    feature_group_pipeline_context = _get_feature_group_pipeline_lineage_context_name(
        feature_group_name=feature_group_name,
        feature_group_creation_time=feature_group_creation_time,
    )
    feature_group_pipeline_version_context = (
        _get_feature_group_pipeline_version_lineage_context_name(
            feature_group_name=feature_group_name,
            feature_group_creation_time=feature_group_creation_time,
        )
    )
    for context_name in [
        feature_group_context,
        feature_group_pipeline_context,
        feature_group_pipeline_version_context,
    ]:
        try:
            logger.info("Verifying existence of context %s.", context_name)
            context.Context.load(context_name=context_name, sagemaker_session=sagemaker_session)
        except ClientError as e:
            if RESOURCE_NOT_FOUND == e.response["Error"]["Code"]:
                raise ValueError(
                    f"Lineage resource {context_name} has not yet been created for feature group"
                    f" {feature_group_name} or has already been deleted. Please try again later."
                )
            raise e


def _validate_pipeline_lineage_resources(pipeline_name: str, sagemaker_session: Session) -> None:
    """Validate existence of pipeline lineage resources.

    Args:
        pipeline_name (str): The name of the pipeline.
    sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed.
    """
    pipeline = sagemaker_session.sagemaker_client.describe_pipeline(PipelineName=pipeline_name)
    pipeline_creation_time = pipeline["CreationTime"].strftime("%s")
    pipeline_context_name = _get_feature_processor_pipeline_lineage_context_name(
        pipeline_name=pipeline_name, pipeline_creation_time=pipeline_creation_time
    )
    try:
        pipeline_context = context.Context.load(
            context_name=pipeline_context_name, sagemaker_session=sagemaker_session
        )
        last_update_time = pipeline_context.properties["LastUpdateTime"]
        pipeline_version_context_name = (
            _get_feature_processor_pipeline_version_lineage_context_name(
                pipeline_name=pipeline_name, pipeline_last_update_time=last_update_time
            )
        )
        context.Context.load(
            context_name=pipeline_version_context_name, sagemaker_session=sagemaker_session
        )
    except ClientError as e:
        if RESOURCE_NOT_FOUND == e.response["Error"]["Code"]:
            raise ValueError(
                "Pipeline lineage resources have not been created yet or have already been deleted"
                ". Please try again later."
            )
        raise e


def _prepare_estimator_request_from_remote_decorator_config(
    remote_decorator_config: _JobSettings,
    s3_base_uri: str,
    client_python_version: str,
    spark_dependency_paths: Dict[str, Optional[str]],
) -> Dict[str, Union[str, int, List]]:
    """Prepares request dictionary used for Estimator creation.

    Args:
        remote_decorator_config (_JobSettings): Configurations used for setting up
            SageMaker Pipeline Step.
        s3_base_uri (str): S3 URI used as destination for dependencies upload.
        client_python_version (str): Python version used on client side.
        spark_dependency_paths (Dict[str, Optional[str]]): A dictionary contains S3 paths spark
            dependency files get uploaded to if present.
    Returns:
        Dict[str, List[str]]: Request dictionary containing configurations for Estimator creation.
    """
    estimator_request_dict = dict(
        role=remote_decorator_config.role,
        max_run=remote_decorator_config.max_runtime_in_seconds,
        max_retry_attempts=remote_decorator_config.max_retry_attempts,
        output_path=s3_base_uri,
        output_kms_key=remote_decorator_config.s3_kms_key,
        image_uri=remote_decorator_config.image_uri,
        input_mode="File",
        volume_size=remote_decorator_config.volume_size,
        volume_kms_key=remote_decorator_config.volume_kms_key,
        instance_count=remote_decorator_config.instance_count,
        instance_type=remote_decorator_config.instance_type,
        encrypt_inter_container_traffic=remote_decorator_config.encrypt_inter_container_traffic,
        environment=remote_decorator_config.environment_variables or {},
    )

    estimator_request_dict["environment"][
        EXECUTION_TIME_PIPELINE_PARAMETER
    ] = SCHEDULED_TIME_PIPELINE_PARAMETER

    if remote_decorator_config.tags:
        estimator_request_dict["tags"] = [
            {"Key": k, "Value": v} for k, v in remote_decorator_config.tags
        ]
    if remote_decorator_config.vpc_config:
        estimator_request_dict["subnets"] = remote_decorator_config.vpc_config[SUBNETS_KEY]
        estimator_request_dict["security_group_ids"] = remote_decorator_config.vpc_config[
            SECURITY_GROUP_IDS_KEY
        ]

    estimator_request_dict.update(
        **_get_container_entry_point_and_arguments(
            remote_decorator_config=remote_decorator_config,
            s3_base_uri=s3_base_uri,
            client_python_version=client_python_version,
            spark_dependency_paths=spark_dependency_paths,
        )
    )

    return estimator_request_dict


def _get_container_entry_point_and_arguments(
    remote_decorator_config: _JobSettings,
    s3_base_uri: str,
    client_python_version: str,
    spark_dependency_paths: Dict[str, Optional[str]],
) -> Dict[str, List[str]]:
    """Extracts the container entry point and container arguments from remote decorator configs

    Args:
        remote_decorator_config (_JobSettings): Configurations used for setting up
            SageMaker Pipeline Step.
        s3_base_uri (str): S3 URI used as destination for dependencies upload.
        client_python_version (str): Python version used on client side.
        spark_dependency_paths (Dict[str, Optional[str]]): A dictionary contains S3 paths spark
            dependency files get uploaded to if present.
    Returns:
        Dict[str, List[str]]: Request dictionary containing container entry point and
            arguments setup.
    """

    spark_config = remote_decorator_config.spark_config
    jobs_container_entrypoint = JOBS_CONTAINER_ENTRYPOINT.copy()

    if spark_dependency_paths[SPARK_JAR_FILES_PATH]:
        jobs_container_entrypoint.extend(["--jars", spark_dependency_paths[SPARK_JAR_FILES_PATH]])

    if spark_dependency_paths[SPARK_PY_FILES_PATH]:
        jobs_container_entrypoint.extend(
            ["--py-files", spark_dependency_paths[SPARK_PY_FILES_PATH]]
        )

    if spark_dependency_paths[SPARK_FILES_PATH]:
        jobs_container_entrypoint.extend(["--files", spark_dependency_paths[SPARK_FILES_PATH]])

    if spark_config and spark_config.spark_event_logs_uri:
        jobs_container_entrypoint.extend(
            ["--spark-event-logs-s3-uri", spark_config.spark_event_logs_uri]
        )

    if spark_config:
        jobs_container_entrypoint.extend([SPARK_APP_SCRIPT_PATH])

    container_args = ["--s3_base_uri", s3_base_uri]
    container_args.extend(["--region", remote_decorator_config.sagemaker_session.boto_region_name])
    container_args.extend(["--client_python_version", client_python_version])

    if remote_decorator_config.s3_kms_key:
        container_args.extend(["--s3_kms_key", remote_decorator_config.s3_kms_key])

    return dict(
        container_entry_point=jobs_container_entrypoint,
        container_arguments=container_args,
    )


def _get_remote_decorator_config_from_input(
    wrapped_func: Callable, sagemaker_session: Session
) -> _JobSettings:
    """Extracts the remote decorator configuration from the wrapped function and other inputs.

    Args:
        wrapped_func (Callable): Wrapped user defined function. If it contains remote decorator
            job settings, configs will be used to construct remote_decorator_config, otherwise
            default job settings will be used.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    Returns:
        _JobSettings: Configurations used for creating sagemaker pipeline step.
    """
    remote_decorator_config = getattr(
        wrapped_func,
        "job_settings",
    )
    # TODO: Remove this after GA
    remote_decorator_config.sagemaker_session = sagemaker_session

    # TODO: This needs to be removed when new mode is introduced.
    if remote_decorator_config.spark_config is None:
        remote_decorator_config.spark_config = SparkConfig()
    remote_decorator_config.image_uri = _JobSettings._get_default_spark_image(sagemaker_session)

    return remote_decorator_config


def _get_feature_processor_inputs(
    wrapped_func: Callable,
) -> Sequence[Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource]]:
    """Retrieve Feature Processor Config Inputs"""
    feature_processor_config: FeatureProcessorConfig = wrapped_func.feature_processor_config
    return feature_processor_config.inputs


def _get_feature_processor_outputs(
    wrapped_func: Callable,
) -> str:
    """Retrieve Feature Processor Config Output"""
    feature_processor_config: FeatureProcessorConfig = wrapped_func.feature_processor_config
    return feature_processor_config.output


def _parse_name_from_arn(fg_uri: str) -> str:
    """Parse the name from a string, if it's an ARN. Otherwise, return the string.

    Args:
        fg_uri (str): The Feature Group Name or ARN.

    Returns:
        str: The Feature Group Name.
    """
    match = re.match(FEATURE_GROUP_ARN_REGEX_PATTERN, fg_uri)
    if match:
        feature_group_name = match.group(4)
        return feature_group_name
    return fg_uri


def _get_tags_from_pipeline_to_propagate_to_lineage_resources(
    pipeline_arn: str, sagemaker_session: Session
) -> List[Dict[str, str]]:
    """Retrieve custom tags attached to sagemakre pipeline

    Args:
        pipeline_arn (str): SageMaker Pipeline Arn.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.

    Returns:
        List[Dict[str, str]]: List of custom tags to be propagated to lineage resources.
    """
    tags_in_pipeline = sagemaker_session.sagemaker_client.list_tags(ResourceArn=pipeline_arn)[
        "Tags"
    ]
    return [d for d in tags_in_pipeline if d["Key"] not in TO_PIPELINE_RESERVED_TAG_KEYS]
