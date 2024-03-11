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
"""Contains class to handle Lineage Associations"""
from __future__ import absolute_import
import logging
from datetime import datetime
from typing import Optional, Iterator, List, Dict, Set, Sequence, Union
import attr
from botocore.exceptions import ClientError

from sagemaker.feature_store.feature_processor._event_bridge_scheduler_helper import (
    EventBridgeSchedulerHelper,
)
from sagemaker.feature_store.feature_processor.lineage._lineage_association_handler import (
    LineageAssociationHandler,
)

# pylint: disable=C0301
from sagemaker.feature_store.feature_processor.lineage._feature_group_lineage_entity_handler import (
    FeatureGroupLineageEntityHandler,
)
from sagemaker.feature_store.feature_processor.lineage._feature_group_contexts import (
    FeatureGroupContexts,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_lineage_entity_handler import (
    PipelineLineageEntityHandler,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_schedule import (
    PipelineSchedule,
)

# pylint: disable=C0301
from sagemaker.feature_store.feature_processor.lineage._pipeline_version_lineage_entity_handler import (
    PipelineVersionLineageEntityHandler,
)
from sagemaker.feature_store.feature_processor.lineage._s3_lineage_entity_handler import (
    S3LineageEntityHandler,
)
from sagemaker.feature_store.feature_processor.lineage._transformation_code import (
    TransformationCode,
)
from sagemaker.feature_store.feature_processor.lineage.constants import (
    SAGEMAKER,
    LAST_UPDATE_TIME,
    PIPELINE_CONTEXT_NAME_KEY,
    PIPELINE_CONTEXT_VERSION_NAME_KEY,
    FEATURE_GROUP_PIPELINE_VERSION_CONTEXT_TYPE,
    DATA_SET,
    TRANSFORMATION_CODE,
    CREATION_TIME,
    RESOURCE_NOT_FOUND,
    ERROR,
    CODE,
    LAST_MODIFIED_TIME,
    TRANSFORMATION_CODE_STATUS_INACTIVE,
    TRANSFORMATION_CODE_STATUS_ACTIVE,
)
from sagemaker.lineage.context import Context
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import AssociationSummary
from sagemaker import Session
from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
)

logger = logging.getLogger(SAGEMAKER)


@attr.s
class FeatureProcessorLineageHandler:
    """Class to Create and Update FeatureProcessor Lineage Entities.

    Attributes:
        pipeline_name (str): Pipeline Name.
        pipeline_arn (str): The ARN of the Pipeline.
        pipeline (str): The details of the Pipeline.
        inputs (Sequence[Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource]]):
            The inputs to the Feature processor.
        output (str): The output Feature Group.
        transformation_code (TransformationCode): The Transformation Code for Feature Processor.
        sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.
    """

    pipeline_name: str = attr.ib()
    pipeline_arn: str = attr.ib()
    pipeline: Dict = attr.ib()
    sagemaker_session: Session = attr.ib()
    inputs: Sequence[Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource]] = attr.ib(
        default=None
    )
    output: str = attr.ib(default=None)
    transformation_code: TransformationCode = attr.ib(default=None)

    def create_lineage(self, tags: Optional[List[Dict[str, str]]] = None) -> None:
        """Create and Update Feature Processor Lineage"""
        input_feature_group_contexts: List[
            FeatureGroupContexts
        ] = self._retrieve_input_feature_group_contexts()
        output_feature_group_contexts: FeatureGroupContexts = (
            self._retrieve_output_feature_group_contexts()
        )
        input_raw_data_artifacts: List[Artifact] = self._retrieve_input_raw_data_artifacts()
        transformation_code_artifact: Optional[
            Artifact
        ] = S3LineageEntityHandler.create_transformation_code_artifact(
            transformation_code=self.transformation_code,
            pipeline_last_update_time=self.pipeline[LAST_MODIFIED_TIME].strftime("%s"),
            sagemaker_session=self.sagemaker_session,
        )
        if transformation_code_artifact is not None:
            logger.info("Created Transformation Code Artifact: %s", transformation_code_artifact)
            if tags:
                transformation_code_artifact.set_tags(tags)  # pylint: disable=E1101
        # Create the Pipeline Lineage for the first time
        if not self._check_if_pipeline_lineage_exists():
            self._create_new_pipeline_lineage(
                input_feature_group_contexts=input_feature_group_contexts,
                input_raw_data_artifacts=input_raw_data_artifacts,
                output_feature_group_contexts=output_feature_group_contexts,
                transformation_code_artifact=transformation_code_artifact,
            )
        else:
            self._update_pipeline_lineage(
                input_feature_group_contexts=input_feature_group_contexts,
                input_raw_data_artifacts=input_raw_data_artifacts,
                output_feature_group_contexts=output_feature_group_contexts,
                transformation_code_artifact=transformation_code_artifact,
            )

    def get_pipeline_lineage_names(self) -> Optional[Dict[str, str]]:
        """Retrieve Pipeline Lineage Names.

        Returns:
            Optional[Dict[str, str]]: Pipeline and Pipeline version lineage names.
        """
        if not self._check_if_pipeline_lineage_exists():
            return None
        pipeline_context: Context = self._get_pipeline_context()
        current_pipeline_version_context: Context = self._get_pipeline_version_context(
            last_update_time=pipeline_context.properties[LAST_UPDATE_TIME]
        )
        return {
            PIPELINE_CONTEXT_NAME_KEY: pipeline_context.context_name,
            PIPELINE_CONTEXT_VERSION_NAME_KEY: current_pipeline_version_context.context_name,
        }

    def create_schedule_lineage(
        self,
        pipeline_name: str,
        schedule_arn,
        schedule_expression,
        state,
        start_date: datetime,
        tags: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Class to Create and Update FeatureProcessor Lineage Entities.

        Arguments:
            pipeline_name (str): Pipeline Name.
            schedule_arn (str): The ARN of the Schedule.
            schedule_expression (str): The expression that defines when the schedule runs.
                It supports at expression, rate expression and cron expression.
            state (str):Specifies whether the schedule is enabled or disabled. Valid values are
                ENABLED and DISABLED. See https://docs.aws.amazon.com/scheduler/latest/APIReference/
                API_CreateSchedule.html#scheduler-CreateSchedule-request-State for more details.
                If not specified, it will default to ENABLED.
            start_date (Optional[datetime]): The date, in UTC, after which the schedule can begin
                invoking its target. Depending on the schedule’s recurrence expression, invocations
                might occur on, or after, the StartDate you specify.
            tags (Optional[List[Dict[str, str]]]): Custom tags to be attached to schedule
                lineage resource.
        """
        pipeline_context: Context = self._get_pipeline_context()
        pipeline_version_context: Context = self._get_pipeline_version_context(
            last_update_time=pipeline_context.properties[LAST_UPDATE_TIME]
        )
        pipeline_schedule: PipelineSchedule = PipelineSchedule(
            schedule_name=pipeline_name,
            schedule_arn=schedule_arn,
            schedule_expression=schedule_expression,
            pipeline_name=pipeline_name,
            state=state,
            start_date=start_date.strftime("%s"),
        )
        schedule_artifact: Artifact = S3LineageEntityHandler.retrieve_pipeline_schedule_artifact(
            pipeline_schedule=pipeline_schedule,
            sagemaker_session=self.sagemaker_session,
        )
        if tags:
            schedule_artifact.set_tags(tags)

        LineageAssociationHandler.add_upstream_schedule_associations(
            schedule_artifact=schedule_artifact,
            pipeline_version_context_arn=pipeline_version_context.context_arn,
            sagemaker_session=self.sagemaker_session,
        )

    def upsert_tags_for_lineage_resources(self, tags: List[Dict[str, str]]) -> None:
        """Add or update tags for lineage resources using tags attached to sagemaker pipeline as

        source of truth.

        Args:
            tags (List[Dict[str, str]]): Custom tags to be attached to lineage resources.
        """
        if not tags:
            return
        pipeline_context: Context = self._get_pipeline_context()
        current_pipeline_version_context: Context = self._get_pipeline_version_context(
            last_update_time=pipeline_context.properties[LAST_UPDATE_TIME]
        )
        input_raw_data_artifacts: List[Artifact] = self._retrieve_input_raw_data_artifacts()
        pipeline_context.set_tags(tags)
        current_pipeline_version_context.set_tags(tags)
        for input_raw_data_artifact in input_raw_data_artifacts:
            input_raw_data_artifact.set_tags(tags)

        event_bridge_scheduler_helper = EventBridgeSchedulerHelper(
            self.sagemaker_session,
            self.sagemaker_session.boto_session.client("scheduler"),
        )
        event_bridge_schedule = event_bridge_scheduler_helper.describe_schedule(self.pipeline_name)

        if event_bridge_schedule:
            schedule_artifact_summary = S3LineageEntityHandler._load_artifact_from_s3_uri(
                s3_uri=event_bridge_schedule["Arn"],
                sagemaker_session=self.sagemaker_session,
            )
            if schedule_artifact_summary is not None:
                pipeline_schedule_artifact: Artifact = (
                    S3LineageEntityHandler.load_artifact_from_arn(
                        artifact_arn=schedule_artifact_summary.artifact_arn,
                        sagemaker_session=self.sagemaker_session,
                    )
                )
                pipeline_schedule_artifact.set_tags(tags)

    def _create_new_pipeline_lineage(
        self,
        input_feature_group_contexts: List[FeatureGroupContexts],
        input_raw_data_artifacts: List[Artifact],
        output_feature_group_contexts: FeatureGroupContexts,
        transformation_code_artifact: Optional[Artifact],
    ) -> None:
        """Create pipeline lineage resources."""

        pipeline_context = self._create_pipeline_lineage_for_new_pipeline()
        pipeline_version_context = self._create_pipeline_version_lineage()
        self._add_associations_for_pipeline(
            # pylint: disable=no-member
            pipeline_context_arn=pipeline_context.context_arn,
            # pylint: disable=no-member
            pipeline_versions_context_arn=pipeline_version_context.context_arn,
            input_feature_group_contexts=input_feature_group_contexts,
            input_raw_data_artifacts=input_raw_data_artifacts,
            output_feature_group_contexts=output_feature_group_contexts,
            transformation_code_artifact=transformation_code_artifact,
        )
        LineageAssociationHandler.add_pipeline_and_pipeline_version_association(
            # pylint: disable=no-member
            pipeline_context_arn=pipeline_context.context_arn,
            # pylint: disable=no-member
            pipeline_version_context_arn=pipeline_version_context.context_arn,
            sagemaker_session=self.sagemaker_session,
        )

    def _update_pipeline_lineage(
        self,
        input_feature_group_contexts: List[FeatureGroupContexts],
        input_raw_data_artifacts: List[Artifact],
        output_feature_group_contexts: FeatureGroupContexts,
        transformation_code_artifact: Optional[Artifact],
    ) -> None:
        """Update pipeline lineage resources."""

        # If pipeline lineage exists then determine whether to create a new version.
        pipeline_context: Context = self._get_pipeline_context()
        current_pipeline_version_context: Context = self._get_pipeline_version_context(
            last_update_time=pipeline_context.properties[LAST_UPDATE_TIME]
        )
        upstream_feature_group_associations: Iterator[
            AssociationSummary
        ] = LineageAssociationHandler.list_upstream_associations(
            # pylint: disable=no-member
            entity_arn=current_pipeline_version_context.context_arn,
            source_type=FEATURE_GROUP_PIPELINE_VERSION_CONTEXT_TYPE,
            sagemaker_session=self.sagemaker_session,
        )

        upstream_raw_data_associations: Iterator[
            AssociationSummary
        ] = LineageAssociationHandler.list_upstream_associations(
            # pylint: disable=no-member
            entity_arn=current_pipeline_version_context.context_arn,
            source_type=DATA_SET,
            sagemaker_session=self.sagemaker_session,
        )

        upstream_transformation_code: Iterator[
            AssociationSummary
        ] = LineageAssociationHandler.list_upstream_associations(
            # pylint: disable=no-member
            entity_arn=current_pipeline_version_context.context_arn,
            source_type=TRANSFORMATION_CODE,
            sagemaker_session=self.sagemaker_session,
        )

        downstream_feature_group_associations: Iterator[
            AssociationSummary
        ] = LineageAssociationHandler.list_downstream_associations(
            # pylint: disable=no-member
            entity_arn=current_pipeline_version_context.context_arn,
            destination_type=FEATURE_GROUP_PIPELINE_VERSION_CONTEXT_TYPE,
            sagemaker_session=self.sagemaker_session,
        )

        is_upstream_feature_group_equal: bool = self._compare_upstream_feature_groups(
            upstream_feature_group_associations=upstream_feature_group_associations,
            input_feature_group_contexts=input_feature_group_contexts,
        )
        is_downstream_feature_group_equal: bool = self._compare_downstream_feature_groups(
            downstream_feature_group_associations=downstream_feature_group_associations,
            output_feature_group_contexts=output_feature_group_contexts,
        )
        is_upstream_raw_data_equal: bool = self._compare_upstream_raw_data(
            upstream_raw_data_associations=upstream_raw_data_associations,
            input_raw_data_artifacts=input_raw_data_artifacts,
        )

        self._update_last_transformation_code(
            upstream_transformation_code_associations=upstream_transformation_code
        )
        if (
            not is_upstream_feature_group_equal
            or not is_downstream_feature_group_equal
            or not is_upstream_raw_data_equal
        ):
            if not is_upstream_raw_data_equal:
                logger.info("Raw data inputs have changed from the last pipeline configuration.")
            if not is_upstream_feature_group_equal:
                logger.info(
                    "Feature group inputs have changed from the last pipeline configuration."
                )
            if not is_downstream_feature_group_equal:
                logger.info(
                    "Feature Group output has changed from the last pipeline configuration."
                )
            pipeline_context.properties["LastUpdateTime"] = self.pipeline[
                "LastModifiedTime"
            ].strftime("%s")
            PipelineLineageEntityHandler.update_pipeline_context(pipeline_context=pipeline_context)
            new_pipeline_version_context: Context = self._create_pipeline_version_lineage()
            self._add_associations_for_pipeline(
                # pylint: disable=no-member
                pipeline_context_arn=pipeline_context.context_arn,
                # pylint: disable=no-member
                pipeline_versions_context_arn=new_pipeline_version_context.context_arn,
                input_feature_group_contexts=input_feature_group_contexts,
                input_raw_data_artifacts=input_raw_data_artifacts,
                output_feature_group_contexts=output_feature_group_contexts,
                transformation_code_artifact=transformation_code_artifact,
            )
            LineageAssociationHandler.add_pipeline_and_pipeline_version_association(
                # pylint: disable=no-member
                pipeline_context_arn=pipeline_context.context_arn,
                # pylint: disable=no-member
                pipeline_version_context_arn=new_pipeline_version_context.context_arn,
                sagemaker_session=self.sagemaker_session,
            )
        elif transformation_code_artifact is not None:
            # We will append the new transformation code artifact
            # to the existing pipeline version.
            LineageAssociationHandler.add_upstream_transformation_code_associations(
                transformation_code_artifact=transformation_code_artifact,
                # pylint: disable=no-member
                pipeline_version_context_arn=current_pipeline_version_context.context_arn,
                sagemaker_session=self.sagemaker_session,
            )

    def _retrieve_input_raw_data_artifacts(self) -> List[Artifact]:
        """Retrieve input Raw Data Artifacts.

        Returns:
            List[Artifact]: List of Raw Data Artifacts.
        """
        raw_data_artifacts: List[Artifact] = list()
        raw_data_s3_uri_set: Set[str] = set()
        for data_source in self.inputs:
            if isinstance(data_source, (CSVDataSource, ParquetDataSource)):
                if data_source.s3_uri not in raw_data_s3_uri_set:
                    raw_data_s3_uri_set.add(data_source.s3_uri)
                    raw_data_artifacts.append(
                        S3LineageEntityHandler.retrieve_raw_data_artifact(
                            raw_data=data_source,
                            sagemaker_session=self.sagemaker_session,
                        )
                    )
        return raw_data_artifacts

    def _compare_upstream_raw_data(
        self,
        upstream_raw_data_associations: Iterator[AssociationSummary],
        input_raw_data_artifacts: List[Artifact],
    ) -> bool:
        """Compare the existing and the new upstream Raw Data.

        Arguments:
            upstream_raw_data_associations (Iterator[AssociationSummary]):
                Upstream existing raw data associations for the pipeline.
            input_raw_data_artifacts (List[Artifact]):
                New Upstream raw data for the pipeline.

        Returns:
            bool: Boolean if old and new upstream is same.
        """
        raw_data_association_set = {
            raw_data_association.source_arn
            for raw_data_association in upstream_raw_data_associations
        }
        if len(raw_data_association_set) != len(input_raw_data_artifacts):
            return False
        for raw_data in input_raw_data_artifacts:
            if raw_data.artifact_arn not in raw_data_association_set:
                return False
        return True

    def _compare_downstream_feature_groups(
        self,
        downstream_feature_group_associations: Iterator[AssociationSummary],
        output_feature_group_contexts: FeatureGroupContexts,
    ) -> bool:
        """Compare the existing and the new downstream Feature Groups.

        Arguments:
            downstream_feature_group_associations (Iterator[AssociationSummary]):
                Downstream existing Feature Group association for the pipeline.
            output_feature_group_contexts (List[Artifact]):
                New Downstream Feature group for the pipeline.

        Returns:
            bool: Boolean if old and new Downstream is same.
        """
        feature_group_association_set = set()
        for feature_group_association in downstream_feature_group_associations:
            feature_group_association_set.add(feature_group_association.destination_arn)
        if len(feature_group_association_set) != 1:
            ValueError(
                f"There should only be one Feature Group as output, "
                f"instead we got {len(feature_group_association_set)}. "
                f"With Feature Group Versions Contexts: {feature_group_association_set}"
            )
        return (
            output_feature_group_contexts.pipeline_version_context_arn
            in feature_group_association_set
        )

    def _compare_upstream_feature_groups(
        self,
        upstream_feature_group_associations: Iterator[AssociationSummary],
        input_feature_group_contexts: List[FeatureGroupContexts],
    ) -> bool:
        """Compare the existing and the new upstream Feature Group.

        Arguments:
            upstream_feature_group_associations (Iterator[AssociationSummary]):
                Upstream existing Feature Group association for the pipeline.
            input_feature_group_contexts (List[Artifact]):
                New Upstream Feature group for the pipeline.

        Returns:
            bool: Boolean if old and new upstream is same.
        """
        feature_group_association_set = set()
        for feature_group_association in upstream_feature_group_associations:
            feature_group_association_set.add(feature_group_association.source_arn)
        if len(feature_group_association_set) != len(input_feature_group_contexts):
            return False
        for feature_group in input_feature_group_contexts:
            if feature_group.pipeline_version_context_arn not in feature_group_association_set:
                return False
        return True

    def _update_last_transformation_code(
        self, upstream_transformation_code_associations: Iterator[AssociationSummary]
    ) -> None:
        """Compare the existing and the new upstream Transformation Code.

        Arguments:
            upstream_transformation_code_associations (Iterator[AssociationSummary]):
                Upstream existing transformation code associations for the pipeline.

        Returns:
            bool: Boolean if old and new upstream is same.
        """
        upstream_transformation_code = next(upstream_transformation_code_associations, None)
        if upstream_transformation_code is None:
            return

        last_transformation_code_artifact = S3LineageEntityHandler.load_artifact_from_arn(
            artifact_arn=upstream_transformation_code.source_arn,
            sagemaker_session=self.sagemaker_session,
        )
        logger.info(
            "Retrieved previous transformation code artifact: %s", last_transformation_code_artifact
        )
        if (
            last_transformation_code_artifact.properties["state"]
            == TRANSFORMATION_CODE_STATUS_ACTIVE
        ):
            last_transformation_code_artifact.properties[
                "state"
            ] = TRANSFORMATION_CODE_STATUS_INACTIVE
            last_transformation_code_artifact.properties["exclusive_end_date"] = self.pipeline[
                LAST_MODIFIED_TIME
            ].strftime("%s")
            S3LineageEntityHandler.update_transformation_code_artifact(
                transformation_code_artifact=last_transformation_code_artifact
            )
            logger.info("Updated the last transformation artifact")

    def _get_pipeline_context(self) -> Context:
        """Retrieve Pipeline Context.

        Returns:
            Context: The Pipeline Context.
        """
        return PipelineLineageEntityHandler.load_pipeline_context(
            pipeline_name=self.pipeline_name,
            creation_time=self.pipeline[CREATION_TIME].strftime("%s"),
            sagemaker_session=self.sagemaker_session,
        )

    def _get_pipeline_version_context(self, last_update_time: str) -> Context:
        """Retrieve Pipeline Version Context.

        Returns:
            Context: The Pipeline Version Context.
        """
        return PipelineVersionLineageEntityHandler.load_pipeline_version_context(
            pipeline_name=self.pipeline_name,
            last_update_time=last_update_time,
            sagemaker_session=self.sagemaker_session,
        )

    def _check_if_pipeline_lineage_exists(self) -> bool:
        """Check if Pipeline Lineage exists.

        Returns:
            bool: Check if pipeline lineage exists.
        """
        try:
            PipelineLineageEntityHandler.load_pipeline_context(
                pipeline_name=self.pipeline_name,
                creation_time=self.pipeline[CREATION_TIME].strftime("%s"),
                sagemaker_session=self.sagemaker_session,
            )
            return True
        except ClientError as e:
            if e.response[ERROR][CODE] == RESOURCE_NOT_FOUND:
                return False
            raise e

    def _retrieve_input_feature_group_contexts(self) -> List[FeatureGroupContexts]:
        """Retrieve input Feature Groups' Context ARNs.

        Returns:
            List[FeatureGroupContexts]: List of Input Feature Groups for the pipeline.
        """
        feature_group_contexts: List[FeatureGroupContexts] = list()
        feature_group_input_set: Set[str] = set()
        for data_source in self.inputs:
            if isinstance(data_source, FeatureGroupDataSource):
                feature_group_name: str = FeatureGroupLineageEntityHandler.parse_name_from_arn(
                    data_source.name
                )
                if feature_group_name not in feature_group_input_set:
                    feature_group_input_set.add(feature_group_name)
                    feature_group_contexts.append(
                        FeatureGroupLineageEntityHandler.retrieve_feature_group_context_arns(
                            feature_group_name=data_source.name,
                            sagemaker_session=self.sagemaker_session,
                        )
                    )
        return feature_group_contexts

    def _retrieve_output_feature_group_contexts(self) -> FeatureGroupContexts:
        """Retrieve output Feature Group's Context ARNs.

        Returns:
            FeatureGroupContexts: The output Feature Group for the pipeline.
        """
        return FeatureGroupLineageEntityHandler.retrieve_feature_group_context_arns(
            feature_group_name=self.output, sagemaker_session=self.sagemaker_session
        )

    def _create_pipeline_lineage_for_new_pipeline(self) -> Context:
        """Create Pipeline Context for a new pipeline.

        Returns:
            Context: The Pipeline Context.
        """
        return PipelineLineageEntityHandler.create_pipeline_context(
            pipeline_name=self.pipeline_name,
            pipeline_arn=self.pipeline_arn,
            creation_time=self.pipeline[CREATION_TIME].strftime("%s"),
            last_update_time=self.pipeline[LAST_MODIFIED_TIME].strftime("%s"),
            sagemaker_session=self.sagemaker_session,
        )

    def _create_pipeline_version_lineage(self) -> Context:
        """Create a new Pipeline Version Context.

        Returns:
            Context: The Pipeline Versions Context.
        """
        return PipelineVersionLineageEntityHandler.create_pipeline_version_context(
            pipeline_name=self.pipeline_name,
            pipeline_arn=self.pipeline_arn,
            last_update_time=self.pipeline[LAST_MODIFIED_TIME].strftime("%s"),
            sagemaker_session=self.sagemaker_session,
        )

    def _add_associations_for_pipeline(
        self,
        pipeline_context_arn: str,
        pipeline_versions_context_arn: str,
        input_feature_group_contexts: List[FeatureGroupContexts],
        input_raw_data_artifacts: List[Artifact],
        output_feature_group_contexts: FeatureGroupContexts,
        transformation_code_artifact: Optional[Artifact] = None,
    ) -> None:
        """Add Feature Processor Lineage Associations for the Pipeline

        Arguments:
            pipeline_context_arn (str): The pipeline Context ARN.
            pipeline_versions_context_arn (str): The pipeline Version Context ARN.
            input_feature_group_contexts (List[FeatureGroupContexts]): List of input FeatureGroups.
            input_raw_data_artifacts (List[Artifact]): List of input raw data.
            output_feature_group_contexts (FeatureGroupContexts): Output Feature Group
            transformation_code_artifact (Optional[Artifact]): The transformation Code.
        """
        LineageAssociationHandler.add_upstream_feature_group_data_associations(
            feature_group_inputs=input_feature_group_contexts,
            pipeline_context_arn=pipeline_context_arn,
            pipeline_version_context_arn=pipeline_versions_context_arn,
            sagemaker_session=self.sagemaker_session,
        )

        LineageAssociationHandler.add_downstream_feature_group_data_associations(
            feature_group_output=output_feature_group_contexts,
            pipeline_context_arn=pipeline_context_arn,
            pipeline_version_context_arn=pipeline_versions_context_arn,
            sagemaker_session=self.sagemaker_session,
        )

        LineageAssociationHandler.add_upstream_raw_data_associations(
            raw_data_inputs=input_raw_data_artifacts,
            pipeline_context_arn=pipeline_context_arn,
            pipeline_version_context_arn=pipeline_versions_context_arn,
            sagemaker_session=self.sagemaker_session,
        )

        if transformation_code_artifact is not None:
            LineageAssociationHandler.add_upstream_transformation_code_associations(
                transformation_code_artifact=transformation_code_artifact,
                pipeline_version_context_arn=pipeline_versions_context_arn,
                sagemaker_session=self.sagemaker_session,
            )
