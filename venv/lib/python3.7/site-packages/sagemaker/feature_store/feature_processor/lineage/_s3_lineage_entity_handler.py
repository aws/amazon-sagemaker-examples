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
"""Contains class to handle S3 Lineage"""
from __future__ import absolute_import
import logging
from typing import Union, Optional, List

from sagemaker import Session
from sagemaker.feature_store.feature_processor import CSVDataSource, ParquetDataSource

# pylint: disable=C0301
from sagemaker.feature_store.feature_processor.lineage._feature_processor_lineage_name_helper import (
    _get_feature_processor_schedule_lineage_artifact_name,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_schedule import (
    PipelineSchedule,
)
from sagemaker.feature_store.feature_processor.lineage._transformation_code import (
    TransformationCode,
)
from sagemaker.feature_store.feature_processor.lineage.constants import (
    TRANSFORMATION_CODE_STATUS_ACTIVE,
    FEP_LINEAGE_PREFIX,
    TRANSFORMATION_CODE_ARTIFACT_NAME,
)
from sagemaker.lineage.artifact import Artifact, ArtifactSummary

logger = logging.getLogger("sagemaker")


class S3LineageEntityHandler:
    """Class for handling FeatureProcessor S3 Artifact Lineage"""

    @staticmethod
    def retrieve_raw_data_artifact(
        raw_data: Union[CSVDataSource, ParquetDataSource], sagemaker_session: Session
    ) -> Artifact:
        """Load or create the FeatureProcessor Pipeline's raw data Artifact.

        Arguments:
            raw_data (Union[CSVDataSource, ParquetDataSource]): The raw data to be retrieved.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Artifact: The raw data artifact.
        """
        load_artifact: ArtifactSummary = S3LineageEntityHandler._load_artifact_from_s3_uri(
            s3_uri=raw_data.s3_uri, sagemaker_session=sagemaker_session
        )
        if load_artifact is not None:
            return S3LineageEntityHandler.load_artifact_from_arn(
                artifact_arn=load_artifact.artifact_arn,
                sagemaker_session=sagemaker_session,
            )
        return S3LineageEntityHandler._create_artifact(
            s3_uri=raw_data.s3_uri,
            artifact_type="DataSet",
            artifact_name="sm-fs-fe-raw-data",
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def update_transformation_code_artifact(
        transformation_code_artifact: Artifact,
    ) -> None:
        """Update Pipeline's transformation code Artifact.

        Arguments:
            transformation_code_artifact (TransformationCode): The transformation code Artifact to be updated.
        """
        transformation_code_artifact.save()

    @staticmethod
    def create_transformation_code_artifact(
        transformation_code: TransformationCode,
        pipeline_last_update_time: str,
        sagemaker_session: Session,
    ) -> Optional[Artifact]:
        """Create the FeatureProcessor Pipeline's transformation code Artifact.

        Arguments:
            transformation_code (TransformationCode): The transformation code to be retrieved.
            pipeline_last_update_time (str): The last update time of the pipeline.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Artifact: The transformation code artifact.
        """
        if transformation_code is None:
            return None

        properties = dict(
            state=TRANSFORMATION_CODE_STATUS_ACTIVE,
            inclusive_start_date=pipeline_last_update_time,
        )
        if transformation_code.name is not None:
            properties["name"] = transformation_code.name
        if transformation_code.author is not None:
            properties["author"] = transformation_code.author

        return S3LineageEntityHandler._create_artifact(
            s3_uri=transformation_code.s3_uri,
            source_types=[dict(SourceIdType="Custom", Value=pipeline_last_update_time)],
            properties=properties,
            artifact_type="TransformationCode",
            artifact_name=f"{FEP_LINEAGE_PREFIX}-"
            f"{TRANSFORMATION_CODE_ARTIFACT_NAME}-"
            f"{pipeline_last_update_time}",
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def retrieve_pipeline_schedule_artifact(
        pipeline_schedule: PipelineSchedule,
        sagemaker_session: Session,
        _get_feature_processor_schedule_lineage_artifact_namef=None,
    ) -> Optional[Artifact]:
        """Load or create the FeatureProcessor Pipeline's schedule Artifact

        Arguments:
            pipeline_schedule (PipelineSchedule): Class to hold the Pipeline Schedule details
            sagemaker_session (Session): Session object which manages interactions
            with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
            function creates one using the default AWS configuration chain.

        Returns:
            Artifact: The Schedule Artifact.
        """
        if pipeline_schedule is None:
            return None
        load_artifact: ArtifactSummary = S3LineageEntityHandler._load_artifact_from_s3_uri(
            s3_uri=pipeline_schedule.schedule_arn,
            sagemaker_session=sagemaker_session,
        )
        if load_artifact is not None:
            pipeline_schedule_artifact: Artifact = S3LineageEntityHandler.load_artifact_from_arn(
                artifact_arn=load_artifact.artifact_arn,
                sagemaker_session=sagemaker_session,
            )
            pipeline_schedule_artifact.properties["pipeline_name"] = pipeline_schedule.pipeline_name
            pipeline_schedule_artifact.properties[
                "schedule_expression"
            ] = pipeline_schedule.schedule_expression
            pipeline_schedule_artifact.properties["state"] = pipeline_schedule.state
            pipeline_schedule_artifact.properties["start_date"] = pipeline_schedule.start_date
            pipeline_schedule_artifact.save()
            return pipeline_schedule_artifact

        return S3LineageEntityHandler._create_artifact(
            s3_uri=pipeline_schedule.schedule_arn,
            artifact_type="PipelineSchedule",
            artifact_name=_get_feature_processor_schedule_lineage_artifact_name(
                schedule_name=pipeline_schedule.schedule_name
            ),
            properties=dict(
                pipeline_name=pipeline_schedule.pipeline_name,
                schedule_expression=pipeline_schedule.schedule_expression,
                state=pipeline_schedule.state,
                start_date=pipeline_schedule.start_date,
            ),
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def load_artifact_from_arn(artifact_arn: str, sagemaker_session: Session) -> Artifact:
        """Load Lineage Artifacts from ARN.

        Arguments:
            artifact_arn (str): The Artifact ARN.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Artifact: The Artifact for the provided ARN.
        """
        return Artifact.load(artifact_arn=artifact_arn, sagemaker_session=sagemaker_session)

    @staticmethod
    def _load_artifact_from_s3_uri(
        s3_uri: str, sagemaker_session: Session
    ) -> Optional[ArtifactSummary]:
        """Load FeatureProcessor S3 Lineage Artifacts.

        Arguments:
            s3_uri (str): The s3 uri of the Artifact.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            ArtifactSummary: The Artifact Summary for the provided S3 URI.
        """
        artifacts = Artifact.list(source_uri=s3_uri, sagemaker_session=sagemaker_session)
        for artifact_summary in artifacts:
            # We want to make sure that source_type is empty.
            # Since SDK will not set it while creating artifacts.
            if (
                artifact_summary.source.source_types is None
                or len(artifact_summary.source.source_types) == 0
            ):
                return artifact_summary
        return None

    @staticmethod
    def _create_artifact(
        s3_uri: str,
        artifact_type: str,
        sagemaker_session: Session,
        properties: Optional[dict] = None,
        artifact_name: Optional[str] = None,
        source_types: Optional[List[dict]] = None,
    ) -> Artifact:
        """Create Lineage Artifacts.

        Arguments:
            s3_uri (str): The s3 uri of the Artifact.
            artifact_type (str): The Artifact type.
            properties (Optional[dict]): The properties of the Artifact.
            artifact_name (Optional[str]): The name of the Artifact.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Artifact: The new Artifact.
        """
        return Artifact.create(
            source_uri=s3_uri,
            source_types=source_types,
            artifact_type=artifact_type,
            artifact_name=artifact_name,
            properties=properties,
            sagemaker_session=sagemaker_session,
        )
