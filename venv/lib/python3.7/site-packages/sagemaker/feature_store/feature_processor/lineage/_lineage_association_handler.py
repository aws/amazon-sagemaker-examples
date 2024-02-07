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
from typing import List, Optional, Iterator
from botocore.exceptions import ClientError

from sagemaker import Session
from sagemaker.feature_store.feature_processor.lineage._feature_group_contexts import (
    FeatureGroupContexts,
)
from sagemaker.feature_store.feature_processor._constants import VALIDATION_EXCEPTION
from sagemaker.feature_store.feature_processor.lineage.constants import (
    CONTRIBUTED_TO,
    ERROR,
    CODE,
    SAGEMAKER,
    ASSOCIATED_WITH,
)
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association, AssociationSummary

logger = logging.getLogger(SAGEMAKER)


class LineageAssociationHandler:
    """Class to handler the FeatureProcessor Lineage Associations"""

    @staticmethod
    def add_upstream_feature_group_data_associations(
        feature_group_inputs: List[FeatureGroupContexts],
        pipeline_context_arn: str,
        pipeline_version_context_arn: str,
        sagemaker_session: Session,
    ) -> None:
        """Add the FeatureProcessor Upstream Feature Group Lineage Associations.

        Arguments:
            feature_group_inputs (List[FeatureGroupContexts]): The input Feature Group List.
            pipeline_context_arn (str): The pipeline context arn.
            pipeline_version_context_arn (str): The pipeline version context arn.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        for feature_group in feature_group_inputs:
            LineageAssociationHandler._add_association(
                source_arn=feature_group.pipeline_context_arn,
                destination_arn=pipeline_context_arn,
                association_type=CONTRIBUTED_TO,
                sagemaker_session=sagemaker_session,
            )
            LineageAssociationHandler._add_association(
                source_arn=feature_group.pipeline_version_context_arn,
                destination_arn=pipeline_version_context_arn,
                association_type=CONTRIBUTED_TO,
                sagemaker_session=sagemaker_session,
            )

    @staticmethod
    def add_upstream_raw_data_associations(
        raw_data_inputs: List[Artifact],
        pipeline_context_arn: str,
        pipeline_version_context_arn: str,
        sagemaker_session: Session,
    ) -> None:
        """Add the FeatureProcessor Upstream Raw Data Lineage Associations.

        Arguments:
            raw_data_inputs (List[Artifact]): The input raw data List.
            pipeline_context_arn (str): The pipeline context arn.
            pipeline_version_context_arn (str): The pipeline version context arn.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        for raw_data_artifact in raw_data_inputs:
            LineageAssociationHandler._add_association(
                source_arn=raw_data_artifact.artifact_arn,
                destination_arn=pipeline_context_arn,
                association_type=CONTRIBUTED_TO,
                sagemaker_session=sagemaker_session,
            )
            LineageAssociationHandler._add_association(
                source_arn=raw_data_artifact.artifact_arn,
                destination_arn=pipeline_version_context_arn,
                association_type=CONTRIBUTED_TO,
                sagemaker_session=sagemaker_session,
            )

    @staticmethod
    def add_upstream_transformation_code_associations(
        transformation_code_artifact: Artifact,
        pipeline_version_context_arn: str,
        sagemaker_session: Session,
    ) -> None:
        """Add the FeatureProcessor Upstream Transformation Code Lineage Associations.

        Arguments:
            transformation_code_artifact (Artifact): The transformation Code Artifact.
            pipeline_version_context_arn (str): The pipeline version context arn.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        LineageAssociationHandler._add_association(
            source_arn=transformation_code_artifact.artifact_arn,
            destination_arn=pipeline_version_context_arn,
            association_type=CONTRIBUTED_TO,
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def add_upstream_schedule_associations(
        schedule_artifact: Artifact,
        pipeline_version_context_arn: str,
        sagemaker_session: Session,
    ) -> None:
        """Add the FeatureProcessor Upstream Schedule Lineage Associations.

        Arguments:
            schedule_artifact (Artifact): The schedule Artifact.
            pipeline_version_context_arn (str): The pipeline version context arn.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        LineageAssociationHandler._add_association(
            source_arn=schedule_artifact.artifact_arn,
            destination_arn=pipeline_version_context_arn,
            association_type=CONTRIBUTED_TO,
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def add_downstream_feature_group_data_associations(
        feature_group_output: FeatureGroupContexts,
        pipeline_context_arn: str,
        pipeline_version_context_arn: str,
        sagemaker_session: Session,
    ) -> None:
        """Add the FeatureProcessor Downstream Feature Group Lineage Associations.

        Arguments:
            feature_group_output (FeatureGroupContexts): The output Feature Group.
            pipeline_context_arn (str): The pipeline context arn.
            pipeline_version_context_arn (str): The pipeline version context arn.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        LineageAssociationHandler._add_association(
            source_arn=pipeline_context_arn,
            destination_arn=feature_group_output.pipeline_context_arn,
            association_type=CONTRIBUTED_TO,
            sagemaker_session=sagemaker_session,
        )
        LineageAssociationHandler._add_association(
            source_arn=pipeline_version_context_arn,
            destination_arn=feature_group_output.pipeline_version_context_arn,
            association_type="ContributedTo",
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def add_pipeline_and_pipeline_version_association(
        pipeline_context_arn: str,
        pipeline_version_context_arn: str,
        sagemaker_session: Session,
    ) -> None:
        """Add the FeatureProcessor Lineage Association

        between the Pipeline and the Pipeline Versions.

        Arguments:
            pipeline_context_arn (str): The pipeline context arn.
            pipeline_version_context_arn (str): The pipeline version context arn.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        LineageAssociationHandler._add_association(
            source_arn=pipeline_context_arn,
            destination_arn=pipeline_version_context_arn,
            association_type=ASSOCIATED_WITH,
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def list_upstream_associations(
        entity_arn: str, source_type: str, sagemaker_session: Session
    ) -> Iterator[AssociationSummary]:
        """List Upstream Lineage Associations.

        Arguments:
            entity_arn (str): The Lineage Entity ARN.
            source_type (str): The Source Type.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        return LineageAssociationHandler._list_association(
            destination_arn=entity_arn,
            source_type=source_type,
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def list_downstream_associations(
        entity_arn: str, destination_type: str, sagemaker_session: Session
    ) -> Iterator[AssociationSummary]:
        """List Downstream Lineage Associations.

        Arguments:
            entity_arn (str): The Lineage Entity ARN.
            destination_type (str): The Destination Type.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        return LineageAssociationHandler._list_association(
            source_arn=entity_arn,
            destination_type=destination_type,
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def _add_association(
        source_arn: str,
        destination_arn: str,
        association_type: str,
        sagemaker_session: Session,
    ) -> None:
        """Add Lineage Association.

        Arguments:
            source_arn (str): The source ARN.
            destination_arn (str): The destination ARN.
            association_type (str): The association type.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        try:
            logger.info(
                "Adding association with source_arn: "
                "%s, destination_arn: %s and association_type: %s.",
                source_arn,
                destination_arn,
                association_type,
            )
            Association.create(
                source_arn=source_arn,
                destination_arn=destination_arn,
                association_type=association_type,
                sagemaker_session=sagemaker_session,
            )
        except ClientError as e:
            if e.response[ERROR][CODE] == VALIDATION_EXCEPTION:
                logger.info("Association already exists")
            else:
                raise e

    @staticmethod
    def _list_association(
        sagemaker_session: Session,
        source_arn: Optional[str] = None,
        source_type: Optional[str] = None,
        destination_arn: Optional[str] = None,
        destination_type: Optional[str] = None,
    ) -> Iterator[AssociationSummary]:
        """List Lineage Associations.

        Arguments:
            source_arn (str): The source ARN.
            source_type (str): The source type.
            destination_arn (str): The destination ARN.
            destination_type (str): The destination type.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.
        """
        return Association.list(
            source_arn=source_arn,
            source_type=source_type,
            destination_arn=destination_arn,
            destination_type=destination_type,
            sagemaker_session=sagemaker_session,
        )
