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
"""Contains class to handle Pipeline Lineage"""
from __future__ import absolute_import
import logging

from sagemaker import Session
from sagemaker.feature_store.feature_processor.lineage.constants import (
    SAGEMAKER,
    PIPELINE_NAME_KEY,
    PIPELINE_CREATION_TIME_KEY,
    LAST_UPDATE_TIME_KEY,
)
from sagemaker.lineage.context import Context

# pylint: disable=C0301
from sagemaker.feature_store.feature_processor.lineage._feature_processor_lineage_name_helper import (
    _get_feature_processor_pipeline_lineage_context_name,
)
from sagemaker.lineage import context

logger = logging.getLogger(SAGEMAKER)


class PipelineLineageEntityHandler:
    """Class for handling FeatureProcessor Pipeline Lineage"""

    @staticmethod
    def create_pipeline_context(
        pipeline_name: str,
        pipeline_arn: str,
        creation_time: str,
        last_update_time: str,
        sagemaker_session: Session,
    ) -> Context:
        """Create the FeatureProcessor Pipeline context.

        Arguments:
            pipeline_name (str): The pipeline name.
            pipeline_arn (str): The pipeline ARN.
            creation_time (str): The pipeline creation time.
            last_update_time (str): The pipeline last update time.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Context: The pipeline context.
        """
        return context.Context.create(
            context_name=_get_feature_processor_pipeline_lineage_context_name(
                pipeline_name, creation_time
            ),
            context_type="FeatureEngineeringPipeline",
            source_uri=pipeline_arn,
            source_type=creation_time,
            properties={
                PIPELINE_NAME_KEY: pipeline_name,
                PIPELINE_CREATION_TIME_KEY: creation_time,
                LAST_UPDATE_TIME_KEY: last_update_time,
            },
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def load_pipeline_context(
        pipeline_name: str, creation_time: str, sagemaker_session: Session
    ) -> Context:
        """Load the FeatureProcessor Pipeline context.

        Arguments:
            pipeline_name (str): The pipeline name.
            creation_time (str): The pipeline creation time.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Context: The pipeline context.
        """
        return Context.load(
            context_name=_get_feature_processor_pipeline_lineage_context_name(
                pipeline_name, creation_time
            ),
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def update_pipeline_context(pipeline_context: Context) -> None:
        """Update the FeatureProcessor Pipeline context

        Arguments:
            pipeline_context (Context): The pipeline context.
        """
        pipeline_context.save()
