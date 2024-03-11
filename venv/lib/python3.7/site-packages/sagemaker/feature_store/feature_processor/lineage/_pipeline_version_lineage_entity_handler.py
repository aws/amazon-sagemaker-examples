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
"""Contains class to handle Pipeline Version Lineage"""
from __future__ import absolute_import
import logging

from sagemaker import Session
from sagemaker.feature_store.feature_processor.lineage.constants import (
    SAGEMAKER,
    PIPELINE_VERSION_CONTEXT_TYPE,
    PIPELINE_NAME_KEY,
    LAST_UPDATE_TIME_KEY,
)
from sagemaker.lineage.context import Context

# pylint: disable=C0301
from sagemaker.feature_store.feature_processor.lineage._feature_processor_lineage_name_helper import (
    _get_feature_processor_pipeline_version_lineage_context_name,
)

logger = logging.getLogger(SAGEMAKER)


class PipelineVersionLineageEntityHandler:
    """Class for handling FeatureProcessor Pipeline Version Lineage"""

    @staticmethod
    def create_pipeline_version_context(
        pipeline_name: str,
        pipeline_arn: str,
        last_update_time: str,
        sagemaker_session: Session,
    ) -> Context:
        """Create the FeatureProcessor Pipeline Version context.

        Arguments:
            pipeline_name (str): The pipeline name.
            pipeline_arn (str): The pipeline ARN.
            last_update_time (str): The pipeline last update time.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Context: The pipeline version context.
        """
        return Context.create(
            context_name=_get_feature_processor_pipeline_version_lineage_context_name(
                pipeline_name, last_update_time
            ),
            context_type=f"{PIPELINE_VERSION_CONTEXT_TYPE}-{pipeline_name}",
            source_uri=pipeline_arn,
            source_type=last_update_time,
            properties={
                PIPELINE_NAME_KEY: pipeline_name,
                LAST_UPDATE_TIME_KEY: last_update_time,
            },
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def load_pipeline_version_context(
        pipeline_name: str, last_update_time: str, sagemaker_session: Session
    ) -> Context:
        """Load the FeatureProcessor Pipeline Version context.

        Arguments:
            pipeline_name (str): The pipeline name.
            last_update_time (str): The pipeline last update time.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Context: The pipeline version context.
        """
        return Context.load(
            context_name=_get_feature_processor_pipeline_version_lineage_context_name(
                pipeline_name, last_update_time
            ),
            sagemaker_session=sagemaker_session,
        )
