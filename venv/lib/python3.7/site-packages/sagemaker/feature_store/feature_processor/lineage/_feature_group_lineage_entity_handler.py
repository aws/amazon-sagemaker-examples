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
"""Contains class to handle Feature Processor Lineage"""
from __future__ import absolute_import

import re
from typing import Dict, Any
import logging

from sagemaker import Session
from sagemaker.feature_store.feature_processor._constants import FEATURE_GROUP_ARN_REGEX_PATTERN
from sagemaker.feature_store.feature_processor.lineage._feature_group_contexts import (
    FeatureGroupContexts,
)
from sagemaker.feature_store.feature_processor.lineage.constants import (
    SAGEMAKER,
    FEATURE_GROUP,
    CREATION_TIME,
)
from sagemaker.lineage.context import Context

# pylint: disable=C0301
from sagemaker.feature_store.feature_processor.lineage._feature_processor_lineage_name_helper import (
    _get_feature_group_pipeline_lineage_context_name,
    _get_feature_group_pipeline_version_lineage_context_name,
)

logger = logging.getLogger(SAGEMAKER)


class FeatureGroupLineageEntityHandler:
    """Class for handling Feature Group Lineage"""

    @staticmethod
    def retrieve_feature_group_context_arns(
        feature_group_name: str, sagemaker_session: Session
    ) -> FeatureGroupContexts:
        """Retrieve Feature Group Contexts.

        Arguments:
            feature_group_name (str): The Feature Group Name.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            FeatureGroupContexts: The Feature Group Pipeline and Version Context.
        """
        feature_group = FeatureGroupLineageEntityHandler._describe_feature_group(
            feature_group_name=FeatureGroupLineageEntityHandler.parse_name_from_arn(
                feature_group_name
            ),
            sagemaker_session=sagemaker_session,
        )
        feature_group_name = feature_group[FEATURE_GROUP]
        feature_group_creation_time = feature_group[CREATION_TIME].strftime("%s")
        feature_group_pipeline_context = (
            FeatureGroupLineageEntityHandler._load_feature_group_pipeline_context(
                feature_group_name=feature_group_name,
                feature_group_creation_time=feature_group_creation_time,
                sagemaker_session=sagemaker_session,
            )
        )
        feature_group_pipeline_version_context = (
            FeatureGroupLineageEntityHandler._load_feature_group_pipeline_version_context(
                feature_group_name=feature_group_name,
                feature_group_creation_time=feature_group_creation_time,
                sagemaker_session=sagemaker_session,
            )
        )
        return FeatureGroupContexts(
            name=feature_group_name,
            pipeline_context_arn=feature_group_pipeline_context.context_arn,
            pipeline_version_context_arn=feature_group_pipeline_version_context.context_arn,
        )

    @staticmethod
    def _describe_feature_group(
        feature_group_name: str, sagemaker_session: Session
    ) -> Dict[str, Any]:
        """Retrieve the Feature Group.

        Arguments:
            feature_group_name (str): The Feature Group Name.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Dict[str, Any]: The Feature Group details.
        """
        feature_group = sagemaker_session.describe_feature_group(
            feature_group_name=feature_group_name
        )
        logger.debug(
            "Called describe_feature_group with %s and received: %s",
            feature_group_name,
            feature_group,
        )
        return feature_group

    @staticmethod
    def _load_feature_group_pipeline_context(
        feature_group_name: str,
        feature_group_creation_time: str,
        sagemaker_session: Session,
    ) -> Context:
        """Retrieve Feature Group Pipeline Context

        Arguments:
            feature_group_name (str): The Feature Group Name.
            feature_group_creation_time (str): The Feature Group Creation Time,
                in long epoch seconds.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Context: The Feature Group Pipeline Context.
        """
        feature_group_pipeline_context = _get_feature_group_pipeline_lineage_context_name(
            feature_group_name=feature_group_name,
            feature_group_creation_time=feature_group_creation_time,
        )
        return Context.load(
            context_name=feature_group_pipeline_context,
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def _load_feature_group_pipeline_version_context(
        feature_group_name: str,
        feature_group_creation_time: str,
        sagemaker_session: Session,
    ) -> Context:
        """Retrieve Feature Group Pipeline Version Context

        Arguments:
            feature_group_name (str): The Feature Group Name.
            feature_group_creation_time (str): The Feature Group Creation Time,
                in long epoch seconds.
            sagemaker_session (Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified, the
                function creates one using the default AWS configuration chain.

        Returns:
            Context: The Feature Group Pipeline Version Context.
        """
        feature_group_pipeline_version_context = (
            _get_feature_group_pipeline_version_lineage_context_name(
                feature_group_name=feature_group_name,
                feature_group_creation_time=feature_group_creation_time,
            )
        )
        return Context.load(
            context_name=feature_group_pipeline_version_context,
            sagemaker_session=sagemaker_session,
        )

    @staticmethod
    def parse_name_from_arn(fg_uri: str) -> str:
        """Parse the name from a string, if it's an ARN. Otherwise, return the string.

        Arguments:
            fg_uri (str): The Feature Group Name or ARN.

        Returns:
            str: The Feature Group Name.
        """
        match = re.match(FEATURE_GROUP_ARN_REGEX_PATTERN, fg_uri)
        if match:
            feature_group_name = match.group(4)
            return feature_group_name
        return fg_uri
