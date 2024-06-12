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
"""Module containing constants for feature_processor and feature_scheduler module."""
from __future__ import absolute_import

FEATURE_GROUP_PIPELINE_VERSION_CONTEXT_TYPE = "FeatureGroupPipelineVersion"
PIPELINE_CONTEXT_TYPE = "FeatureEngineeringPipeline"
PIPELINE_VERSION_CONTEXT_TYPE = "FeatureEngineeringPipelineVersion"
PIPELINE_CONTEXT_NAME_SUFFIX = "fep"
PIPELINE_VERSION_CONTEXT_NAME_SUFFIX = "fep-ver"
FEP_LINEAGE_PREFIX = "sm-fs-fe"
DATA_SET = "DataSet"
TRANSFORMATION_CODE = "TransformationCode"
LAST_UPDATE_TIME = "LastUpdateTime"
LAST_MODIFIED_TIME = "LastModifiedTime"
CREATION_TIME = "CreationTime"
RESOURCE_NOT_FOUND = "ResourceNotFound"
ERROR = "Error"
CODE = "Code"
SAGEMAKER = "sagemaker"
CONTRIBUTED_TO = "ContributedTo"
ASSOCIATED_WITH = "AssociatedWith"
FEATURE_GROUP = "FeatureGroupName"
FEATURE_GROUP_PIPELINE_SUFFIX = "feature-group-pipeline"
FEATURE_GROUP_PIPELINE_VERSION_SUFFIX = "feature-group-pipeline-version"
PIPELINE_CONTEXT_NAME_KEY = "pipeline_context_name"
PIPELINE_CONTEXT_VERSION_NAME_KEY = "pipeline_version_context_name"
PIPELINE_NAME_KEY = "PipelineName"
PIPELINE_CREATION_TIME_KEY = "PipelineCreationTime"
LAST_UPDATE_TIME_KEY = "LastUpdateTime"
TRANSFORMATION_CODE_STATUS_ACTIVE = "Active"
TRANSFORMATION_CODE_STATUS_INACTIVE = "Inactive"
TRANSFORMATION_CODE_ARTIFACT_NAME = "transformation-code"
