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

from sagemaker.workflow.parameters import Parameter, ParameterTypeEnum

DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_SCHEDULE_STATE = "ENABLED"
UNDERSCORE = "_"
RESOURCE_NOT_FOUND_EXCEPTION = "ResourceNotFoundException"
RESOURCE_NOT_FOUND = "ResourceNotFound"
EXECUTION_TIME_PIPELINE_PARAMETER = "scheduled_time"
VALIDATION_EXCEPTION = "ValidationException"
EVENT_BRIDGE_INVOCATION_TIME = "<aws.scheduler.scheduled-time>"
SCHEDULED_TIME_PIPELINE_PARAMETER = Parameter(
    name=EXECUTION_TIME_PIPELINE_PARAMETER, parameter_type=ParameterTypeEnum.STRING
)
EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # 2023-01-01T07:00:00Z
NO_FLEXIBLE_TIME_WINDOW = dict(Mode="OFF")
PIPELINE_NAME_MAXIMUM_LENGTH = 80
PIPELINE_CONTEXT_TYPE = "FeatureEngineeringPipeline"
SPARK_JAR_FILES_PATH = "submit_jars_s3_paths"
SPARK_PY_FILES_PATH = "submit_py_files_s3_paths"
SPARK_FILES_PATH = "submit_files_s3_path"
FEATURE_PROCESSOR_TAG_KEY = "sm-fs-fe:created-from"
FEATURE_PROCESSOR_TAG_VALUE = "fp-to-pipeline"
FEATURE_GROUP_ARN_REGEX_PATTERN = r"arn:(.*?):sagemaker:(.*?):(.*?):feature-group/(.*?)$"
SAGEMAKER_WHL_FILE_S3_PATH = "s3://ada-private-beta/sagemaker-2.151.1.dev0-py2.py3-none-any.whl"
S3_DATA_DISTRIBUTION_TYPE = "FullyReplicated"
PIPELINE_CONTEXT_NAME_TAG_KEY = "sm-fs-fe:feature-engineering-pipeline-context-name"
PIPELINE_VERSION_CONTEXT_NAME_TAG_KEY = "sm-fs-fe:feature-engineering-pipeline-version-context-name"
TO_PIPELINE_RESERVED_TAG_KEYS = [
    FEATURE_PROCESSOR_TAG_KEY,
    PIPELINE_CONTEXT_NAME_TAG_KEY,
    PIPELINE_VERSION_CONTEXT_NAME_TAG_KEY,
]
