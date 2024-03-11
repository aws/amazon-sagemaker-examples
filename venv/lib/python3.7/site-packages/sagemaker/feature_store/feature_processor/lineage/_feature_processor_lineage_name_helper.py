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
"""Contains class to handle lineage resource name generation."""
from __future__ import absolute_import

FEATURE_PROCESSOR_CREATED_PREFIX = "sm-fs-fe"
FEATURE_GROUP_PIPELINE_CONTEXT_SUFFIX = "feature-group-pipeline"
FEATURE_GROUP_PIPELINE_CONTEXT_VERSION_SUFFIX = "feature-group-pipeline-version"
FEATURE_PROCESSOR_PIPELINE_CONTEXT_SUFFIX = "fep"
FEATURE_PROCESSOR_PIPELINE_VERSION_CONTEXT_SUFFIX = "fep-ver"


def _get_feature_processor_lineage_context_name(
    resource_name: str,
    resource_creation_time: str,
    lineage_context_prefix: str = None,
    lineage_context_suffix: str = None,
) -> str:
    """Generic naming generation function for lineage resources used by feature_processor."""
    context_name_base = [f"{resource_name}-{resource_creation_time}"]
    if lineage_context_prefix:
        context_name_base.insert(0, lineage_context_prefix)
    if lineage_context_suffix:
        context_name_base.append(lineage_context_suffix)
    return "-".join(context_name_base)


def _get_feature_group_lineage_context_name(
    feature_group_name: str, feature_group_creation_time: str
) -> str:
    """Generate context name for feature group contexts."""
    return _get_feature_processor_lineage_context_name(
        resource_name=feature_group_name, resource_creation_time=feature_group_creation_time
    )


def _get_feature_group_pipeline_lineage_context_name(
    feature_group_name: str, feature_group_creation_time: str
) -> str:
    """Generate context name for feature group pipeline."""
    return _get_feature_processor_lineage_context_name(
        resource_name=feature_group_name,
        resource_creation_time=feature_group_creation_time,
        lineage_context_suffix=FEATURE_GROUP_PIPELINE_CONTEXT_SUFFIX,
    )


def _get_feature_group_pipeline_version_lineage_context_name(
    feature_group_name: str, feature_group_creation_time: str
) -> str:
    """Generate context name for feature group pipeline version."""
    return _get_feature_processor_lineage_context_name(
        resource_name=feature_group_name,
        resource_creation_time=feature_group_creation_time,
        lineage_context_suffix=FEATURE_GROUP_PIPELINE_CONTEXT_VERSION_SUFFIX,
    )


def _get_feature_processor_pipeline_lineage_context_name(
    pipeline_name: str, pipeline_creation_time: str
) -> str:
    """Generate context name for feature processor pipeline."""
    return _get_feature_processor_lineage_context_name(
        resource_name=pipeline_name,
        resource_creation_time=pipeline_creation_time,
        lineage_context_prefix=FEATURE_PROCESSOR_CREATED_PREFIX,
        lineage_context_suffix=FEATURE_PROCESSOR_PIPELINE_CONTEXT_SUFFIX,
    )


def _get_feature_processor_pipeline_version_lineage_context_name(
    pipeline_name: str, pipeline_last_update_time: str
) -> str:
    """Generate context name for feature processor pipeline version."""
    return _get_feature_processor_lineage_context_name(
        resource_name=pipeline_name,
        resource_creation_time=pipeline_last_update_time,
        lineage_context_prefix=FEATURE_PROCESSOR_CREATED_PREFIX,
        lineage_context_suffix=FEATURE_PROCESSOR_PIPELINE_VERSION_CONTEXT_SUFFIX,
    )


def _get_feature_processor_schedule_lineage_artifact_name(schedule_name: str) -> str:
    """Generate artifact name for feature processor pipeline schedule."""
    return "-".join([FEATURE_PROCESSOR_CREATED_PREFIX, schedule_name])
