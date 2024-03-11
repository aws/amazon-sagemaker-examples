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
"""Contains class to store Feature Group Contexts"""
from __future__ import absolute_import
import attr


@attr.s
class FeatureGroupContexts:
    """A Feature Group Context data source.

    Attributes:
        feature_group_name (str): The name of the Feature Group.
        feature_group_pipeline_context_arn (str): The ARN of the Feature Group Pipeline Context.
        feature_group_pipeline_version_context_arn (str):
            The ARN of the Feature Group Versions Context
    """

    name: str = attr.ib()
    pipeline_context_arn: str = attr.ib()
    pipeline_version_context_arn: str = attr.ib()
