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
"""Pipeline experiment config for SageMaker pipeline."""
from __future__ import absolute_import


class PipelineDefinitionConfig:
    """Pipeline Definition Configuration for SageMaker pipeline."""

    def __init__(self, use_custom_job_prefix: bool):
        """Create a `PipelineDefinitionConfig`.

        Examples: Use a `PipelineDefinitionConfig` to turn on custom job prefixing::

            PipelineDefinitionConfig(use_custom_job_prefix=True)

        Args:
            use_custom_job_prefix (bool): A feature flag to toggle on/off custom name prefixing
                during pipeline orchestration.
        """
        self.use_custom_job_prefix = use_custom_job_prefix
