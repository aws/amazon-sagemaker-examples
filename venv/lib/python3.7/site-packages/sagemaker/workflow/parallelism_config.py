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
"""Pipeline Parallelism Configuration"""
from __future__ import absolute_import
from sagemaker.workflow.entities import RequestType


class ParallelismConfiguration:
    """Parallelism config for SageMaker pipeline."""

    def __init__(self, max_parallel_execution_steps: int):
        """Create a ParallelismConfiguration

        Args:
            max_parallel_execution_steps, int:
                max number of steps which could be parallelized
        """
        self.max_parallel_execution_steps = max_parallel_execution_steps

    def to_request(self) -> RequestType:
        """Returns: the request structure."""
        return {
            "MaxParallelExecutionSteps": self.max_parallel_execution_steps,
        }
