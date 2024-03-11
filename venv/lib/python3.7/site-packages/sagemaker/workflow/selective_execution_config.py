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
from typing import List
from sagemaker.workflow.entities import RequestType


class SelectiveExecutionConfig:
    """The selective execution configuration, which defines a subset of pipeline steps to run in

    another SageMaker pipeline run.
    """

    def __init__(self, selected_steps: List[str], source_pipeline_execution_arn: str = None):
        """Create a `SelectiveExecutionConfig`.

        Args:
            source_pipeline_execution_arn (str): The ARN from a reference execution of the
                current pipeline. Used to copy input collaterals needed for the selected
                steps to run. The execution status of the pipeline can be `Stopped`, `Failed`, or
                `Succeeded`.
            selected_steps (List[str]): A list of pipeline steps to run. All step(s) in all
                path(s) between two selected steps should be included.
        """
        self.source_pipeline_execution_arn = source_pipeline_execution_arn
        self.selected_steps = selected_steps

    def _build_selected_steps_from_list(self) -> RequestType:
        """Get the request structure for the list of selected steps."""
        selected_step_list = []
        for selected_step in self.selected_steps:
            selected_step_list.append(dict(StepName=selected_step))
        return selected_step_list

    def to_request(self) -> RequestType:
        """Convert `SelectiveExecutionConfig` object to request dict."""
        request = {}

        if self.source_pipeline_execution_arn is not None:
            request["SourcePipelineExecutionArn"] = self.source_pipeline_execution_arn

        if self.selected_steps is not None:
            request["SelectedSteps"] = self._build_selected_steps_from_list()

        return request
