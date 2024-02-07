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
"""The `Step` definitions for SageMaker Pipelines Workflows."""
from __future__ import absolute_import

from typing import List, Union, Optional

from sagemaker.workflow.entities import (
    RequestType,
    PipelineVariable,
)
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.steps import Step, StepTypeEnum


class FailStep(Step):
    """`FailStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        error_message: Union[str, PipelineVariable] = None,
        display_name: str = None,
        description: str = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Constructs a `FailStep`.

        Args:
            name (str): The name of the `FailStep`. A name is required and must be
                unique within a pipeline.
            error_message (str or PipelineVariable):
                An error message defined by the user.
                Once the `FailStep` is reached, the execution fails and the
                error message is set as the failure reason (default: None).
            display_name (str): The display name of the `FailStep`.
                The display name provides better UI readability. (default: None).
            description (str): The description of the `FailStep` (default: None).
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `FailStep`
                depends on.
                If a listed `Step` name does not exist, an error is returned (default: None).
        """
        super(FailStep, self).__init__(
            name, display_name, description, StepTypeEnum.FAIL, depends_on
        )
        self.error_message = error_message if error_message is not None else ""

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to define the `FailStep`."""
        return dict(ErrorMessage=self.error_message)

    @property
    def properties(self):
        """A `Properties` object is not available for the `FailStep`.

        Executing a `FailStep` will terminate the pipeline.
        `FailStep` properties should not be referenced.
        """
        raise RuntimeError(
            "FailStep is a terminal step and the Properties object is not available for it."
        )
