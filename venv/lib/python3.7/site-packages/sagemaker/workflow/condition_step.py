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
"""The step definitions for workflow."""
from __future__ import absolute_import

from typing import List, Union, Optional


from sagemaker.deprecations import deprecated_class
from sagemaker.workflow.conditions import Condition
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.functions import JsonGet as NewJsonGet
from sagemaker.workflow.steps import (
    Step,
    StepTypeEnum,
)
from sagemaker.workflow.utilities import list_to_request
from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.properties import (
    Properties,
    PropertyFile,
)


class ConditionStep(Step):
    """Conditional step for pipelines to support conditional branching in the execution of steps."""

    def __init__(
        self,
        name: str,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        display_name: str = None,
        description: str = None,
        conditions: List[Condition] = None,
        if_steps: List[Union[Step, StepCollection]] = None,
        else_steps: List[Union[Step, StepCollection]] = None,
    ):
        """Construct a ConditionStep for pipelines to support conditional branching.

        If all of the conditions in the condition list evaluate to True, the `if_steps` are
        marked as ready for execution. Otherwise, the `else_steps` are marked as ready for
        execution.

        Args:
            name (str): The name of the condition step.
            depends_on (List[Union[str, Step, StepCollection]]): The list of `Step`/StepCollection`
                names or `Step` instances or `StepCollection` instances that the current `Step`
                depends on.
            display_name (str): The display name of the condition step.
            description (str): The description of the condition step.
            conditions (List[Condition]): A list of `sagemaker.workflow.conditions.Condition`
                instances.
            if_steps (List[Union[Step, StepCollection]]): A list of `sagemaker.workflow.steps.Step`
                or `sagemaker.workflow.step_collections.StepCollection` instances that are
                marked as ready for execution if the list of conditions evaluates to True.
            else_steps (List[Union[Step, StepCollection]]): A list of `sagemaker.workflow.steps.Step`
                or `sagemaker.workflow.step_collections.StepCollection` instances that are
                marked as ready for execution if the list of conditions evaluates to False.
        """
        super(ConditionStep, self).__init__(
            name, display_name, description, StepTypeEnum.CONDITION, depends_on
        )
        self.conditions = conditions or []
        self.if_steps = if_steps or []
        self.else_steps = else_steps or []

        root_prop = Properties(step_name=name)
        root_prop.__dict__["Outcome"] = Properties(step_name=name, path="Outcome")
        self._properties = root_prop

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to define the conditional branching in the pipeline."""
        return dict(
            Conditions=[condition.to_request() for condition in self.conditions],
            IfSteps=list_to_request(self.if_steps),
            ElseSteps=list_to_request(self.else_steps),
        )

    @property
    def step_only_arguments(self):
        """Argument dict pertaining to the step only, and not the `if_steps` or `else_steps`."""
        return dict(Conditions=[condition.to_request() for condition in self.conditions])

    @property
    def properties(self):
        """A simple Properties object with `Outcome` as the only property"""
        return self._properties


class JsonGet(NewJsonGet):  # pragma: no cover
    """Get JSON properties from PropertyFiles.

    Attributes:
        step (Step): The step from which to get the property file.
        property_file (Union[PropertyFile, str]): Either a PropertyFile instance
            or the name of a property file.
        json_path (str): The JSON path expression to the requested value.
    """

    def __init__(self, step: Step, property_file: Union[PropertyFile, str], json_path: str):
        super().__init__(step_name=step.name, property_file=property_file, json_path=json_path)


JsonGet = deprecated_class(JsonGet, "JsonGet")
