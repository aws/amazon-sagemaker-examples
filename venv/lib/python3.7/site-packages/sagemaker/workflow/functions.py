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

from typing import List, Union

import attr

from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.properties import PropertyFile


@attr.s
class Join(PipelineVariable):
    """Join together properties.

    Examples:
    Build a Amazon S3 Uri with bucket name parameter and pipeline execution Id and use it
    as training input::

        bucket = ParameterString('bucket', default_value='my-bucket')

        TrainingInput(
            s3_data=Join(
                on='/',
                values=['s3:/', bucket, ExecutionVariables.PIPELINE_EXECUTION_ID]
            ),
            content_type="text/csv")

    Attributes:
        values (List[Union[PrimitiveType, Parameter, Expression]]):
            The primitive type values, parameters, step properties, expressions to join.
        on (str): The string to join the values on (Defaults to "").
    """

    on: str = attr.ib(factory=str)
    values: List = attr.ib(factory=list)

    def to_string(self) -> PipelineVariable:
        """Prompt the pipeline to convert the pipeline variable to String in runtime

        As Join is treated as String in runtime, no extra actions are needed.
        """
        return self

    @property
    def expr(self):
        """The expression dict for a `Join` function."""

        return {
            "Std:Join": {
                "On": self.on,
                "Values": [
                    value.expr if hasattr(value, "expr") else value for value in self.values
                ],
            },
        }

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        steps = []
        for value in self.values:
            if isinstance(value, PipelineVariable):
                steps.extend(value._referenced_steps)
        return steps


@attr.s
class JsonGet(PipelineVariable):
    """Get JSON properties from PropertyFiles.

    Attributes:
        step_name (str): The step name from which to get the property file.
        property_file (Union[PropertyFile, str]): Either a PropertyFile instance
            or the name of a property file.
        json_path (str): The JSON path expression to the requested value.
    """

    step_name: str = attr.ib()
    property_file: Union[PropertyFile, str] = attr.ib()
    json_path: str = attr.ib()

    @property
    def expr(self):
        """The expression dict for a `JsonGet` function."""
        if not isinstance(self.step_name, str) or not self.step_name:
            raise ValueError("Please give a valid step name as a string")

        if isinstance(self.property_file, PropertyFile):
            name = self.property_file.name
        else:
            name = self.property_file
        return {
            "Std:JsonGet": {
                "PropertyFile": {"Get": f"Steps.{self.step_name}.PropertyFiles.{name}"},
                "Path": self.json_path,
            }
        }

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        return [self.step_name]
