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

from typing import List, Dict, Optional, Union
from enum import Enum
import warnings

import attr

from sagemaker.workflow.entities import (
    RequestType,
)
from sagemaker.workflow.properties import (
    Properties,
)
from sagemaker.workflow.entities import (
    DefaultEnumMeta,
)
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.steps import Step, StepTypeEnum, CacheConfig
from sagemaker.lambda_helper import Lambda


class LambdaOutputTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """LambdaOutput type enum."""

    String = "String"
    Integer = "Integer"
    Boolean = "Boolean"
    Float = "Float"


@attr.s
class LambdaOutput:
    """Output for a lambdaback step.

    Attributes:
        output_name (str): The output name
        output_type (LambdaOutputTypeEnum): The output type
    """

    output_name: str = attr.ib(default=None)
    output_type: LambdaOutputTypeEnum = attr.ib(default=LambdaOutputTypeEnum.String)

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        return {
            "OutputName": self.output_name,
            "OutputType": self.output_type.value,
        }

    def expr(self, step_name) -> Dict[str, str]:
        """The 'Get' expression dict for a `LambdaOutput`."""
        return LambdaOutput._expr(self.output_name, step_name)

    @classmethod
    def _expr(cls, name, step_name):
        """An internal classmethod for the 'Get' expression dict for a `LambdaOutput`.

        Args:
            name (str): The name of the lambda output.
            step_name (str): The name of the step the lambda step associated
                with this output belongs to.
        """
        return {"Get": f"Steps.{step_name}.OutputParameters['{name}']"}


class LambdaStep(Step):
    """Lambda step for workflow."""

    def __init__(
        self,
        name: str,
        lambda_func: Lambda,
        display_name: str = None,
        description: str = None,
        inputs: dict = None,
        outputs: List[LambdaOutput] = None,
        cache_config: CacheConfig = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Constructs a LambdaStep.

        Args:
            name (str): The name of the lambda step.
            display_name (str): The display name of the Lambda step.
            description (str): The description of the Lambda step.
            lambda_func (str): An instance of sagemaker.lambda_helper.Lambda.
                If lambda arn is specified in the instance, LambdaStep just invokes the function,
                else lambda function will be created while creating the pipeline.
            inputs (dict): Input arguments that will be provided
                to the lambda function.
            outputs (List[LambdaOutput]): List of outputs from the lambda function.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `LambdaStep`
                depends on.
        """
        super(LambdaStep, self).__init__(
            name, display_name, description, StepTypeEnum.LAMBDA, depends_on
        )
        self.lambda_func = lambda_func
        self.outputs = outputs if outputs is not None else []
        self.cache_config = cache_config
        self.inputs = inputs if inputs is not None else {}

        root_prop = Properties(step_name=name)

        property_dict = {}
        for output in self.outputs:
            property_dict[output.output_name] = Properties(
                step_name=name, path=f"OutputParameters['{output.output_name}']"
            )

        root_prop.__dict__["Outputs"] = property_dict
        self._properties = root_prop

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to define the lambda step."""
        return self.inputs

    @property
    def properties(self):
        """A Properties object representing the output parameters of the lambda step."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        function_arn = self._get_function_arn()
        request_dict["FunctionArn"] = function_arn

        request_dict["OutputParameters"] = list(map(lambda op: op.to_request(), self.outputs))

        return request_dict

    def _get_function_arn(self):
        """Returns the lambda function arn

        It upserts a lambda function if function name is provided.
        It updates a lambda function if lambda arn and code is provided.
        It is a no-op if code is not provided but function arn is provided.
        """
        if self.lambda_func.function_arn is None:
            response = self.lambda_func.upsert()
            return response["FunctionArn"]

        if self.lambda_func.zipped_code_dir is None and self.lambda_func.script is None:
            warnings.warn(
                "Lambda function won't be updated because zipped_code_dir \
                or script is not provided."
            )
            return self.lambda_func.function_arn

        response = self.lambda_func.update()
        return response["FunctionArn"]
