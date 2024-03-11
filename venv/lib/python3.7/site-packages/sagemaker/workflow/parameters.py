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
"""Pipeline parameters and conditions for workflow."""
from __future__ import absolute_import

from enum import Enum
from functools import partial
from typing import Dict, List, Type

import attr

from sagemaker.workflow.entities import (
    DefaultEnumMeta,
    Entity,
    PrimitiveType,
    RequestType,
    PipelineVariable,
)


class ParameterTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """Parameter type enum."""

    STRING = "String"
    INTEGER = "Integer"
    BOOLEAN = "Boolean"
    FLOAT = "Float"

    @property
    def python_type(self) -> Type:
        """Provide the Python type of the enum value."""
        mapping = {
            ParameterTypeEnum.STRING: str,
            ParameterTypeEnum.INTEGER: int,
            ParameterTypeEnum.BOOLEAN: bool,
            ParameterTypeEnum.FLOAT: float,
        }
        return mapping[self]


@attr.s
class Parameter(PipelineVariable, Entity):
    """Pipeline parameter for workflow.

    Attributes:
        name (str): The name of the parameter.
        parameter_type (ParameterTypeEnum): The type of the parameter.
        default_value (PrimitiveType): The default value of the parameter.
    """

    name: str = attr.ib(factory=str)
    parameter_type: ParameterTypeEnum = attr.ib(factory=ParameterTypeEnum.factory)
    default_value: PrimitiveType = attr.ib(default=None)

    @default_value.validator
    def _check_default_value(self, _, value):
        """Check whether the default value is compatible with the parameter type.

        Args:
            _: unused argument required by attrs validator decorator.
            value: The value to check the type for.

        Raises:
            `TypeError` if the value is not compatible with the instance's Python type.
        """
        self._check_default_value_type(value, self.parameter_type.python_type)

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        value = {
            "Name": self.name,
            "Type": self.parameter_type.value,
        }
        if self.default_value is not None:
            value["DefaultValue"] = self.default_value
        return value

    @property
    def expr(self) -> Dict[str, str]:
        """The 'Get' expression dict for a `Parameter`."""
        return Parameter._expr(self.name)

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        return []

    @classmethod
    def _expr(cls, name):
        """An internal classmethod for the 'Get' expression dict for a `Parameter`.

        Args:
            name (str): The name of the parameter.
        """
        return {"Get": f"Parameters.{name}"}

    @classmethod
    def _check_default_value_type(cls, value, python_type):
        """Check whether the default value is compatible with the parameter type.

        Args:
            value: The value to check the type for.
            python_type: The type to check the value against.

        Raises:
            `TypeError` if the value is not compatible with the instance's Python type.
        """
        if value and not isinstance(value, python_type):
            raise TypeError("The default value specified does not match the Parameter Python type.")


# NOTE: partials do not handle metadata well, but make for at least "partial" syntactic sugar :-P
# proper implementation postponed, for "reasons": https://bugs.python.org/issue33419
# NOTE: cannot subclass bool: http://mail.python.org/pipermail/python-dev/2002-March/020822.html
ParameterBoolean = partial(Parameter, parameter_type=ParameterTypeEnum.BOOLEAN)


class ParameterString(Parameter):
    """String parameter for pipelines."""

    def __init__(self, name: str, default_value: str = None, enum_values: List[str] = None):
        """Create a pipeline string parameter.

        Args:
            name (str): The name of the parameter.
            default_value (str): The default value of the parameter.
                The default value could be overridden at start of an execution.
                If not set or it is set to None, a value must be provided
                at the start of the execution.
            enum_values (List[str]): Enum values for this parameter.
        """
        super(ParameterString, self).__init__(
            name=name, parameter_type=ParameterTypeEnum.STRING, default_value=default_value
        )
        self.enum_values = enum_values

    def __hash__(self):
        """Hash function for parameter types"""
        return hash(tuple(self.to_request()))

    def to_string(self) -> PipelineVariable:
        """Prompt the pipeline to convert the pipeline variable to String in runtime

        As ParameterString is treated as String in runtime, no extra actions are needed.
        """
        return self

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        request_dict = super(ParameterString, self).to_request()
        if self.enum_values:
            request_dict["EnumValues"] = self.enum_values
        return request_dict


class ParameterInteger(Parameter):
    """Integer parameter for pipelines."""

    def __init__(self, name: str, default_value: int = None):
        """Create a pipeline integer parameter.

        Args:
            name (str): The name of the parameter.
            default_value (int): The default value of the parameter.
                The default value could be overridden at start of an execution.
                If not set or it is set to None, a value must be provided
                at the start of the execution.
        """
        super(ParameterInteger, self).__init__(
            name=name, parameter_type=ParameterTypeEnum.INTEGER, default_value=default_value
        )


class ParameterFloat(Parameter):
    """Float parameter for pipelines."""

    def __init__(self, name: str, default_value: float = None):
        """Create a pipeline float parameter.

        Args:
            name (str): The name of the parameter.
            default_value (float): The default value of the parameter.
                The default value could be overridden at start of an execution.
                If not set or it is set to None, a value must be provided
                at the start of the execution.
        """
        super(ParameterFloat, self).__init__(
            name=name, parameter_type=ParameterTypeEnum.FLOAT, default_value=default_value
        )
