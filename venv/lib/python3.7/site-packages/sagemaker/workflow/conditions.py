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
"""Conditions for condition steps.

Ideally, some of these comparison conditions would be implemented as "partial classes",
but use of functools.partial doesn't set correct metadata/type information.
"""
from __future__ import absolute_import

import abc

from enum import Enum
from typing import Dict, List, Union

import attr

from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import (
    DefaultEnumMeta,
    Entity,
    Expression,
    PrimitiveType,
    RequestType,
)
from sagemaker.workflow.execution_variables import ExecutionVariable
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.entities import PipelineVariable

# TODO: consider base class for those with an expr method, rather than defining a type here
ConditionValueType = Union[ExecutionVariable, Parameter, Properties]


class ConditionTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """Condition type enum."""

    EQ = "Equals"
    GT = "GreaterThan"
    GTE = "GreaterThanOrEqualTo"
    IN = "In"
    LT = "LessThan"
    LTE = "LessThanOrEqualTo"
    NOT = "Not"
    OR = "Or"


@attr.s
class Condition(Entity):
    """Abstract Condition entity.

    Attributes:
        condition_type (ConditionTypeEnum): The type of condition.
    """

    condition_type: ConditionTypeEnum = attr.ib(factory=ConditionTypeEnum.factory)

    @property
    @abc.abstractmethod
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""


@attr.s
class ConditionComparison(Condition):
    """Generic comparison condition that can be used to derive specific condition comparisons.

    Attributes:
        left (Union[ConditionValueType, PrimitiveType]): The execution variable, parameter,
            property, or Python primitive value to use in the comparison.
        right (Union[ConditionValueType, PrimitiveType]): The execution variable,
            parameter, property, or Python primitive value to compare to.
    """

    left: Union[ConditionValueType, PrimitiveType] = attr.ib(default=None)
    right: Union[ConditionValueType, PrimitiveType] = attr.ib(default=None)

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        return {
            "Type": self.condition_type.value,
            "LeftValue": self.left,
            "RightValue": self.right,
        }

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        steps = []
        if isinstance(self.left, PipelineVariable):
            steps.extend(self.left._referenced_steps)
        if isinstance(self.right, PipelineVariable):
            steps.extend(self.right._referenced_steps)
        return steps


class ConditionEquals(ConditionComparison):
    """A condition for equality comparisons."""

    def __init__(
        self,
        left: Union[ConditionValueType, PrimitiveType],
        right: Union[ConditionValueType, PrimitiveType],
    ):
        """Construct A condition for equality comparisons.

        Args:
            left (Union[ConditionValueType, PrimitiveType]): The execution variable,
                parameter, property, or Python primitive value to use in the comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or Python primitive value to compare to.
        """

        super(ConditionEquals, self).__init__(ConditionTypeEnum.EQ, left, right)


class ConditionGreaterThan(ConditionComparison):
    """A condition for greater than comparisons."""

    def __init__(
        self,
        left: Union[ConditionValueType, PrimitiveType],
        right: Union[ConditionValueType, PrimitiveType],
    ):
        """Construct an instance of ConditionGreaterThan for greater than comparisons.

        Args:
            left (Union[ConditionValueType, PrimitiveType]): The execution variable,
                parameter, property, or Python primitive value to use in the comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or Python primitive value to compare to.
        """

        super(ConditionGreaterThan, self).__init__(ConditionTypeEnum.GT, left, right)


class ConditionGreaterThanOrEqualTo(ConditionComparison):
    """A condition for greater than or equal to comparisons."""

    def __init__(
        self,
        left: Union[ConditionValueType, PrimitiveType],
        right: Union[ConditionValueType, PrimitiveType],
    ):
        """Construct of ConditionGreaterThanOrEqualTo for greater than or equal to comparisons.

        Args:
            left (Union[ConditionValueType, PrimitiveType]): The execution variable,
                parameter, property, or Python primitive value to use in the comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or Python primitive value to compare to.
        """

        super(ConditionGreaterThanOrEqualTo, self).__init__(ConditionTypeEnum.GTE, left, right)


class ConditionLessThan(ConditionComparison):
    """A condition for less than comparisons."""

    def __init__(
        self,
        left: Union[ConditionValueType, PrimitiveType],
        right: Union[ConditionValueType, PrimitiveType],
    ):
        """Construct an instance of ConditionLessThan for less than comparisons.

        Args:
            left (Union[ConditionValueType, PrimitiveType]): The execution variable,
                parameter, property, or Python primitive value to use in the comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or Python primitive value to compare to.
        """

        super(ConditionLessThan, self).__init__(ConditionTypeEnum.LT, left, right)


class ConditionLessThanOrEqualTo(ConditionComparison):
    """A condition for less than or equal to comparisons."""

    def __init__(
        self,
        left: Union[ConditionValueType, PrimitiveType],
        right: Union[ConditionValueType, PrimitiveType],
    ):
        """Construct ConditionLessThanOrEqualTo for less than or equal to comparisons.

        Args:
            left (Union[ConditionValueType, PrimitiveType]): The execution variable,
                parameter, property, or Python primitive value to use in the comparison.
            right (Union[ConditionValueType, PrimitiveType]): The execution
                variable, parameter, property, or Python primitive value to compare to.
        """

        super(ConditionLessThanOrEqualTo, self).__init__(ConditionTypeEnum.LTE, left, right)


class ConditionIn(Condition):
    """A condition to check membership."""

    def __init__(
        self,
        value: Union[ConditionValueType, PrimitiveType],
        in_values: List[Union[ConditionValueType, PrimitiveType]],
    ):
        """Construct a `ConditionIn` condition to check membership.

        Args:
            value (Union[ConditionValueType, PrimitiveType]): The execution variable,
                parameter, property or primitive value to check for membership.
            in_values (List[Union[ConditionValueType, PrimitiveType]]): The list
                of values to check for membership in.
        """
        super(ConditionIn, self).__init__(ConditionTypeEnum.IN)
        self.value = value
        self.in_values = in_values

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        return {
            "Type": self.condition_type.value,
            "QueryValue": self.value,
            "Values": self.in_values,
        }

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        steps = []
        if isinstance(self.value, PipelineVariable):
            steps.extend(self.value._referenced_steps)
        for in_value in self.in_values:
            if isinstance(in_value, PipelineVariable):
                steps.extend(in_value._referenced_steps)
        return steps


class ConditionNot(Condition):
    """A condition for negating another `Condition`."""

    def __init__(self, expression: Condition):
        """Construct a `ConditionNot` condition for negating another `Condition`.

        Attributes:
            expression (Condition): A `Condition` to take the negation of.
        """
        super(ConditionNot, self).__init__(ConditionTypeEnum.NOT)
        self.expression = expression

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        return {"Type": self.condition_type.value, "Expression": self.expression.to_request()}

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        return self.expression._referenced_steps


class ConditionOr(Condition):
    """A condition for taking the logical OR of a list of `Condition` instances."""

    def __init__(self, conditions: List[Condition] = None):
        """Construct a `ConditionOr` condition.

        Attributes:
            conditions (List[Condition]): A list of `Condition` instances to logically OR.
        """
        super(ConditionOr, self).__init__(ConditionTypeEnum.OR)
        self.conditions = conditions or []

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        return {
            "Type": self.condition_type.value,
            "Conditions": [condition.to_request() for condition in self.conditions],
        }

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        steps = []
        for condition in self.conditions:
            steps.extend(condition._referenced_steps)
        return steps


def primitive_or_expr(
    value: Union[ExecutionVariable, Expression, PrimitiveType, Parameter, Properties]
) -> Union[Dict[str, str], PrimitiveType]:
    """Provide the expression of the value or return value if it is a primitive.

    Args:
        value (Union[ConditionValueType, PrimitiveType]): The value to evaluate.
    Returns:
        Either the expression of the value or the primitive value.
    """
    if is_pipeline_variable(value):
        return value.expr
    return value
