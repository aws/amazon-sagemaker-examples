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
"""This module stores filters related to SageMaker JumpStart."""
from __future__ import absolute_import
from ast import literal_eval
from enum import Enum
from typing import Dict, List, Union, Any

from sagemaker.jumpstart.types import JumpStartDataHolderType


class BooleanValues(str, Enum):
    """Enum class for boolean values.

    This is a status value that an ``Operand`` can resolve to.
    """

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    UNEVALUATED = "unevaluated"


class FilterOperators(str, Enum):
    """Enum class for filter operators for JumpStart models."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"


class SpecialSupportedFilterKeys(str, Enum):
    """Enum class for special supported filter keys."""

    TASK = "task"
    FRAMEWORK = "framework"
    SUPPORTED_MODEL = "supported_model"


FILTER_OPERATOR_STRING_MAPPINGS = {
    FilterOperators.EQUALS: ["===", "==", "equals", "is"],
    FilterOperators.NOT_EQUALS: ["!==", "!=", "not equals", "is not"],
    FilterOperators.IN: ["in"],
    FilterOperators.NOT_IN: ["not in"],
}


_PAD_ALPHABETIC_OPERATOR = (
    lambda operator: f" {operator} "
    if any(character.isalpha() for character in operator)
    else operator
)

ACCEPTABLE_OPERATORS_IN_PARSE_ORDER = (
    list(map(_PAD_ALPHABETIC_OPERATOR, FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_EQUALS]))
    + list(map(_PAD_ALPHABETIC_OPERATOR, FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_IN]))
    + list(map(_PAD_ALPHABETIC_OPERATOR, FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.EQUALS]))
    + list(map(_PAD_ALPHABETIC_OPERATOR, FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.IN]))
)


SPECIAL_SUPPORTED_FILTER_KEYS = set(
    [
        SpecialSupportedFilterKeys.TASK,
        SpecialSupportedFilterKeys.FRAMEWORK,
        SpecialSupportedFilterKeys.SUPPORTED_MODEL,
    ]
)


class Operand:
    """Operand class for filtering JumpStart content."""

    def __init__(
        self, unresolved_value: Any, resolved_value: BooleanValues = BooleanValues.UNEVALUATED
    ):
        self.unresolved_value: Any = unresolved_value
        self._resolved_value: BooleanValues = resolved_value

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield self

    def eval(self) -> None:
        """Evaluates operand."""
        return

    @property
    def resolved_value(self) -> BooleanValues:
        """Getter method for resolved_value."""
        return self._resolved_value

    @resolved_value.setter
    def resolved_value(self, new_resolved_value: Any) -> None:
        """Setter method for resolved_value. Resolved_value must be of type ``BooleanValues``."""
        if isinstance(new_resolved_value, BooleanValues):
            self._resolved_value = new_resolved_value
            return
        raise RuntimeError(
            "Resolved value must be of type BooleanValues, "
            f"but got type {type(new_resolved_value)}."
        )

    @staticmethod
    def validate_operand(operand: Any) -> Any:
        """Validate operand and return ``Operand`` object.

        Args:
            operand (Any): The operand to validate.

        Raises:
            RuntimeError: If the operand is not of ``Operand`` or ``str`` type.
        """
        if isinstance(operand, str):
            if operand.lower() == BooleanValues.TRUE.lower():
                operand = Operand(operand, resolved_value=BooleanValues.TRUE)
            elif operand.lower() == BooleanValues.FALSE.lower():
                operand = Operand(operand, resolved_value=BooleanValues.FALSE)
            elif operand.lower() == BooleanValues.UNKNOWN.lower():
                operand = Operand(operand, resolved_value=BooleanValues.UNKNOWN)
            else:
                operand = Operand(parse_filter_string(operand))
        elif not issubclass(type(operand), Operand):
            raise RuntimeError(f"Operand '{operand}' is not supported.")
        return operand


class Operator(Operand):
    """Operator class for filtering JumpStart content.

    An operator in this case corresponds to an operand that is also an operation.
    For example, given the expression ``(True or True) and True``,
    ``(True or True)`` is an operand to an ``And`` expression, but is also itself an
    operator. ``(True or True) and True`` would also be considered an operator.
    """

    def __init__(
        self,
        resolved_value: BooleanValues = BooleanValues.UNEVALUATED,
        unresolved_value: Any = None,
    ):
        """Initializes ``Operator`` instance.

        Args:
            resolved_value (BooleanValues): Optional. The resolved value of the operator.
                (Default: BooleanValues.UNEVALUATED).
            unresolved_value (Any): Optional. The unresolved value of the operator.
                (Default: None).
        """
        super().__init__(unresolved_value=unresolved_value, resolved_value=resolved_value)

    def eval(self) -> None:
        """Evaluates operator."""
        return

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield self


class And(Operator):
    """And operator class for filtering JumpStart content."""

    def __init__(
        self,
        *operands: Union[Operand, str],
    ) -> None:
        """Instantiates And object.

        Args:
            operand (Operand): Operand for And-ing.

        Raises:
            RuntimeError: If the operands cannot be validated.
        """
        self.operands: List[Operand] = list(operands)  # type: ignore
        for i in range(len(self.operands)):
            self.operands[i] = Operand.validate_operand(self.operands[i])
        super().__init__()

    def eval(self) -> None:
        """Evaluates operator.

        Raises:
            RuntimeError: If the operands remain unevaluated after calling ``eval``,
                or if the resolved value isn't a ``BooleanValues`` type.
        """
        incomplete_expression = False
        for operand in self.operands:
            if not issubclass(type(operand), Operand):
                raise RuntimeError(
                    f"Operand must be subclass of ``Operand``, but got {type(operand)}"
                )
            if operand.resolved_value == BooleanValues.UNEVALUATED:
                operand.eval()
                if operand.resolved_value == BooleanValues.UNEVALUATED:
                    raise RuntimeError(
                        "Operand remains unevaluated after calling ``eval`` function."
                    )
            if operand.resolved_value == BooleanValues.FALSE:
                self.resolved_value = BooleanValues.FALSE
                return
            if operand.resolved_value == BooleanValues.UNKNOWN:
                incomplete_expression = True
        if not incomplete_expression:
            self.resolved_value = BooleanValues.TRUE
        else:
            self.resolved_value = BooleanValues.UNKNOWN

    def __iter__(self) -> Any:
        """Returns an iterator."""
        for operand in self.operands:
            yield from operand
        yield self


class Constant(Operator):
    """Constant operator class for filtering JumpStart content."""

    def __init__(
        self,
        constant: BooleanValues,
    ):
        """Instantiates Constant operator object.

        Args:
            constant (BooleanValues): Value of constant.
        """
        super().__init__(constant)

    def eval(self) -> None:
        """Evaluates constant"""
        return

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield self


class Identity(Operator):
    """Identity operator class for filtering JumpStart content."""

    def __init__(
        self,
        operand: Union[Operand, str],
    ):
        """Instantiates Identity object.

        Args:
            operand (Union[Operand, str]): Operand for identity operation.
        """
        super().__init__()
        self.operand = Operand.validate_operand(operand)

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield self
        yield from self.operand

    def eval(self) -> None:
        """Evaluates operator.

        Raises:
            RuntimeError: If the operand remains unevaluated after calling ``eval``,
                or if the resolved value isn't a ``BooleanValues`` type.
        """
        if not issubclass(type(self.operand), Operand):
            raise RuntimeError(
                f"Operand must be subclass of ``Operand``, but got {type(self.operand)}"
            )
        if self.operand.resolved_value == BooleanValues.UNEVALUATED:
            self.operand.eval()
            if self.operand.resolved_value == BooleanValues.UNEVALUATED:
                raise RuntimeError("Operand remains unevaluated after calling ``eval`` function.")
        if not isinstance(self.operand.resolved_value, BooleanValues):
            raise RuntimeError(self.operand.resolved_value)
        self.resolved_value = self.operand.resolved_value


class Or(Operator):
    """Or operator class for filtering JumpStart content."""

    def __init__(
        self,
        *operands: Union[Operand, str],
    ) -> None:
        """Instantiates Or object.

        Args:
            operands (Operand): Operand for Or-ing.

        Raises:
            RuntimeError: If the operands cannot be validated.
        """
        self.operands: List[Operand] = list(operands)  # type: ignore
        for i in range(len(self.operands)):
            self.operands[i] = Operand.validate_operand(self.operands[i])
        super().__init__()

    def eval(self) -> None:
        """Evaluates operator.

        Raises:
            RuntimeError: If the operands remain unevaluated after calling ``eval``,
                or if the resolved value isn't a ``BooleanValues`` type.
        """
        incomplete_expression = False
        for operand in self.operands:
            if not issubclass(type(operand), Operand):
                raise RuntimeError(
                    f"Operand must be subclass of ``Operand``, but got {type(operand)}"
                )
            if operand.resolved_value == BooleanValues.UNEVALUATED:
                operand.eval()
                if operand.resolved_value == BooleanValues.UNEVALUATED:
                    raise RuntimeError(
                        "Operand remains unevaluated after calling ``eval`` function."
                    )
            if operand.resolved_value == BooleanValues.TRUE:
                self.resolved_value = BooleanValues.TRUE
                return
            if operand.resolved_value == BooleanValues.UNKNOWN:
                incomplete_expression = True
        if not incomplete_expression:
            self.resolved_value = BooleanValues.FALSE
        else:
            self.resolved_value = BooleanValues.UNKNOWN

    def __iter__(self) -> Any:
        """Returns an iterator."""
        for operand in self.operands:
            yield from operand
        yield self


class Not(Operator):
    """Not operator class for filtering JumpStart content."""

    def __init__(
        self,
        operand: Union[Operand, str],
    ) -> None:
        """Instantiates Not object.

        Args:
            operand (Operand): Operand for Not-ing.
        """
        self.operand: Operand = Operand.validate_operand(operand)
        super().__init__()

    def eval(self) -> None:
        """Evaluates operator.

        Raises:
            RuntimeError: If the operand remains unevaluated after calling ``eval``,
                or if the resolved value isn't a ``BooleanValues`` type.
        """

        if not issubclass(type(self.operand), Operand):
            raise RuntimeError(
                f"Operand must be subclass of ``Operand``, but got {type(self.operand)}"
            )
        if self.operand.resolved_value == BooleanValues.UNEVALUATED:
            self.operand.eval()
            if self.operand.resolved_value == BooleanValues.UNEVALUATED:
                raise RuntimeError("Operand remains unevaluated after calling ``eval`` function.")
        if self.operand.resolved_value == BooleanValues.TRUE:
            self.resolved_value = BooleanValues.FALSE
            return
        if self.operand.resolved_value == BooleanValues.FALSE:
            self.resolved_value = BooleanValues.TRUE
            return
        self.resolved_value = BooleanValues.UNKNOWN

    def __iter__(self) -> Any:
        """Returns an iterator."""
        yield from self.operand
        yield self


class ModelFilter(JumpStartDataHolderType):
    """Data holder class to store model filters.

    For a given filter string "task == ic", the key corresponds to
    "task" and the value corresponds to "ic", with the operation being
    "==".
    """

    __slots__ = ["key", "value", "operator"]

    def __init__(self, key: str, value: str, operator: str):
        """Instantiates ``ModelFilter`` object.

        Args:
            key (str): The key in metadata for the model filter.
            value (str): The value of the metadata for the model filter.
            operator (str): The operator used in the model filter.
        """
        self.key = key
        self.value = value
        self.operator = operator


def parse_filter_string(filter_string: str) -> ModelFilter:
    """Parse filter string and return a serialized ``ModelFilter`` object.

    Args:
        filter_string (str): The filter string to be serialized to an object.
    """

    for operator in ACCEPTABLE_OPERATORS_IN_PARSE_ORDER:
        split_filter_string = filter_string.split(operator)
        if len(split_filter_string) == 2:
            return ModelFilter(
                key=split_filter_string[0].strip(),
                value=split_filter_string[1].strip(),
                operator=operator.strip(),
            )
    raise ValueError(f"Cannot parse filter string: {filter_string}")


def evaluate_filter_expression(  # pylint: disable=too-many-return-statements
    model_filter: ModelFilter,
    cached_model_value: Union[str, bool, int, float, Dict[str, Any], List[Any]],
) -> BooleanValues:
    """Evaluates model filter with cached model spec value, returns boolean.

    Args:
        model_filter (ModelFilter): The model filter for evaluation.
        cached_model_value (Any): The value in the model manifest/spec that should be used to
            evaluate the filter.
    """
    if model_filter.operator in FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.EQUALS]:
        model_filter_value = model_filter.value
        if isinstance(cached_model_value, bool):
            cached_model_value = str(cached_model_value).lower()
            model_filter_value = model_filter.value.lower()
        if str(model_filter_value) == str(cached_model_value):
            return BooleanValues.TRUE
        return BooleanValues.FALSE
    if model_filter.operator in FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_EQUALS]:
        if isinstance(cached_model_value, bool):
            cached_model_value = str(cached_model_value).lower()
            model_filter.value = model_filter.value.lower()
        if str(model_filter.value) == str(cached_model_value):
            return BooleanValues.FALSE
        return BooleanValues.TRUE
    if model_filter.operator in FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.IN]:
        py_obj = literal_eval(model_filter.value)
        try:
            iter(py_obj)
        except TypeError:
            return BooleanValues.FALSE
        if cached_model_value in py_obj:
            return BooleanValues.TRUE
        return BooleanValues.FALSE
    if model_filter.operator in FILTER_OPERATOR_STRING_MAPPINGS[FilterOperators.NOT_IN]:
        py_obj = literal_eval(model_filter.value)
        try:
            iter(py_obj)
        except TypeError:
            return BooleanValues.TRUE
        if cached_model_value in py_obj:
            return BooleanValues.FALSE
        return BooleanValues.TRUE
    raise RuntimeError(f"Bad operator: {model_filter.operator}")
