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
"""Placeholder docstring"""
from __future__ import absolute_import

import json
from typing import Union

from sagemaker.workflow.entities import PipelineVariable
from sagemaker.utils import to_string


class ParameterRange(object):
    """Base class for representing parameter ranges.

    This is used to define what hyperparameters to tune for an Amazon SageMaker
    hyperparameter tuning job and to verify hyperparameters for Marketplace Algorithms.
    """

    __all_types__ = ("Continuous", "Categorical", "Integer")

    def __init__(
        self,
        min_value: Union[int, float, PipelineVariable],
        max_value: Union[int, float, PipelineVariable],
        scaling_type: Union[str, PipelineVariable] = "Auto",
    ):
        """Initialize a parameter range.

        Args:
            min_value (float or int or PipelineVariable): The minimum value for the range.
            max_value (float or int or PipelineVariable): The maximum value for the range.
            scaling_type (str or PipelineVariable): The scale used for searching the range during
                tuning (default: 'Auto'). Valid values: 'Auto', 'Linear',
                'Logarithmic' and 'ReverseLogarithmic'.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.scaling_type = scaling_type

    def is_valid(self, value):
        """Determine if a value is valid within this ParameterRange.

        Args:
            value (float or int): The value to be verified.

        Returns:
            bool: True if valid, False otherwise.
        """
        return self.min_value <= value <= self.max_value

    @classmethod
    def cast_to_type(cls, value):
        """Placeholder docstring"""
        return float(value)

    def as_tuning_range(self, name):
        """Represent the parameter range as a dictionary.

        It is suitable for a request to create an Amazon SageMaker hyperparameter tuning job.

        Args:
            name (str): The name of the hyperparameter.

        Returns:
            dict[str, str]: A dictionary that contains the name and values of
            the hyperparameter.
        """
        return {
            "Name": name,
            "MinValue": to_string(self.min_value),
            "MaxValue": to_string(self.max_value),
            "ScalingType": self.scaling_type,
        }


class ContinuousParameter(ParameterRange):
    """A class for representing hyperparameters that have a continuous range of possible values.

    Args:
            min_value (float): The minimum value for the range.
            max_value (float): The maximum value for the range.
    """

    __name__ = "Continuous"

    @classmethod
    def cast_to_type(cls, value):
        """Placeholder docstring"""
        return float(value)


class CategoricalParameter(ParameterRange):
    """A class for representing hyperparameters that have a discrete list of possible values."""

    __name__ = "Categorical"

    def __init__(self, values):  # pylint: disable=super-init-not-called
        """Initialize a ``CategoricalParameter``.

        Args:
            values (list or object): The possible values for the hyperparameter.
                This input will be converted into a list of strings.
        """
        values = values if isinstance(values, list) else [values]
        self.values = [to_string(v) for v in values]

    def as_tuning_range(self, name):
        """Represent the parameter range as a dictionary.

        It is suitable for a request to create an Amazon SageMaker hyperparameter tuning job.

        Args:
            name (str): The name of the hyperparameter.

        Returns:
            dict[str, list[str]]: A dictionary that contains the name and values
            of the hyperparameter.
        """
        return {"Name": name, "Values": self.values}

    def as_json_range(self, name):
        """Represent the parameter range as a dictionary.

        Dictionary is suitable for a request to create an Amazon SageMaker hyperparameter tuning job
        using one of the deep learning frameworks.

        The deep learning framework images require that hyperparameters be
        serialized as JSON.

        Args:
            name (str): The name of the hyperparameter.

        Returns:
            dict[str, list[str]]: A dictionary that contains the name and values of the
            hyperparameter, where the values are serialized as JSON.
        """
        return {"Name": name, "Values": [json.dumps(v) for v in self.values]}

    def is_valid(self, value):
        """Placeholder docstring"""
        return value in self.values

    @classmethod
    def cast_to_type(cls, value):
        """Placeholder docstring"""
        return str(value)


class IntegerParameter(ParameterRange):
    """A class for representing hyperparameters that have an integer range of possible values.

    Args:
        min_value (int): The minimum value for the range.
        max_value (int): The maximum value for the range.
    """

    __name__ = "Integer"

    @classmethod
    def cast_to_type(cls, value):
        """Placeholder docstring"""
        return int(value)
