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
"""Module that contains validators and a validation chain"""
from __future__ import absolute_import

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, List

import attr

from sagemaker.feature_store.feature_processor._data_source import (
    FeatureGroupDataSource,
)
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)
from sagemaker.feature_store.feature_processor._input_offset_parser import (
    InputOffsetParser,
)


@attr.s
class Validator(ABC):
    """Base class for all validators. Errors are raised if validation fails."""

    @abstractmethod
    def validate(self, udf: Callable[..., Any], fp_config: FeatureProcessorConfig) -> None:
        """Validates FeatureProcessorConfig and a UDF."""


@attr.s
class ValidatorChain:
    """Executes a series of validators."""

    validators: List[Validator] = attr.ib()

    def validate(self, udf: Callable[..., Any], fp_config: FeatureProcessorConfig) -> None:
        """Validates a value using the list of validators.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Raises:
            ValueError: If there are any validation errors raised by the validators in this chain.
        """
        for validator in self.validators:
            validator.validate(udf, fp_config)


class FeatureProcessorArgValidator(Validator):
    """A validator for arguments provided to FeatureProcessor."""

    def validate(self, udf: Callable[..., Any], fp_config: FeatureProcessorConfig) -> None:
        """Temporary validator for unsupported feature_processor parameters.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.
        """
        # TODO: Validate target_stores values.


class InputValidator(Validator):
    """A validator for the 'input' parameter."""

    def validate(self, udf: Callable[..., Any], fp_config: FeatureProcessorConfig) -> None:
        """Validate the arguments provided to the decorator's input parameter.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Raises:
            ValueError: If no inputs are provided.
        """

        inputs = fp_config.inputs
        if inputs is None or len(inputs) == 0:
            raise ValueError("At least one input is required.")


class SparkUDFSignatureValidator(Validator):
    """A validator for PySpark UDF signatures."""

    def validate(self, udf: Callable[..., Any], fp_config: FeatureProcessorConfig) -> None:
        """Validate the signature of the UDF based on the configurations provided to the decorator.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Raises (ValueError): raises ValueError when any of the following scenario happen:
           1. No input provided to feature_processor.
           2. Number of provided parameters does not match with that of provided inputs.
           3. Required parameters are not provided in the right order.
        """
        parameters = list(inspect.signature(udf).parameters.keys())
        input_parameters = self._get_input_params(udf)
        if len(input_parameters) < 1:
            raise ValueError("feature_processor expects at least 1 input parameter.")

        # Validate count of input parameters against requested inputs.
        num_data_sources = len(fp_config.inputs)
        if len(input_parameters) != num_data_sources:
            raise ValueError(
                f"feature_processor expected a function with ({num_data_sources}) parameter(s)"
                f" before any optional 'params' or 'spark' parameters for the ({num_data_sources})"
                f" requested data source(s)."
            )

        # Validate position of non-input parameters.
        if "params" in parameters and parameters[-1] != "params" and parameters[-2] != "params":
            raise ValueError(
                "feature_processor expected the 'params' parameter to be the last or second last"
                " parameter after input parameters."
            )

        if "spark" in parameters and parameters[-1] != "spark" and parameters[-2] != "spark":
            raise ValueError(
                "feature_processor expected the 'spark' parameter to be the last or second last"
                " parameter after input parameters."
            )

    def _get_input_params(self, udf: Callable[..., Any]) -> List[str]:
        """Get the parameters that correspond to the inputs for a UDF.

        Args:
            udf (Callable[..., Any]): the user provided function.
        """
        parameters = list(inspect.signature(udf).parameters.keys())

        # Remove non-input parameter names.
        if "params" in parameters:
            parameters.remove("params")
        if "spark" in parameters:
            parameters.remove("spark")

        return parameters


class InputOffsetValidator(Validator):
    """An Validator for input offset."""

    def validate(self, udf: Callable[..., Any], fp_config: FeatureProcessorConfig) -> None:
        """Validate the start and end input offset provided to the decorator.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Raises (ValueError): raises ValueError when input_start_offset is later than
            input_end_offset.
        """

        for config_input in fp_config.inputs:
            if isinstance(input, FeatureGroupDataSource):
                input_start_offset = config_input.input_start_offset
                input_end_offset = config_input.input_end_offset
                start_td = InputOffsetParser.parse_offset_to_timedelta(input_start_offset)
                end_td = InputOffsetParser.parse_offset_to_timedelta(input_end_offset)
                if start_td and end_td and start_td > end_td:
                    raise ValueError("input_start_offset should be always before input_end_offset.")
