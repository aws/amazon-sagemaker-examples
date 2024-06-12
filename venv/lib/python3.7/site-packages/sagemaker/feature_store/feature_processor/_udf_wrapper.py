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
"""This module provides a wrapper for user provided functions."""
from __future__ import absolute_import

import functools
from typing import Any, Callable, Dict, Generic, Tuple, TypeVar

import attr

from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)
from sagemaker.feature_store.feature_processor._udf_arg_provider import UDFArgProvider
from sagemaker.feature_store.feature_processor._udf_output_receiver import (
    UDFOutputReceiver,
)

T = TypeVar("T")


@attr.s
class UDFWrapper(Generic[T]):
    """Class that wraps a user provided function."""

    udf_arg_provider: UDFArgProvider[T] = attr.ib()
    udf_output_receiver: UDFOutputReceiver[T] = attr.ib()

    def wrap(self, udf: Callable[..., T], fp_config: FeatureProcessorConfig) -> Callable[..., None]:
        """Wrap the provided UDF with the logic defined by the FeatureProcessorConfig.

        General functionality of the wrapper function includes but is not limited to loading data
        sources and ingesting output data to a Feature Group.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Returns:
            Callable[..., None]: the user provided function wrapped with feature_processor logic.
        """

        @functools.wraps(udf)
        def wrapper() -> None:
            udf_args, udf_kwargs = self._prepare_udf_args(
                udf=udf,
                fp_config=fp_config,
            )

            output = udf(*udf_args, **udf_kwargs)

            self.udf_output_receiver.ingest_udf_output(output, fp_config)

        return wrapper

    def _prepare_udf_args(
        self,
        udf: Callable[..., T],
        fp_config: FeatureProcessorConfig,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Generate the arguments for the user defined function, provided by the wrapper function.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Returns:
            Tuple[Tuple[Any, ...], Dict[str, Any]]: A tuple positional arguments and keyword
                arguments for the UDF.
        """
        args = ()
        kwargs = {
            **self.udf_arg_provider.provide_input_args(udf, fp_config),
            **self.udf_arg_provider.provide_params_arg(udf, fp_config),
            **self.udf_arg_provider.provide_additional_kwargs(udf),
        }

        return (args, kwargs)
