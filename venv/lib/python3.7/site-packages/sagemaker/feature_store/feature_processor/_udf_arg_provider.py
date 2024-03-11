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
"""Contains classes for loading arguments for the parameters defined in the UDF."""
from __future__ import absolute_import

from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Callable, Dict, Generic, List, OrderedDict, TypeVar, Union

import attr
from pyspark.sql import DataFrame, SparkSession

from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
)
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)
from sagemaker.feature_store.feature_processor._input_loader import (
    SparkDataFrameInputLoader,
)
from sagemaker.feature_store.feature_processor._params_loader import ParamsLoader
from sagemaker.feature_store.feature_processor._spark_factory import SparkSessionFactory

T = TypeVar("T")


@attr.s
class UDFArgProvider(Generic[T], ABC):
    """Base class for arguments providers for the UDF.

    Args:
        Generic (T): The type of the auto-loaded data values.
    """

    @abstractmethod
    def provide_input_args(
        self, udf: Callable[..., T], fp_config: FeatureProcessorConfig
    ) -> OrderedDict[str, T]:
        """Provides a dict of (input name, auto-loaded data) using the feature_processor parameters.

        The input name is the udfs parameter name, and the data source is the one defined at the
        same index (as the input name) in fp_config.inputs.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Returns:
            OrderedDict[str, T]: The loaded data sources, in the same order as fp_config.inputs.
        """

    @abstractmethod
    def provide_params_arg(
        self, udf: Callable[..., T], fp_config: FeatureProcessorConfig
    ) -> Dict[str, Dict]:
        """Provides the 'params' argument that is provided to the UDF.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Returns:
            Dict[str, Dict]: A combination of user defined parameters (in fp_config) and system
                provided parameters.
        """

    @abstractmethod
    def provide_additional_kwargs(self, udf: Callable[..., T]) -> Dict[str, Any]:
        """Provides any additional arguments to be provided to the UDF, dependent on the mode.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.

        Returns:
            Dict[str, Any]: additional kwargs for the user function.
        """


@attr.s
class SparkArgProvider(UDFArgProvider[DataFrame]):
    """Provides arguments to Spark UDFs."""

    PARAMS_ARG_NAME = "params"
    SPARK_SESSION_ARG_NAME = "spark"

    params_loader: ParamsLoader = attr.ib()
    input_loader: SparkDataFrameInputLoader = attr.ib()
    spark_session_factory: SparkSessionFactory = attr.ib()

    def provide_input_args(
        self, udf: Callable[..., DataFrame], fp_config: FeatureProcessorConfig
    ) -> OrderedDict[str, DataFrame]:
        """Provide a DataFrame for each requested input.

        Args:
            udf (Callable[..., T]): The feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Raises:
            ValueError: If the signature of the UDF does not match the fp_config.inputs.
            ValueError: If there are no inputs provided to the user defined function.

        Returns:
            OrderedDict[str, DataFrame]: The loaded data sources, in the same order as
                fp_config.inputs.
        """
        udf_parameter_names = list(signature(udf).parameters.keys())
        udf_input_names = self._get_input_parameters(udf_parameter_names)

        if len(udf_input_names) == 0:
            raise ValueError("Expected at least one input to the user defined function.")

        if len(udf_input_names) != len(fp_config.inputs):
            raise ValueError(
                f"The signature of the user defined function does not match the list of inputs"
                f" requested. Expected {len(fp_config.inputs)} parameter(s)."
            )

        return OrderedDict(
            (input_name, self._load_data_frame(input_uri))
            for (input_name, input_uri) in zip(udf_input_names, fp_config.inputs)
        )

    def provide_params_arg(
        self, udf: Callable[..., DataFrame], fp_config: FeatureProcessorConfig
    ) -> Dict[str, Union[str, Dict]]:
        """Provide params for the UDF. If the udf has a parameter named 'params'.

        Args:
            udf (Callable[..., T]): the feature_processor wrapped user function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.
        """
        return (
            self.params_loader.get_parameter_args(fp_config)
            if self._has_param(udf, self.PARAMS_ARG_NAME)
            else {}
        )

    def provide_additional_kwargs(self, udf: Callable[..., DataFrame]) -> Dict[str, SparkSession]:
        """Provide the Spark session. If the udf has a parameter named 'spark'.

        Args:
            udf (Callable[..., T]): the feature_processor wrapped user function.
        """
        return (
            {self.SPARK_SESSION_ARG_NAME: self.spark_session_factory.spark_session}
            if self._has_param(udf, self.SPARK_SESSION_ARG_NAME)
            else {}
        )

    def _get_input_parameters(self, udf_parameter_names: List[str]) -> List[str]:
        """Parses the parameter names from the UDF that correspond to the input data sources.

        This function assumes that the udf signature's `params` and `spark` parameters are at the
        end, in any order, if provided.

        Args:
            udf_parameter_names (List[str]): The full list of parameters names in the UDF.

        Returns:
            List[str]: A subset of parameter names corresponding to the input data sources.
        """
        inputs_end_index = len(udf_parameter_names) - 1

        # Reduce range based on the position of optional kwargs of the UDF.
        if self.PARAMS_ARG_NAME in udf_parameter_names:
            inputs_end_index = udf_parameter_names.index(self.PARAMS_ARG_NAME) - 1

        if self.SPARK_SESSION_ARG_NAME in udf_parameter_names:
            inputs_end_index = min(
                inputs_end_index,
                udf_parameter_names.index(self.SPARK_SESSION_ARG_NAME) - 1,
            )

        return udf_parameter_names[: inputs_end_index + 1]

    def _load_data_frame(
        self,
        data_source: Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource],
    ) -> DataFrame:
        """Given a data source definition, load the data as a Spark DataFrame.

        Args:
            data_source (Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource]):
                A user specified data source from the feature_processor decorator's parameters.

        Returns:
            DataFrame: The contents of the data source as a Spark DataFrame.
        """
        if isinstance(data_source, (CSVDataSource, ParquetDataSource)):
            return self.input_loader.load_from_s3(data_source)

        if isinstance(data_source, FeatureGroupDataSource):
            return self.input_loader.load_from_feature_group(data_source)

        raise ValueError(f"Unknown data source type: {type(data_source)}")

    def _has_param(self, udf: Callable, name: str) -> bool:
        """Determine if a function has a parameter with a given name.

        Args:
            udf (Callable): the user defined function.
            name (str): the name of the parameter.

        Returns:
            bool: True if the udf contains a parameter with the name.
        """
        return name in list(signature(udf).parameters.keys())
