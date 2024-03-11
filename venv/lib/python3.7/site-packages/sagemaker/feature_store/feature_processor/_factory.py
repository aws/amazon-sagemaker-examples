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
"""Contains static factory classes to instantiate complex objects for the FeatureProcessor."""
from __future__ import absolute_import

from pyspark.sql import DataFrame

from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode
from sagemaker.feature_store.feature_processor._env import EnvironmentHelper
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)
from sagemaker.feature_store.feature_processor._input_loader import (
    SparkDataFrameInputLoader,
)
from sagemaker.feature_store.feature_processor._params_loader import (
    ParamsLoader,
    SystemParamsLoader,
)
from sagemaker.feature_store.feature_processor._spark_factory import (
    FeatureStoreManagerFactory,
    SparkSessionFactory,
)
from sagemaker.feature_store.feature_processor._udf_arg_provider import SparkArgProvider
from sagemaker.feature_store.feature_processor._udf_output_receiver import (
    SparkOutputReceiver,
)
from sagemaker.feature_store.feature_processor._udf_wrapper import UDFWrapper
from sagemaker.feature_store.feature_processor._validation import (
    FeatureProcessorArgValidator,
    InputValidator,
    SparkUDFSignatureValidator,
    InputOffsetValidator,
    ValidatorChain,
)


class ValidatorFactory:
    """Static factory to handle ValidationChain instantiation."""

    @staticmethod
    def get_validation_chain(fp_config: FeatureProcessorConfig) -> ValidatorChain:
        """Instantiate a ValidationChain"""
        base_validators = [
            InputValidator(),
            FeatureProcessorArgValidator(),
            InputOffsetValidator(),
        ]

        mode = fp_config.mode
        if FeatureProcessorMode.PYSPARK == mode:
            base_validators.append(SparkUDFSignatureValidator())
            return ValidatorChain(validators=base_validators)

        raise ValueError(f"FeatureProcessorMode {mode} is not supported.")


class UDFWrapperFactory:
    """Static factory to handle UDFWrapper instantiation at runtime."""

    @staticmethod
    def get_udf_wrapper(fp_config: FeatureProcessorConfig) -> UDFWrapper:
        """Instantiate a UDFWrapper based on the FeatureProcessingMode.

        Args:
            fp_config (FeatureProcessorConfig): the configuration values for the
                feature_processor decorator.

        Raises:
            ValueError: if an unsupported FeatureProcessorMode is provided in fp_config.

        Returns:
            UDFWrapper: An instance of UDFWrapper to decorate the UDF.
        """
        mode = fp_config.mode

        if FeatureProcessorMode.PYSPARK == mode:
            return UDFWrapperFactory._get_spark_udf_wrapper()

        raise ValueError(f"FeatureProcessorMode {mode} is not supported.")

    @staticmethod
    def _get_spark_udf_wrapper() -> UDFWrapper[DataFrame]:
        """Instantiate a new UDFWrapper for PySpark functions."""
        spark_session_factory = UDFWrapperFactory._get_spark_session_factory()
        feature_store_manager_factory = UDFWrapperFactory._get_feature_store_manager_factory()

        output_manager = UDFWrapperFactory._get_spark_output_receiver(feature_store_manager_factory)
        arg_provider = UDFWrapperFactory._get_spark_arg_provider(spark_session_factory)

        return UDFWrapper[DataFrame](arg_provider, output_manager)

    @staticmethod
    def _get_spark_arg_provider(
        spark_session_factory: SparkSessionFactory,
    ) -> SparkArgProvider:
        """Instantiate a new SparkArgProvider for PySpark functions.

        Args:
            spark_session_factory (SparkSessionFactory): A factory to provide a reference to the
                SparkSession initialized for the feature_processor wrapped function. The factory
                lazily loads the SparkSession, i.e. defers to function execution time.

        Returns:
            SparkArgProvider: An instance that generates arguments to provide to the
                feature_processor wrapped function.
        """
        environment_helper = EnvironmentHelper()

        system_parameters_arg_provider = SystemParamsLoader(environment_helper)
        params_arg_provider = ParamsLoader(system_parameters_arg_provider)
        input_loader = SparkDataFrameInputLoader(spark_session_factory, environment_helper)

        return SparkArgProvider(params_arg_provider, input_loader, spark_session_factory)

    @staticmethod
    def _get_spark_output_receiver(
        feature_store_manager_factory: FeatureStoreManagerFactory,
    ) -> SparkOutputReceiver:
        """Instantiate a new SparkOutputManager for PySpark functions.

        Args:
            feature_store_manager_factory (FeatureStoreManagerFactory): A factory to provide
                that provides a FeaturStoreManager that handles data ingestion to a Feature Group.
                The factory lazily loads the FeatureStoreManager.

        Returns:
            SparkOutputReceiver: An instance that handles outputs of the wrapped function.
        """
        return SparkOutputReceiver(feature_store_manager_factory)

    @staticmethod
    def _get_spark_session_factory() -> SparkSessionFactory:
        """Instantiate a new SparkSessionFactory"""
        environment_helper = EnvironmentHelper()
        return SparkSessionFactory(environment_helper)

    @staticmethod
    def _get_feature_store_manager_factory() -> FeatureStoreManagerFactory:
        """Instantiate a new FeatureStoreManagerFactory"""
        return FeatureStoreManagerFactory()
