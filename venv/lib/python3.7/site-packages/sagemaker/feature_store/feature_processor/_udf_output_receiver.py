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
"""Contains classes for handling UDF outputs"""
from __future__ import absolute_import

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attr
from py4j.protocol import Py4JJavaError
from pyspark.sql import DataFrame

from sagemaker.feature_store.feature_processor import IngestionError
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)
from sagemaker.feature_store.feature_processor._spark_factory import (
    FeatureStoreManagerFactory,
)

T = TypeVar("T")

logger = logging.getLogger("sagemaker")


class UDFOutputReceiver(Generic[T], ABC):
    """Base class for handling outputs of the UDF."""

    @abstractmethod
    def ingest_udf_output(self, output: T, fp_config: FeatureProcessorConfig) -> None:
        """Ingests data to the output feature group.

        Args:
            output (T): The output of the feature_processor wrapped function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.
        """


@attr.s
class SparkOutputReceiver(UDFOutputReceiver[DataFrame]):
    """Handles the Spark DataFrame the output from the UDF"""

    feature_store_manager_factory: FeatureStoreManagerFactory = attr.ib()

    def ingest_udf_output(self, output: DataFrame, fp_config: FeatureProcessorConfig) -> None:
        """Ingests UDF to the output Feature Group.

        Args:
            output (T): The output of the feature_processor wrapped function.
            fp_config (FeatureProcessorConfig): The configuration for the feature_processor.

        Raises:
            Py4JError: If there is a problem with Py4J, including client code errors.
            IngestionError: If any rows are not ingested successfully then a sample of the records,
                with failure reasons, is logged.
        """
        if fp_config.enable_ingestion is False:
            logging.info("Ingestion is disabled. Skipping ingestion.")
            return

        logger.info(
            "Ingesting transformed data to %s with target_stores: %s",
            fp_config.output,
            fp_config.target_stores,
        )

        feature_store_manager = self.feature_store_manager_factory.feature_store_manager
        try:
            feature_store_manager.ingest_data(
                input_data_frame=output,
                feature_group_arn=fp_config.output,
                target_stores=fp_config.target_stores,
            )
        except Py4JJavaError as e:
            if e.java_exception.getClass().getSimpleName() == "StreamIngestionFailureException":
                logger.warning(
                    "Ingestion did not complete successfully. Failed records and error messages"
                    " have been printed to the console."
                )
                feature_store_manager.get_failed_stream_ingestion_data_frame().show(
                    n=20, truncate=False
                )
                raise IngestionError(e.java_exception)

            raise e

        logger.info("Ingestion to %s complete.", fp_config.output)
