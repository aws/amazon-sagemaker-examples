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
"""Contains classes that loads user specified input sources (e.g. Feature Groups, S3 URIs, etc)."""
from __future__ import absolute_import

import logging
import re
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

import attr
from pyspark.sql import DataFrame

from sagemaker import Session
from sagemaker.feature_store.feature_processor._constants import FEATURE_GROUP_ARN_REGEX_PATTERN
from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
    IcebergTableDataSource,
)
from sagemaker.feature_store.feature_processor._spark_factory import SparkSessionFactory
from sagemaker.feature_store.feature_processor._input_offset_parser import (
    InputOffsetParser,
)
from sagemaker.feature_store.feature_processor._env import EnvironmentHelper

T = TypeVar("T")

logger = logging.getLogger("sagemaker")


class InputLoader(Generic[T], ABC):
    """Loads the contents of a Feature Group's offline store or contents at an S3 URI."""

    @abstractmethod
    def load_from_feature_group(self, feature_group_data_source: FeatureGroupDataSource) -> T:
        """Load the data from a Feature Group's offline store.

        Args:
            feature_group_data_source (FeatureGroupDataSource): the feature group source.

        Returns:
            T: The contents of the offline store as an instance of type T.
        """

    @abstractmethod
    def load_from_s3(self, s3_data_source: Union[CSVDataSource, ParquetDataSource]) -> T:
        """Load the contents from an S3 based data source.

        Args:
            s3_data_source (Union[CSVDataSource, ParquetDataSource]): a data source that is based
                in S3.

        Returns:
            T: The contents stored at the data source as an instance of type T.
        """


@attr.s
class SparkDataFrameInputLoader(InputLoader[DataFrame]):
    """InputLoader that reads data in as a Spark DataFrame."""

    spark_session_factory: SparkSessionFactory = attr.ib()
    environment_helper: EnvironmentHelper = attr.ib()
    sagemaker_session: Optional[Session] = attr.ib(default=None)

    _supported_table_format = ["Iceberg", "Glue", None]

    def load_from_feature_group(
        self, feature_group_data_source: FeatureGroupDataSource
    ) -> DataFrame:
        """Load the contents of a Feature Group's offline store as a DataFrame.

        Args:
            feature_group_data_source (FeatureGroupDataSource): the Feature Group source.

        Raises:
            ValueError: If the Feature Group does not have an Offline Store.
            ValueError: If the Feature Group's Table Type is not supported by the feature_processor.

        Returns:
            DataFrame: A Spark DataFrame containing the contents of the Feature Group's
                offline store.
        """
        sagemaker_session: Session = self.sagemaker_session or Session()

        feature_group_name = feature_group_data_source.name
        feature_group = sagemaker_session.describe_feature_group(
            self._parse_name_from_arn(feature_group_name)
        )
        logger.debug(
            "Called describe_feature_group with %s and received: %s",
            feature_group_name,
            feature_group,
        )

        if "OfflineStoreConfig" not in feature_group:
            raise ValueError(
                f"Input Feature Groups must have an enabled Offline Store."
                f" Feature Group: {feature_group_name} does not have an Offline Store enabled."
            )

        offline_store_uri = feature_group["OfflineStoreConfig"]["S3StorageConfig"][
            "ResolvedOutputS3Uri"
        ]

        table_format = feature_group["OfflineStoreConfig"].get("TableFormat", None)

        if table_format not in self._supported_table_format:
            raise ValueError(
                f"Feature group with table format {table_format} is not supported. "
                f"The table format should be one of {self._supported_table_format}."
            )

        start_offset = feature_group_data_source.input_start_offset
        end_offset = feature_group_data_source.input_end_offset

        if table_format == "Iceberg":
            data_catalog_config = feature_group["OfflineStoreConfig"]["DataCatalogConfig"]
            return self.load_from_iceberg_table(
                IcebergTableDataSource(
                    offline_store_uri,
                    data_catalog_config["Catalog"],
                    data_catalog_config["Database"],
                    data_catalog_config["TableName"],
                ),
                feature_group["EventTimeFeatureName"],
                start_offset,
                end_offset,
            )

        return self.load_from_date_partitioned_s3(
            ParquetDataSource(offline_store_uri), start_offset, end_offset
        )

    def load_from_date_partitioned_s3(
        self,
        s3_data_source: ParquetDataSource,
        input_start_offset: str,
        input_end_offset: str,
    ) -> DataFrame:
        """Load the contents from a Feature Group's partitioned offline S3 as a DataFrame.

        Args:
            s3_data_source (ParquetDataSource):
                A data source that is based in S3.
            input_start_offset (str): Start offset that is used to calculate the input start date.
            input_end_offset (str): End offset that is used to calculate the input end date.

        Returns:
            DataFrame: Contents of the data loaded from S3.
        """

        spark_session = self.spark_session_factory.spark_session
        s3a_uri = s3_data_source.s3_uri.replace("s3://", "s3a://")
        filter_condition = self._get_s3_partitions_offset_filter_condition(
            input_start_offset, input_end_offset
        )

        logger.info(
            "Loading data from %s with filtering condition %s.",
            s3a_uri,
            filter_condition,
        )
        input_df = spark_session.read.parquet(s3a_uri)
        if filter_condition:
            input_df = input_df.filter(filter_condition)

        return input_df

    def load_from_s3(self, s3_data_source: Union[CSVDataSource, ParquetDataSource]) -> DataFrame:
        """Load the contents from an S3 based data source as a DataFrame.

        Args:
            s3_data_source (Union[CSVDataSource, ParquetDataSource]):
                A data source that is based in S3.

        Raises:
            ValueError: If an invalid DataSource is provided.

        Returns:
            DataFrame: Contents of the data loaded from S3.
        """
        spark_session = self.spark_session_factory.spark_session
        s3a_uri = s3_data_source.s3_uri.replace("s3://", "s3a://")

        if isinstance(s3_data_source, CSVDataSource):
            # TODO: Accept `schema` parameter. (Inferring schema requires a pass through every row)
            logger.info("Loading data from %s.", s3a_uri)
            return spark_session.read.csv(
                s3a_uri,
                header=s3_data_source.csv_header,
                inferSchema=s3_data_source.csv_infer_schema,
            )

        if isinstance(s3_data_source, ParquetDataSource):
            logger.info("Loading data from %s.", s3a_uri)
            return spark_session.read.parquet(s3a_uri)

        raise ValueError("An invalid data source was provided.")

    def load_from_iceberg_table(
        self,
        iceberg_table_data_source: IcebergTableDataSource,
        event_time_feature_name: str,
        input_start_offset: str,
        input_end_offset: str,
    ) -> DataFrame:
        """Load the contents from an Iceberg table as a DataFrame.

        Args:
            iceberg_table_data_source (IcebergTableDataSource): An Iceberg Table source.
            event_time_feature_name (str): Event time feature's name of feature group.
            input_start_offset (str): Start offset that is used to calculate the input start date.
            input_end_offset (str): End offset that is used to calculate the input end date.

        Returns:
            DataFrame: Contents of the Iceberg Table as a Spark DataFrame.
        """
        catalog = iceberg_table_data_source.catalog.lower()
        database = iceberg_table_data_source.database.lower()
        table = iceberg_table_data_source.table.lower()
        iceberg_table = f"{catalog}.{database}.{table}"

        spark_session = self.spark_session_factory.get_spark_session_with_iceberg_config(
            iceberg_table_data_source.warehouse_s3_uri, catalog
        )

        filter_condition = self._get_iceberg_offset_filter_condition(
            event_time_feature_name,
            input_start_offset,
            input_end_offset,
        )

        iceberg_df = spark_session.table(iceberg_table)

        if filter_condition:
            logger.info(
                "The filter condition for iceberg feature group is %s.",
                filter_condition,
            )
            iceberg_df = iceberg_df.filter(filter_condition)

        return iceberg_df

    def _get_iceberg_offset_filter_condition(
        self,
        event_time_feature_name: str,
        input_start_offset: str,
        input_end_offset: str,
    ):
        """Load the contents from an Iceberg table as a DataFrame.

        Args:
            iceberg_table_data_source (IcebergTableDataSource): An Iceberg Table source.
            input_start_offset (str): Start offset that is used to calculate the input start date.
            input_end_offset (str): End offset that is used to calculate the input end date.

        Returns:
            DataFrame: Contents of the Iceberg Table as a Spark DataFrame.
        """
        if input_start_offset is None and input_end_offset is None:
            return None

        offset_parser = InputOffsetParser(self.environment_helper.get_job_scheduled_time())
        start_offset_time = offset_parser.get_iso_format_offset_date(input_start_offset)
        end_offset_time = offset_parser.get_iso_format_offset_date(input_end_offset)

        start_condition = (
            f"{event_time_feature_name} >= '{start_offset_time}'" if input_start_offset else None
        )
        end_condition = (
            f"{event_time_feature_name} < '{end_offset_time}'" if input_end_offset else None
        )

        conditions = filter(None, [start_condition, end_condition])
        return " AND ".join(conditions)

    def _get_s3_partitions_offset_filter_condition(
        self, input_start_offset: str, input_end_offset: str
    ) -> str:
        """Get s3 partitions filter condition based on input offsets.

        Args:
            input_start_offset (str): Start offset that is used to calculate the input start date.
            input_end_offset (str): End offset that is used to calculate the input end date.

        Returns:
            str: A SQL string that defines the condition of time range filter.
        """
        if input_start_offset is None and input_end_offset is None:
            return None

        offset_parser = InputOffsetParser(self.environment_helper.get_job_scheduled_time())
        (
            start_year,
            start_month,
            start_day,
            start_hour,
        ) = offset_parser.get_offset_date_year_month_day_hour(input_start_offset)
        (
            end_year,
            end_month,
            end_day,
            end_hour,
        ) = offset_parser.get_offset_date_year_month_day_hour(input_end_offset)

        # Include all records that event time is between start_year and end_year
        start_year_include_condition = f"year >= '{start_year}'" if input_start_offset else None
        end_year_include_condition = f"year <= '{end_year}'" if input_end_offset else None
        year_include_condition = " AND ".join(
            filter(None, [start_year_include_condition, end_year_include_condition])
        )

        # Exclude all records that the event time is earlier than the start or later than the end
        start_offset_exclude_condition = (
            f"(year = '{start_year}' AND month < '{start_month}') "
            f"OR (year = '{start_year}' AND month = '{start_month}' AND day < '{start_day}') "
            f"OR (year = '{start_year}' AND month = '{start_month}' AND day = '{start_day}' "
            f"AND hour < '{start_hour}')"
            if input_start_offset
            else None
        )
        end_offset_exclude_condition = (
            f"(year = '{end_year}' AND month > '{end_month}') "
            f"OR (year = '{end_year}' AND month = '{end_month}' AND day > '{end_day}') "
            f"OR (year = '{end_year}' AND month = '{end_month}' AND day = '{end_day}' "
            f"AND hour >= '{end_hour}')"
            if input_end_offset
            else None
        )
        offset_exclude_condition = " OR ".join(
            filter(None, [start_offset_exclude_condition, end_offset_exclude_condition])
        )

        filter_condition = f"({year_include_condition}) AND NOT ({offset_exclude_condition})"

        logger.info("The filter condition for hive feature group is %s.", filter_condition)

        return filter_condition

    def _parse_name_from_arn(self, fg_uri: str) -> str:
        """Parse a Feature Group's name from an arn.

        Args:
            fg_uri (str): a string identifier of the Feature Group.

        Returns:
            str: the name of the feature group.
        """
        match = re.match(FEATURE_GROUP_ARN_REGEX_PATTERN, fg_uri)
        if match:
            feature_group_name = match.group(4)
            return feature_group_name
        return fg_uri
