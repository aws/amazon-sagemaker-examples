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
"""Contains factory classes for instantiating Spark objects."""
from __future__ import absolute_import

from functools import lru_cache
from typing import List, Tuple

import feature_store_pyspark
import feature_store_pyspark.FeatureStoreManager as fsm
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

from sagemaker.feature_store.feature_processor._env import EnvironmentHelper

SPARK_APP_NAME = "FeatureProcessor"


class SparkSessionFactory:
    """Lazy loading, memoizing, instantiation of SparkSessions.

    Useful when you want to defer SparkSession instantiation and provide access to the same
    instance throughout the application.
    """

    def __init__(self, environment_helper: EnvironmentHelper) -> None:
        """Initialize the SparkSessionFactory.

        Args:
            environment_helper (EnvironmentHelper): A helper class to determine the current
                execution.
        """
        self.environment_helper = environment_helper

    @property
    @lru_cache()
    def spark_session(self) -> SparkSession:
        """Instantiate a new SparkSession or return the existing one."""
        is_training_job = self.environment_helper.is_training_job()
        instance_count = self.environment_helper.get_instance_count()

        spark_configs = self._get_spark_configs(is_training_job)
        spark_conf = SparkConf().setAll(spark_configs).setAppName(SPARK_APP_NAME)

        if instance_count == 1:
            spark_conf.setMaster("local[*]")

        sc = SparkContext.getOrCreate(conf=spark_conf)

        jsc = sc._jsc  # Java Spark Context (JVM SparkContext)
        for cfg in self._get_jsc_hadoop_configs():
            jsc.hadoopConfiguration().set(cfg[0], cfg[1])

        return SparkSession(sparkContext=sc)

    def _get_spark_configs(self, is_training_job) -> List[Tuple[str, str]]:
        """Generate Spark Configurations optimized for feature_processing functionality.

        Args:
            is_training_job (bool): a boolean indicating whether the current execution environment
                is a training job or not.

        Returns:
            List[Tuple[str, str]]: Spark configurations.
        """
        spark_configs = [
            (
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                ",".join(
                    [
                        "com.amazonaws.auth.ContainerCredentialsProvider",
                        "com.amazonaws.auth.profile.ProfileCredentialsProvider",
                        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                    ]
                ),
            ),
            # spark-3.3.1#recommended-settings-for-writing-to-object-stores - https://tinyurl.com/54rkhef6
            ("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2"),
            (
                "spark.hadoop.mapreduce.fileoutputcommitter.cleanup-failures.ignored",
                "true",
            ),
            ("spark.hadoop.parquet.enable.summary-metadata", "false"),
            # spark-3.3.1#parquet-io-settings https://tinyurl.com/59a7uhwu
            ("spark.sql.parquet.mergeSchema", "false"),
            ("spark.sql.parquet.filterPushdown", "true"),
            ("spark.sql.hive.metastorePartitionPruning", "true"),
            # hadoop-aws#performance - https://tinyurl.com/mutxj96f
            ("spark.hadoop.fs.s3a.threads.max", "500"),
            ("spark.hadoop.fs.s3a.connection.maximum", "500"),
            ("spark.hadoop.fs.s3a.experimental.input.fadvise", "normal"),
            ("spark.hadoop.fs.s3a.block.size", "128M"),
            ("spark.hadoop.fs.s3a.fast.upload.buffer", "disk"),
            ("spark.hadoop.fs.trash.interval", "0"),
            ("spark.port.maxRetries", "50"),
        ]

        if not is_training_job:
            spark_configs.extend(
                (
                    (
                        "spark.jars",
                        ",".join(feature_store_pyspark.classpath_jars()),
                    ),
                    (
                        "spark.jars.packages",
                        ",".join(
                            [
                                "org.apache.hadoop:hadoop-aws:3.3.1",
                                "org.apache.hadoop:hadoop-common:3.3.1",
                            ]
                        ),
                    ),
                )
            )
        return spark_configs

    def _get_jsc_hadoop_configs(self) -> List[Tuple[str, str]]:
        """JVM SparkContext Hadoop configurations."""
        # Skip generation of _SUCCESS files to speed up writes.
        return [("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")]

    def _get_iceberg_configs(self, warehouse_s3_uri: str, catalog: str) -> List[Tuple[str, str]]:
        """Spark configurations for reading and writing data from Iceberg Table sources.

        Args:
            warehouse_s3_uri (str): The S3 URI of the warehouse.
            catalog (str): The catalog.

        Returns:
            List[Tuple[str, str]]: the Spark configurations.
        """
        catalog = catalog.lower()
        return [
            (f"spark.sql.catalog.{catalog}", "smfs.shaded.org.apache.iceberg.spark.SparkCatalog"),
            (f"spark.sql.catalog.{catalog}.warehouse", warehouse_s3_uri),
            (
                f"spark.sql.catalog.{catalog}.catalog-impl",
                "smfs.shaded.org.apache.iceberg.aws.glue.GlueCatalog",
            ),
            (
                f"spark.sql.catalog.{catalog}.io-impl",
                "smfs.shaded.org.apache.iceberg.aws.s3.S3FileIO",
            ),
            (f"spark.sql.catalog.{catalog}.glue.skip-name-validation", "true"),
        ]

    def get_spark_session_with_iceberg_config(self, warehouse_s3_uri, catalog) -> SparkSession:
        """Get an instance of the SparkSession with Iceberg settings configured.

        Args:
            warehouse_s3_uri (str): The S3 URI of the warehouse.
            catalog (str): The catalog.

        Returns:
            SparkSession: A SparkSession ready to support reading and writing data from an Iceberg
                Table.
        """
        conf = self.spark_session.conf

        for cfg in self._get_iceberg_configs(warehouse_s3_uri, catalog):
            conf.set(cfg[0], cfg[1])

        return self.spark_session


class FeatureStoreManagerFactory:
    """Lazy loading, memoizing, instantiation of FeatureStoreManagers.

    Useful when you want to defer FeatureStoreManagers instantiation and provide access to the same
    instance throughout the application.
    """

    @property
    @lru_cache()
    def feature_store_manager(self) -> fsm.FeatureStoreManager:
        """Instansiate a new FeatureStoreManager."""
        return fsm.FeatureStoreManager()
