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
"""The FeatureGroup entity for FeatureStore.

A feature group is a logical grouping of features, defined in the Feature Store,
to describe records. A feature group definition is composed of a list of feature definitions,
a record identifier name, and configurations for its online and offline store.
Create feature group, describe feature group, update feature groups, delete feature group and
list feature groups APIs can be used to manage feature groups.
"""

from __future__ import absolute_import

import copy
import logging
import math
import os
import tempfile
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, List, Dict, Any, Union
from urllib.parse import urlparse

from multiprocessing.pool import AsyncResult
import signal
import attr
import pandas as pd
from pandas import DataFrame

import boto3
from botocore.config import Config
from pathos.multiprocessing import ProcessingPool

from sagemaker.config import (
    FEATURE_GROUP_ROLE_ARN_PATH,
    FEATURE_GROUP_OFFLINE_STORE_KMS_KEY_ID_PATH,
    FEATURE_GROUP_ONLINE_STORE_KMS_KEY_ID_PATH,
)
from sagemaker.session import Session
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)
from sagemaker.feature_store.inputs import (
    OnlineStoreConfig,
    OnlineStoreSecurityConfig,
    S3StorageConfig,
    OfflineStoreConfig,
    DataCatalogConfig,
    FeatureValue,
    FeatureParameter,
    TableFormatEnum,
    DeletionModeEnum,
    TtlDuration,
    OnlineStoreConfigUpdate,
)
from sagemaker.utils import resolve_value_from_config

logger = logging.getLogger(__name__)


@attr.s
class AthenaQuery:
    """Class to manage querying of feature store data with AWS Athena.

    This class instantiates a AthenaQuery object that is used to retrieve data from feature store
    via standard SQL queries.

    Attributes:
        catalog (str): name of the data catalog.
        database (str): name of the database.
        table_name (str): name of the table.
        sagemaker_session (Session): instance of the Session class to perform boto calls.
    """

    catalog: str = attr.ib()
    database: str = attr.ib()
    table_name: str = attr.ib()
    sagemaker_session: Session = attr.ib()
    _current_query_execution_id: str = attr.ib(init=False, default=None)
    _result_bucket: str = attr.ib(init=False, default=None)
    _result_file_prefix: str = attr.ib(init=False, default=None)

    def run(
        self, query_string: str, output_location: str, kms_key: str = None, workgroup: str = None
    ) -> str:
        """Execute a SQL query given a query string, output location and kms key.

        This method executes the SQL query using Athena and outputs the results to output_location
        and returns the execution id of the query.

        Args:
            query_string: SQL query string.
            output_location: S3 URI of the query result.
            kms_key: KMS key id. If set, will be used to encrypt the query result file.
            workgroup (str): The name of the workgroup in which the query is being started.

        Returns:
            Execution id of the query.
        """
        response = self.sagemaker_session.start_query_execution(
            catalog=self.catalog,
            database=self.database,
            query_string=query_string,
            output_location=output_location,
            kms_key=kms_key,
            workgroup=workgroup,
        )
        self._current_query_execution_id = response["QueryExecutionId"]
        parse_result = urlparse(output_location, allow_fragments=False)
        self._result_bucket = parse_result.netloc
        self._result_file_prefix = parse_result.path.strip("/")
        return self._current_query_execution_id

    def wait(self):
        """Wait for the current query to finish."""
        self.sagemaker_session.wait_for_athena_query(
            query_execution_id=self._current_query_execution_id
        )

    def get_query_execution(self) -> Dict[str, Any]:
        """Get execution status of the current query.

        Returns:
            Response dict from Athena.
        """
        return self.sagemaker_session.get_query_execution(
            query_execution_id=self._current_query_execution_id
        )

    def as_dataframe(self) -> DataFrame:
        """Download the result of the current query and load it into a DataFrame.

        Returns:
            A pandas DataFrame contains the query result.
        """
        query_state = self.get_query_execution().get("QueryExecution").get("Status").get("State")
        if query_state != "SUCCEEDED":
            if query_state in ("QUEUED", "RUNNING"):
                raise RuntimeError(
                    f"Current query {self._current_query_execution_id} is still being executed."
                )
            raise RuntimeError(f"Failed to execute query {self._current_query_execution_id}")

        output_filename = os.path.join(
            tempfile.gettempdir(), f"{self._current_query_execution_id}.csv"
        )
        self.sagemaker_session.download_athena_query_result(
            bucket=self._result_bucket,
            prefix=self._result_file_prefix,
            query_execution_id=self._current_query_execution_id,
            filename=output_filename,
        )
        return pd.read_csv(output_filename, delimiter=",")


@attr.s
class IngestionManagerPandas:
    """Class to manage the multi-threaded data ingestion process.

    This class will manage the data ingestion process which is multi-threaded.

    Attributes:
        feature_group_name (str): name of the Feature Group.
        sagemaker_fs_runtime_client_config (Config): instance of the Config class
            for boto calls.
        sagemaker_session (Session): session instance to perform boto calls.
        data_frame (DataFrame): pandas DataFrame to be ingested to the given feature group.
        max_workers (int): number of threads to create.
        max_processes (int): number of processes to create. Each process spawns
            ``max_workers`` threads.
        profile_name (str): the profile credential should be used for ``PutRecord``
            (default: None).
    """

    feature_group_name: str = attr.ib()
    sagemaker_fs_runtime_client_config: Config = attr.ib(default=None)
    sagemaker_session: Session = attr.ib(default=None)
    max_workers: int = attr.ib(default=1)
    max_processes: int = attr.ib(default=1)
    profile_name: str = attr.ib(default=None)
    _async_result: AsyncResult = attr.ib(default=None)
    _processing_pool: ProcessingPool = attr.ib(default=None)
    _failed_indices: List[int] = attr.ib(factory=list)

    @staticmethod
    def _ingest_single_batch(
        data_frame: DataFrame,
        feature_group_name: str,
        client_config: Config,
        start_index: int,
        end_index: int,
        profile_name: str = None,
    ) -> List[int]:
        """Ingest a single batch of DataFrame rows into FeatureStore.

        Args:
            data_frame (DataFrame): source DataFrame to be ingested.
            feature_group_name (str): name of the Feature Group.
            client_config (Config): Configuration for the sagemaker feature store runtime
                client to perform boto calls.
            start_index (int): starting position to ingest in this batch.
            end_index (int): ending position to ingest in this batch.
            profile_name (str): the profile credential should be used for ``PutRecord``
                (default: None).

        Returns:
            List of row indices that failed to be ingested.
        """
        retry_config = client_config.retries
        if "max_attempts" not in retry_config and "total_max_attempts" not in retry_config:
            client_config = copy.deepcopy(client_config)
            client_config.retries = {"max_attempts": 10, "mode": "standard"}
        sagemaker_fs_runtime_client = boto3.Session(profile_name=profile_name).client(
            service_name="sagemaker-featurestore-runtime", config=client_config
        )

        logger.info("Started ingesting index %d to %d", start_index, end_index)
        failed_rows = list()
        for row in data_frame[start_index:end_index].itertuples():
            IngestionManagerPandas._ingest_row(
                data_frame=data_frame,
                row=row,
                feature_group_name=feature_group_name,
                sagemaker_fs_runtime_client=sagemaker_fs_runtime_client,
                failed_rows=failed_rows,
            )
        return failed_rows

    @property
    def failed_rows(self) -> List[int]:
        """Get rows that failed to ingest.

        Returns:
            List of row indices that failed to be ingested.
        """
        return self._failed_indices

    def wait(self, timeout=None):
        """Wait for the ingestion process to finish.

        Args:
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.
        """
        try:
            results = self._async_result.get(timeout=timeout)
        except KeyboardInterrupt as i:
            # terminate workers abruptly on keyboard interrupt.
            self._processing_pool.terminate()
            self._processing_pool.close()
            self._processing_pool.clear()
            raise i
        else:
            # terminate normally
            self._processing_pool.close()
            self._processing_pool.clear()

        self._failed_indices = [
            failed_index for failed_indices in results for failed_index in failed_indices
        ]

        if len(self._failed_indices) > 0:
            raise IngestionError(
                self._failed_indices,
                f"Failed to ingest some data into FeatureGroup {self.feature_group_name}",
            )

    @staticmethod
    def _ingest_row(
        data_frame: DataFrame,
        row: int,
        feature_group_name: str,
        sagemaker_fs_runtime_client: Session,
        failed_rows: List[int],
    ):
        """Ingest a single Dataframe row into FeatureStore.

        Args:
            data_frame (DataFrame): source DataFrame to be ingested.
            row (int): current row that is being ingested
            feature_group_name (str): name of the Feature Group.
            sagemaker_featurestore_runtime_client (Session): session instance to perform boto calls.
            failed_rows (List[int]): list of indices from the data frame for which ingestion failed.


        Returns:
            int of row indices that failed to be ingested.
        """
        record = [
            FeatureValue(
                feature_name=data_frame.columns[index - 1],
                value_as_string=str(row[index]),
            )
            for index in range(1, len(row))
            if pd.notna(row[index])
        ]
        try:
            sagemaker_fs_runtime_client.put_record(
                FeatureGroupName=feature_group_name,
                Record=[value.to_dict() for value in record],
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to ingest row %d: %s", row[0], e)
            failed_rows.append(row[0])

    def _run_single_process_single_thread(self, data_frame: DataFrame):
        """Ingest a utilizing single process and single thread.

        Args:
            data_frame (DataFrame): source DataFrame to be ingested.
        """
        logger.info("Started ingesting index %d to %d")
        failed_rows = list()
        sagemaker_fs_runtime_client = self.sagemaker_session.sagemaker_featurestore_runtime_client
        for row in data_frame.itertuples():
            IngestionManagerPandas._ingest_row(
                data_frame=data_frame,
                row=row,
                feature_group_name=self.feature_group_name,
                sagemaker_fs_runtime_client=sagemaker_fs_runtime_client,
                failed_rows=failed_rows,
            )
        self._failed_indices = failed_rows

        if len(self._failed_indices) > 0:
            raise IngestionError(
                self._failed_indices,
                f"Failed to ingest some data into FeatureGroup {self.feature_group_name}",
            )

    def _run_multi_process(self, data_frame: DataFrame, wait=True, timeout=None):
        """Start the ingestion process with the specified number of processes.

        Args:
            data_frame (DataFrame): source DataFrame to be ingested.
            wait (bool): whether to wait for the ingestion to finish or not.
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.
        """
        # pylint: disable=I1101
        batch_size = math.ceil(data_frame.shape[0] / self.max_processes)
        # pylint: enable=I1101

        args = []
        for i in range(self.max_processes):
            start_index = min(i * batch_size, data_frame.shape[0])
            end_index = min(i * batch_size + batch_size, data_frame.shape[0])
            args += [
                (
                    self.max_workers,
                    self.feature_group_name,
                    self.sagemaker_fs_runtime_client_config,
                    data_frame[start_index:end_index],
                    start_index,
                    timeout,
                    self.profile_name,
                )
            ]

        def init_worker():
            # ignore keyboard interrupts in child processes.
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        self._processing_pool = ProcessingPool(self.max_processes, init_worker)
        self._processing_pool.restart(force=True)

        f = lambda x: IngestionManagerPandas._run_multi_threaded(*x)  # noqa: E731
        self._async_result = self._processing_pool.amap(f, args)

        if wait:
            self.wait(timeout=timeout)

    @staticmethod
    def _run_multi_threaded(
        max_workers: int,
        feature_group_name: str,
        sagemaker_fs_runtime_client_config: Config,
        data_frame: DataFrame,
        row_offset=0,
        timeout=None,
        profile_name=None,
    ) -> List[int]:
        """Start the ingestion process.

        Args:
            data_frame (DataFrame): source DataFrame to be ingested.
            row_offset (int): if ``data_frame`` is a partition of a parent DataFrame, then the
                index of the parent where ``data_frame`` starts. Otherwise, 0.
            wait (bool): whether to wait for the ingestion to finish or not.
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.
            profile_name (str): the profile credential should be used for ``PutRecord``
                (default: None).

        Returns:
            List of row indices that failed to be ingested.
        """
        executor = ThreadPoolExecutor(max_workers=max_workers)
        # pylint: disable=I1101
        batch_size = math.ceil(data_frame.shape[0] / max_workers)
        # pylint: enable=I1101

        futures = {}
        for i in range(max_workers):
            start_index = min(i * batch_size, data_frame.shape[0])
            end_index = min(i * batch_size + batch_size, data_frame.shape[0])
            futures[
                executor.submit(
                    IngestionManagerPandas._ingest_single_batch,
                    feature_group_name=feature_group_name,
                    data_frame=data_frame,
                    start_index=start_index,
                    end_index=end_index,
                    client_config=sagemaker_fs_runtime_client_config,
                    profile_name=profile_name,
                )
            ] = (start_index + row_offset, end_index + row_offset)

        failed_indices = list()
        for future in as_completed(futures, timeout=timeout):
            start, end = futures[future]
            failed_rows = future.result()
            if not failed_rows:
                logger.info("Successfully ingested row %d to %d", start, end)
            failed_indices += failed_rows

        executor.shutdown(wait=False)

        return failed_indices

    def run(self, data_frame: DataFrame, wait=True, timeout=None):
        """Start the ingestion process.

        Args:
            data_frame (DataFrame): source DataFrame to be ingested.
            wait (bool): whether to wait for the ingestion to finish or not.
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.
        """
        if self.max_workers == 1 and self.max_processes == 1 and self.profile_name is None:
            self._run_single_process_single_thread(data_frame=data_frame)
        else:
            self._run_multi_process(data_frame=data_frame, wait=wait, timeout=timeout)


class IngestionError(Exception):
    """Exception raised for errors during ingestion.

    Attributes:
        failed_rows: list of indices from the data frame for which ingestion failed.
        message: explanation of the error
    """

    def __init__(self, failed_rows, message):
        super(IngestionError, self).__init__(message)
        self.failed_rows = failed_rows
        self.message = message

    def __str__(self) -> str:
        """String representation of the error."""
        return f"{self.failed_rows} -> {self.message}"


@attr.s
class FeatureGroup:
    """FeatureGroup definition.

    This class instantiates a FeatureGroup object that comprises of a name for the FeatureGroup,
    session instance, and a list of feature definition objects i.e., FeatureDefinition.

    Attributes:
        name (str): name of the FeatureGroup instance.
        sagemaker_session (Session): session instance to perform boto calls.
            If None, a new Session will be created.
        feature_definitions (Sequence[FeatureDefinition]): list of FeatureDefinitions.
    """

    name: str = attr.ib(factory=str)
    sagemaker_session: Session = attr.ib(factory=Session)
    feature_definitions: Sequence[FeatureDefinition] = attr.ib(factory=list)

    _INTEGER_TYPES = [
        "int_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]
    _FLOAT_TYPES = ["float_", "float16", "float32", "float64"]
    DTYPE_TO_FEATURE_DEFINITION_CLS_MAP: Dict[str, FeatureTypeEnum] = {
        type: FeatureTypeEnum.INTEGRAL for type in _INTEGER_TYPES
    }
    DTYPE_TO_FEATURE_DEFINITION_CLS_MAP.update(
        {type: FeatureTypeEnum.FRACTIONAL for type in _FLOAT_TYPES}
    )
    DTYPE_TO_FEATURE_DEFINITION_CLS_MAP["string"] = FeatureTypeEnum.STRING
    DTYPE_TO_FEATURE_DEFINITION_CLS_MAP["object"] = FeatureTypeEnum.STRING

    _FEATURE_TYPE_TO_DDL_DATA_TYPE_MAP = {
        FeatureTypeEnum.INTEGRAL.value: "INT",
        FeatureTypeEnum.FRACTIONAL.value: "FLOAT",
        FeatureTypeEnum.STRING.value: "STRING",
    }

    def create(
        self,
        s3_uri: Union[str, bool],
        record_identifier_name: str,
        event_time_feature_name: str,
        role_arn: str = None,
        online_store_kms_key_id: str = None,
        enable_online_store: bool = False,
        ttl_duration: TtlDuration = None,
        offline_store_kms_key_id: str = None,
        disable_glue_table_creation: bool = False,
        data_catalog_config: DataCatalogConfig = None,
        description: str = None,
        tags: List[Dict[str, str]] = None,
        table_format: TableFormatEnum = None,
    ) -> Dict[str, Any]:
        """Create a SageMaker FeatureStore FeatureGroup.

        Args:
            s3_uri (Union[str, bool]): S3 URI of the offline store, set to
                ``False`` to disable offline store.
            record_identifier_name (str): name of the record identifier feature.
            event_time_feature_name (str): name of the event time feature.
            role_arn (str): ARN of the role used to call CreateFeatureGroup.
            online_store_kms_key_id (str): KMS key ARN for online store (default: None).
            ttl_duration (TtlDuration): Default time to live duration for records (default: None).
            enable_online_store (bool): whether to enable online store or not (default: False).
            offline_store_kms_key_id (str): KMS key ARN for offline store (default: None).
                If a KMS encryption key is not specified, SageMaker encrypts all data at
                rest using the default AWS KMS key. By defining your bucket-level key for
                SSE, you can reduce the cost of AWS KMS requests.
                For more information, see
                `Bucket Key
                <https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucket-key.html>`_
                in the Amazon S3 User Guide.
            disable_glue_table_creation (bool): whether to turn off Glue table creation
                or not (default: False).
            data_catalog_config (DataCatalogConfig): configuration for
                Metadata store (default: None).
            description (str): description of the FeatureGroup (default: None).
            tags (List[Dict[str, str]]): list of tags for labeling a FeatureGroup (default: None).
            table_format (TableFormatEnum): format of the offline store table (default: None).

        Returns:
            Response dict from service.
        """
        role_arn = resolve_value_from_config(
            role_arn, FEATURE_GROUP_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        offline_store_kms_key_id = resolve_value_from_config(
            offline_store_kms_key_id,
            FEATURE_GROUP_OFFLINE_STORE_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        online_store_kms_key_id = resolve_value_from_config(
            online_store_kms_key_id,
            FEATURE_GROUP_ONLINE_STORE_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        if not role_arn:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig,
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create a Feature Group.")
        create_feature_store_args = dict(
            feature_group_name=self.name,
            record_identifier_name=record_identifier_name,
            event_time_feature_name=event_time_feature_name,
            feature_definitions=[
                feature_definition.to_dict() for feature_definition in self.feature_definitions
            ],
            role_arn=role_arn,
            description=description,
            tags=tags,
        )

        # online store configuration
        if enable_online_store:
            online_store_config = OnlineStoreConfig(
                enable_online_store=enable_online_store,
                ttl_duration=ttl_duration,
            )
            if online_store_kms_key_id is not None:
                online_store_config.online_store_security_config = OnlineStoreSecurityConfig(
                    kms_key_id=online_store_kms_key_id
                )
            create_feature_store_args.update({"online_store_config": online_store_config.to_dict()})

        # offline store configuration
        if s3_uri:
            s3_storage_config = S3StorageConfig(s3_uri=s3_uri)
            if offline_store_kms_key_id:
                s3_storage_config.kms_key_id = offline_store_kms_key_id
            offline_store_config = OfflineStoreConfig(
                s3_storage_config=s3_storage_config,
                disable_glue_table_creation=disable_glue_table_creation,
                data_catalog_config=data_catalog_config,
                table_format=table_format,
            )
            create_feature_store_args.update(
                {"offline_store_config": offline_store_config.to_dict()}
            )

        return self.sagemaker_session.create_feature_group(**create_feature_store_args)

    def delete(self):
        """Delete a FeatureGroup."""
        self.sagemaker_session.delete_feature_group(feature_group_name=self.name)

    def describe(self, next_token: str = None) -> Dict[str, Any]:
        """Describe a FeatureGroup.

        Args:
            next_token (str): next_token to get next page of features.

        Returns:
            Response dict from the service.
        """
        return self.sagemaker_session.describe_feature_group(
            feature_group_name=self.name, next_token=next_token
        )

    def update(
        self,
        feature_additions: Sequence[FeatureDefinition] = None,
        online_store_config: OnlineStoreConfigUpdate = None,
    ) -> Dict[str, Any]:
        """Update a FeatureGroup and add new features from the given feature definitions.

        Args:
            feature_additions (Sequence[Dict[str, str]): list of feature definitions to be updated.
            online_store_config (OnlineStoreConfigUpdate): online store config to be updated.

        Returns:
            Response dict from service.
        """

        if feature_additions is None:
            feature_additions_parameter = None
        else:
            feature_additions_parameter = [
                feature_addition.to_dict() for feature_addition in feature_additions
            ]

        if online_store_config is None:
            online_store_config_parameter = None
        else:
            online_store_config_parameter = online_store_config.to_dict()

        return self.sagemaker_session.update_feature_group(
            feature_group_name=self.name,
            feature_additions=feature_additions_parameter,
            online_store_config=online_store_config_parameter,
        )

    def update_feature_metadata(
        self,
        feature_name: str,
        description: str = None,
        parameter_additions: Sequence[FeatureParameter] = None,
        parameter_removals: Sequence[str] = None,
    ) -> Dict[str, Any]:
        """Update a feature metadata and add/remove metadata.

        Args:
            feature_name (str): name of the feature to update.
            description (str): description of the feature to update.
            parameter_additions (Sequence[Dict[str, str]): list of feature parameter to be added.
            parameter_removals (Sequence[str]): list of feature parameter key to be removed.

        Returns:
            Response dict from service.
        """
        return self.sagemaker_session.update_feature_metadata(
            feature_group_name=self.name,
            feature_name=feature_name,
            description=description,
            parameter_additions=[
                parameter_addition.to_dict() for parameter_addition in (parameter_additions or [])
            ],
            parameter_removals=(parameter_removals or []),
        )

    def describe_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """Describe feature metadata by feature name.

        Args:
            feature_name (str): name of the feature.
        Returns:
            Response dict from service.
        """

        return self.sagemaker_session.describe_feature_metadata(
            feature_group_name=self.name, feature_name=feature_name
        )

    def list_tags(self) -> Sequence[Dict[str, str]]:
        """List all tags for a feature group.

        Returns:
            list of key, value pair of the tags.
        """

        feature_group_arn = self.sagemaker_session.describe_feature_group(
            feature_group_name=self.name
        ).get("FeatureGroupArn")

        return self.sagemaker_session.list_tags(resource_arn=feature_group_arn)

    def list_parameters_for_feature_metadata(self, feature_name: str) -> Sequence[Dict[str, str]]:
        """List all parameters for a feature metadata.

        Args:
            feature_name (str): name of the feature.
        Returns:
            list of key, value pair of the parameters.
        """

        return self.sagemaker_session.describe_feature_metadata(
            feature_group_name=self.name, feature_name=feature_name
        ).get("Parameters")

    def load_feature_definitions(
        self,
        data_frame: DataFrame,
    ) -> Sequence[FeatureDefinition]:
        """Load feature definitions from a Pandas DataFrame.

        Column name is used as feature name. Feature type is inferred from the dtype
        of the column. Dtype int_, int8, int16, int32, int64, uint8, uint16, uint32
        and uint64 are mapped to Integral feature type. Dtype float_, float16, float32
        and float64 are mapped to Fractional feature type. string dtype is mapped to
        String feature type.

        No feature definitions will be loaded if the given data_frame contains
        unsupported dtypes.

        Args:
            data_frame (DataFrame):

        Returns:
            list of FeatureDefinition
        """
        feature_definitions = []
        for column in data_frame:
            feature_type = self.DTYPE_TO_FEATURE_DEFINITION_CLS_MAP.get(
                str(data_frame[column].dtype).lower(), None
            )
            if feature_type:
                feature_definitions.append(
                    FeatureDefinition(feature_name=column, feature_type=feature_type)
                )
            else:
                raise ValueError(
                    f"Failed to infer Feature type based on dtype {data_frame[column].dtype} "
                    f"for column {column}."
                )
        self.feature_definitions = feature_definitions
        return self.feature_definitions

    def get_record(
        self,
        record_identifier_value_as_string: str,
        feature_names: Sequence[str] = None,
    ) -> Sequence[Dict[str, str]]:
        """Get a single record in a FeatureGroup

        Args:
            record_identifier_value_as_string (String):
                a String representing the value of the record identifier.
            feature_names (Sequence[String]):
                a list of Strings representing feature names.
        """
        return self.sagemaker_session.get_record(
            record_identifier_value_as_string=record_identifier_value_as_string,
            feature_group_name=self.name,
            feature_names=feature_names,
        ).get("Record")

    def put_record(self, record: Sequence[FeatureValue], ttl_duration: TtlDuration = None):
        """Put a single record in the FeatureGroup.

        Args:
            record (Sequence[FeatureValue]): a list contains feature values.
            ttl_duration (TtlDuration): customer specified ttl duration.
        """

        if ttl_duration is not None:
            return self.sagemaker_session.put_record(
                feature_group_name=self.name,
                record=[value.to_dict() for value in record],
                ttl_duration=ttl_duration.to_dict(),
            )

        return self.sagemaker_session.put_record(
            feature_group_name=self.name,
            record=[value.to_dict() for value in record],
        )

    def delete_record(
        self,
        record_identifier_value_as_string: str,
        event_time: str,
        deletion_mode: DeletionModeEnum = DeletionModeEnum.SOFT_DELETE,
    ):
        """Delete a single record from a FeatureGroup.

        Args:
            record_identifier_value_as_string (String):
                a String representing the value of the record identifier.
            event_time (String):
                a timestamp format String indicating when the deletion event occurred.
            deletion_mode (DeletionModeEnum):
                deletion mode for deleting record. (default: DetectionModeEnum.SOFT_DELETE)
        """

        return self.sagemaker_session.delete_record(
            feature_group_name=self.name,
            record_identifier_value_as_string=record_identifier_value_as_string,
            event_time=event_time,
            deletion_mode=deletion_mode.value,
        )

    def ingest(
        self,
        data_frame: DataFrame,
        max_workers: int = 1,
        max_processes: int = 1,
        wait: bool = True,
        timeout: Union[int, float] = None,
        profile_name: str = None,
    ) -> IngestionManagerPandas:
        """Ingest the content of a pandas DataFrame to feature store.

        ``max_worker`` the number of threads created to work on different partitions of the
        ``data_frame`` in parallel.

        ``max_processes`` the number of processes will be created to work on different
        partitions of the ``data_frame`` in parallel, each with ``max_worker`` threads.

        The ingest function attempts to ingest all records in the data frame. SageMaker
        Feature Store throws an exception if it fails to ingest any records.

        If ``wait`` is ``True``, Feature Store runs the ``ingest`` function synchronously.
        You receive an ``IngestionError`` if there are any records that can't be ingested.
        If ``wait`` is ``False``, Feature Store runs the ``ingest`` function asynchronously.

        Instead of setting ``wait`` to ``True`` in the ``ingest`` function, you can invoke
        the ``wait`` function on the returned instance of ``IngestionManagerPandas`` to run
        the ``ingest`` function synchronously.

        To access the rows that failed to ingest, set ``wait`` to ``False``. The
        ``IngestionError.failed_rows`` object saves all of the rows that failed to ingest.

        `profile_name` argument is an optional one. It will use the default credential if None is
        passed. This `profile_name` is used in the sagemaker_featurestore_runtime client only. See
        https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html for more
        about the default credential.

        Args:
            data_frame (DataFrame): data_frame to be ingested to feature store.
            max_workers (int): number of threads to be created.
            max_processes (int): number of processes to be created. Each process spawns
                ``max_worker`` number of threads.
            wait (bool): whether to wait for the ingestion to finish or not.
            timeout (Union[int, float]): ``concurrent.futures.TimeoutError`` will be raised
                if timeout is reached.
            profile_name (str): the profile credential should be used for ``PutRecord``
                (default: None).

        Returns:
            An instance of IngestionManagerPandas.
        """
        if max_processes <= 0:
            raise RuntimeError("max_processes must be greater than 0.")

        if max_workers <= 0:
            raise RuntimeError("max_workers must be greater than 0.")

        if profile_name is None and self.sagemaker_session.boto_session.profile_name != "default":
            profile_name = self.sagemaker_session.boto_session.profile_name

        manager = IngestionManagerPandas(
            feature_group_name=self.name,
            sagemaker_session=self.sagemaker_session,
            sagemaker_fs_runtime_client_config=self.sagemaker_session.sagemaker_featurestore_runtime_client.meta.config,
            max_workers=max_workers,
            max_processes=max_processes,
            profile_name=profile_name,
        )

        manager.run(data_frame=data_frame, wait=wait, timeout=timeout)

        return manager

    def athena_query(self) -> AthenaQuery:
        """Create an AthenaQuery instance.

        Returns:
            An instance of AthenaQuery initialized with data catalog configurations.
        """
        response = self.describe()
        data_catalog_config = response.get("OfflineStoreConfig").get("DataCatalogConfig", None)
        disable_glue = data_catalog_config.get("DisableGlueTableCreation", False)
        if data_catalog_config:
            query = AthenaQuery(
                catalog=data_catalog_config["Catalog"] if disable_glue else "AwsDataCatalog",
                database=data_catalog_config["Database"],
                table_name=data_catalog_config["TableName"],
                sagemaker_session=self.sagemaker_session,
            )
            return query
        raise RuntimeError("No metastore is configured with this feature group.")

    def as_hive_ddl(self, database: str = "sagemaker_featurestore", table_name: str = None) -> str:
        """Generate Hive DDL commands to define or change structure of tables or databases in Hive.

        Schema of the table is generated based on the feature definitions. Columns are named
        after feature name and data-type are inferred based on feature type. Integral feature
        type is mapped to INT data-type. Fractional feature type is mapped to FLOAT data-type.
        String feature type is mapped to STRING data-type.

        Args:
            database: name of the database. If not set "sagemaker_featurestore" will be used.
            table_name: name of the table. If not set the name of this feature group will be
                used.

        Returns:
            Generated create table DDL string.
        """
        if not table_name:
            table_name = self.name

        resolved_output_s3_uri = (
            self.describe()
            .get("OfflineStoreConfig")
            .get("S3StorageConfig")
            .get("ResolvedOutputS3Uri")
        )

        ddl = f"CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table_name} (\n"
        for definition in self.feature_definitions:
            ddl += (
                f"  {definition.feature_name} "
                f"{self._FEATURE_TYPE_TO_DDL_DATA_TYPE_MAP.get(definition.feature_type.value)}\n"
            )
        ddl += "  write_time TIMESTAMP\n"
        ddl += "  event_time TIMESTAMP\n"
        ddl += "  is_deleted BOOLEAN\n"
        ddl += ")\n"
        ddl += (
            "ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'\n"
            "  STORED AS\n"
            "  INPUTFORMAT 'parquet.hive.DeprecatedParquetInputFormat'\n"
            "  OUTPUTFORMAT 'parquet.hive.DeprecatedParquetOutputFormat'\n"
            f"LOCATION '{resolved_output_s3_uri}'"
        )
        return ddl
