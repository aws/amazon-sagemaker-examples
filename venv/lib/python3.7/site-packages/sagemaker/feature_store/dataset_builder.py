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
"""Dataset Builder

A Dataset Builder is a builder class for generating a dataset by providing conditions.
"""
from __future__ import absolute_import

import datetime
from enum import Enum
import os
from typing import Any, Dict, List, Tuple, Union

import attr
import pandas as pd

from sagemaker import Session, s3, utils
from sagemaker.feature_store.feature_group import FeatureDefinition, FeatureGroup, FeatureTypeEnum


_DEFAULT_CATALOG = "AwsDataCatalog"
_DEFAULT_DATABASE = "sagemaker_featurestore"


@attr.s
class TableType(Enum):
    """Enum of Table types.

    The data type of a table can be FeatureGroup or DataFrame.
    """

    FEATURE_GROUP = "FeatureGroup"
    DATA_FRAME = "DataFrame"


@attr.s
class JoinTypeEnum(Enum):
    """Enum of Join types.

    The Join type can be "INNER_JOIN", "LEFT_JOIN", "RIGHT_JOIN", "FULL_JOIN", or "CROSS_JOIN".
    """

    INNER_JOIN = "JOIN"
    LEFT_JOIN = "LEFT JOIN"
    RIGHT_JOIN = "RIGHT JOIN"
    FULL_JOIN = "FULL JOIN"
    CROSS_JOIN = "CROSS JOIN"


@attr.s
class JoinComparatorEnum(Enum):
    """Enum of Join comparators.

    The Join comparator can be "EQUALS", "GREATER_THAN", "LESS_THAN",
    "GREATER_THAN_OR_EQUAL_TO", "LESS_THAN_OR_EQUAL_TO" or "NOT_EQUAL_TO"
    """

    EQUALS = "="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL_TO = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL_TO = "<="
    NOT_EQUAL_TO = "<>"


@attr.s
class FeatureGroupToBeMerged:
    """FeatureGroup metadata which will be used for SQL join.

    This class instantiates a FeatureGroupToBeMerged object that comprises a list of feature names,
    a list of feature names which will be included in SQL query, a database, an Athena table name,
    a feature name of record identifier, a feature name of event time identifier and a feature name
    of base which is the target join key.

    Attributes:
        features (List[str]): A list of strings representing feature names of this FeatureGroup.
        included_feature_names (List[str]): A list of strings representing features to be
            included in the SQL join.
        projected_feature_names (List[str]): A list of strings representing features to be
            included for final projection in output.
        catalog (str): A string representing the catalog.
        database (str): A string representing the database.
        table_name (str): A string representing the Athena table name of this FeatureGroup.
        record_identifier_feature_name (str): A string representing the record identifier feature.
        event_time_identifier_feature (FeatureDefinition): A FeatureDefinition representing the
            event time identifier feature.
        target_feature_name_in_base (str): A string representing the feature name in base which will
            be used as target join key (default: None).
        table_type (TableType): A TableType representing the type of table if it is Feature Group or
            Panda Data Frame (default: None).
        feature_name_in_target (str): A string representing the feature name in the target feature
            group that will be compared to the target feature in the base feature group.
            If None is provided, the record identifier feature will be used in the
            SQL join. (default: None).
        join_comparator (JoinComparatorEnum): A JoinComparatorEnum representing the comparator
            used when joining the target feature in the base feature group and the feature
            in the target feature group. (default: JoinComparatorEnum.EQUALS).
        join_type (JoinTypeEnum): A JoinTypeEnum representing the type of join between
            the base and target feature groups. (default: JoinTypeEnum.INNER_JOIN).
    """

    features: List[str] = attr.ib()
    included_feature_names: List[str] = attr.ib()
    projected_feature_names: List[str] = attr.ib()
    catalog: str = attr.ib()
    database: str = attr.ib()
    table_name: str = attr.ib()
    record_identifier_feature_name: str = attr.ib()
    event_time_identifier_feature: FeatureDefinition = attr.ib()
    target_feature_name_in_base: str = attr.ib(default=None)
    table_type: TableType = attr.ib(default=None)
    feature_name_in_target: str = attr.ib(default=None)
    join_comparator: JoinComparatorEnum = attr.ib(default=JoinComparatorEnum.EQUALS)
    join_type: JoinTypeEnum = attr.ib(default=JoinTypeEnum.INNER_JOIN)


def construct_feature_group_to_be_merged(
    target_feature_group: FeatureGroup,
    included_feature_names: List[str],
    target_feature_name_in_base: str = None,
    feature_name_in_target: str = None,
    join_comparator: JoinComparatorEnum = JoinComparatorEnum.EQUALS,
    join_type: JoinTypeEnum = JoinTypeEnum.INNER_JOIN,
) -> FeatureGroupToBeMerged:
    """Construct a FeatureGroupToBeMerged object by provided parameters.

    Args:
        feature_group (FeatureGroup): A FeatureGroup object.
        included_feature_names (List[str]): A list of strings representing features to be
            included in the output.
        target_feature_name_in_base (str): A string representing the feature name in base which
            will be used as target join key (default: None).
        feature_name_in_target (str): A string representing the feature name in the target feature
            group that will be compared to the target feature in the base feature group.
            If None is provided, the record identifier feature will be used in the
            SQL join. (default: None).
        join_comparator (JoinComparatorEnum): A JoinComparatorEnum representing the comparator
            used when joining the target feature in the base feature group and the feature
            in the target feature group. (default: JoinComparatorEnum.EQUALS).
        join_type (JoinTypeEnum): A JoinTypeEnum representing the type of join between
            the base and target feature groups. (default: JoinTypeEnum.INNER_JOIN).
    Returns:
        A FeatureGroupToBeMerged object.

    Raises:
        ValueError: Invalid feature name(s) in included_feature_names.
    """
    feature_group_metadata = target_feature_group.describe()
    data_catalog_config = feature_group_metadata.get("OfflineStoreConfig", {}).get(
        "DataCatalogConfig", None
    )
    if not data_catalog_config:
        raise RuntimeError(
            f"No metastore is configured with FeatureGroup {target_feature_group.name}."
        )

    record_identifier_feature_name = feature_group_metadata.get("RecordIdentifierFeatureName", None)
    feature_definitions = feature_group_metadata.get("FeatureDefinitions", [])
    event_time_identifier_feature_name = feature_group_metadata.get("EventTimeFeatureName", None)
    event_time_identifier_feature_type = FeatureTypeEnum(
        next(
            filter(
                lambda f: f.get("FeatureName", None) == event_time_identifier_feature_name,
                feature_definitions,
            ),
            {},
        ).get("FeatureType", None)
    )
    table_name = data_catalog_config.get("TableName", None)
    database = data_catalog_config.get("Database", None)
    disable_glue = feature_group_metadata.get("DisableGlueTableCreation", False)
    catalog = data_catalog_config.get("Catalog", None) if disable_glue else _DEFAULT_CATALOG
    features = [feature.get("FeatureName", None) for feature in feature_definitions]

    if feature_name_in_target is not None and feature_name_in_target not in features:
        raise ValueError(
            f"Feature {feature_name_in_target} not found in FeatureGroup {target_feature_group.name}"
        )

    for included_feature in included_feature_names or []:
        if included_feature not in features:
            raise ValueError(
                f"Feature {included_feature} not found in FeatureGroup {target_feature_group.name}"
            )
    if not included_feature_names:
        included_feature_names = features
        projected_feature_names = features.copy()
    else:
        projected_feature_names = included_feature_names.copy()
        if record_identifier_feature_name not in included_feature_names:
            included_feature_names.append(record_identifier_feature_name)
        if event_time_identifier_feature_name not in included_feature_names:
            included_feature_names.append(event_time_identifier_feature_name)
    return FeatureGroupToBeMerged(
        features,
        included_feature_names,
        projected_feature_names,
        catalog,
        database,
        table_name,
        record_identifier_feature_name,
        FeatureDefinition(event_time_identifier_feature_name, event_time_identifier_feature_type),
        target_feature_name_in_base,
        TableType.FEATURE_GROUP,
        feature_name_in_target,
        join_comparator,
        join_type,
    )


@attr.s
class DatasetBuilder:
    """DatasetBuilder definition.

    This class instantiates a DatasetBuilder object that comprises a base, a list of feature names,
    an output path and a KMS key ID.

    Attributes:
        _sagemaker_session (Session): Session instance to perform boto calls.
        _base (Union[FeatureGroup, DataFrame]): A base which can be either a FeatureGroup or a
            pandas.DataFrame and will be used to merge other FeatureGroups and generate a Dataset.
        _output_path (str): An S3 URI which stores the output .csv file.
        _record_identifier_feature_name (str): A string representing the record identifier feature
            if base is a DataFrame (default: None).
        _event_time_identifier_feature_name (str): A string representing the event time identifier
            feature if base is a DataFrame (default: None).
        _included_feature_names (List[str]): A list of strings representing features to be
            included in the output. If not set, all features will be included in the output.
            (default: None).
        _kms_key_id (str): A KMS key id. If set, will be used to encrypt the result file
            (default: None).
        _point_in_time_accurate_join (bool): A boolean representing if point-in-time join
            is applied to the resulting dataframe when calling "to_dataframe".
            When set to True, users can retrieve data using “row-level time travel”
            according to the event times provided to the DatasetBuilder. This requires that the
            entity dataframe with event times is submitted as the base in the constructor
            (default: False).
        _include_duplicated_records (bool): A boolean representing whether the resulting dataframe
            when calling "to_dataframe" should include duplicated records (default: False).
        _include_deleted_records (bool): A boolean representing whether the resulting
            dataframe when calling "to_dataframe" should include deleted records (default: False).
        _number_of_recent_records (int): An integer representing how many records will be
            returned for each record identifier (default: 1).
        _number_of_records (int): An integer representing the number of records that should be
            returned in the resulting dataframe when calling "to_dataframe" (default: None).
        _write_time_ending_timestamp (datetime.datetime): A datetime that represents the latest
            write time for a record to be included in the resulting dataset. Records with a
            newer write time will be omitted from the resulting dataset. (default: None).
        _event_time_starting_timestamp (datetime.datetime): A datetime that represents the earliest
            event time for a record to be included in the resulting dataset. Records
            with an older event time will be omitted from the resulting dataset. (default: None).
        _event_time_ending_timestamp (datetime.datetime): A datetime that represents the latest
            event time for a record to be included in the resulting dataset. Records
            with a newer event time will be omitted from the resulting dataset. (default: None).
        _feature_groups_to_be_merged (List[FeatureGroupToBeMerged]): A list of
            FeatureGroupToBeMerged which will be joined to base (default: []).
        _event_time_identifier_feature_type (FeatureTypeEnum): A FeatureTypeEnum representing the
            type of event time identifier feature (default: None).
    """

    _sagemaker_session: Session = attr.ib()
    _base: Union[FeatureGroup, pd.DataFrame] = attr.ib()
    _output_path: str = attr.ib()
    _record_identifier_feature_name: str = attr.ib(default=None)
    _event_time_identifier_feature_name: str = attr.ib(default=None)
    _included_feature_names: List[str] = attr.ib(default=None)
    _kms_key_id: str = attr.ib(default=None)

    _point_in_time_accurate_join: bool = attr.ib(init=False, default=False)
    _include_duplicated_records: bool = attr.ib(init=False, default=False)
    _include_deleted_records: bool = attr.ib(init=False, default=False)
    _number_of_recent_records: int = attr.ib(init=False, default=None)
    _number_of_records: int = attr.ib(init=False, default=None)
    _write_time_ending_timestamp: datetime.datetime = attr.ib(init=False, default=None)
    _event_time_starting_timestamp: datetime.datetime = attr.ib(init=False, default=None)
    _event_time_ending_timestamp: datetime.datetime = attr.ib(init=False, default=None)
    _feature_groups_to_be_merged: List[FeatureGroupToBeMerged] = attr.ib(init=False, factory=list)
    _event_time_identifier_feature_type: FeatureTypeEnum = attr.ib(default=None)

    _DATAFRAME_TYPE_TO_COLUMN_TYPE_MAP = {
        "object": "STRING",
        "int64": "INT",
        "float64": "DOUBLE",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
    }

    def with_feature_group(
        self,
        feature_group: FeatureGroup,
        target_feature_name_in_base: str = None,
        included_feature_names: List[str] = None,
        feature_name_in_target: str = None,
        join_comparator: JoinComparatorEnum = JoinComparatorEnum.EQUALS,
        join_type: JoinTypeEnum = JoinTypeEnum.INNER_JOIN,
    ):
        """Join FeatureGroup with base.

        Args:
            feature_group (FeatureGroup): A target FeatureGroup which will be joined to base.
            target_feature_name_in_base (str): A string representing the feature name in base which
                will be used as a join key (default: None).
            included_feature_names (List[str]): A list of strings representing features to be
                included in the output (default: None).
            feature_name_in_target (str): A string representing the feature name in the target
                feature group that will be compared to the target feature in the base feature group.
                If None is provided, the record identifier feature will be used in the
                SQL join. (default: None).
            join_comparator (JoinComparatorEnum): A JoinComparatorEnum representing the comparator
                used when joining the target feature in the base feature group and the feature
                in the target feature group. (default: JoinComparatorEnum.EQUALS).
            join_type (JoinTypeEnum): A JoinTypeEnum representing the type of join between
                the base and target feature groups. (default: JoinTypeEnum.INNER_JOIN).
            Returns:
                This DatasetBuilder object.
        """
        self._feature_groups_to_be_merged.append(
            construct_feature_group_to_be_merged(
                feature_group,
                included_feature_names,
                target_feature_name_in_base,
                feature_name_in_target,
                join_comparator,
                join_type,
            )
        )
        return self

    def point_in_time_accurate_join(self):
        """Enable point-in-time accurate join.

        Returns:
            This DatasetBuilder object.
        """
        self._point_in_time_accurate_join = True
        return self

    def include_duplicated_records(self):
        """Include duplicated records in dataset.

        Returns:
            This DatasetBuilder object.
        """
        self._include_duplicated_records = True
        return self

    def include_deleted_records(self):
        """Include deleted records in dataset.

        Returns:
            This DatasetBuilder object.
        """
        self._include_deleted_records = True
        return self

    def with_number_of_recent_records_by_record_identifier(self, number_of_recent_records: int):
        """Set number_of_recent_records field with provided input.

        Args:
            number_of_recent_records (int): An int that how many recent records will be returned for
                each record identifier.
        Returns:
            This DatasetBuilder object.
        """
        self._number_of_recent_records = number_of_recent_records
        return self

    def with_number_of_records_from_query_results(self, number_of_records: int):
        """Set number_of_records field with provided input.

        Args:
            number_of_records (int): An int that how many records will be returned.
        Returns:
            This DatasetBuilder object.
        """
        self._number_of_records = number_of_records
        return self

    def as_of(self, timestamp: datetime.datetime):
        """Set write_time_ending_timestamp field with provided input.

        Args:
            timestamp (datetime.datetime): A datetime that all records' write time in dataset will
                be before it.
        Returns:
            This DatasetBuilder object.
        """
        self._write_time_ending_timestamp = timestamp
        return self

    def with_event_time_range(
        self,
        starting_timestamp: datetime.datetime = None,
        ending_timestamp: datetime.datetime = None,
    ):
        """Set event_time_starting_timestamp and event_time_ending_timestamp with provided inputs.

        Args:
            starting_timestamp (datetime.datetime): A datetime that all records' event time in
                dataset will be after it (default: None).
            ending_timestamp (datetime.datetime): A datetime that all records' event time in dataset
                will be before it (default: None).
        Returns:
            This DatasetBuilder object.
        """
        self._event_time_starting_timestamp = starting_timestamp
        self._event_time_ending_timestamp = ending_timestamp
        return self

    def to_csv_file(self) -> Tuple[str, str]:
        """Get query string and result in .csv format file

        Returns:
            The S3 path of the .csv file.
            The query string executed.
        """
        if isinstance(self._base, pd.DataFrame):
            temp_id = utils.unique_name_from_base("dataframe-base")
            local_file_name = f"{temp_id}.csv"
            desired_s3_folder = f"{self._output_path}/{temp_id}"
            self._base.to_csv(local_file_name, index=False, header=False)
            s3.S3Uploader.upload(
                local_path=local_file_name,
                desired_s3_uri=desired_s3_folder,
                sagemaker_session=self._sagemaker_session,
                kms_key=self._kms_key_id,
            )
            os.remove(local_file_name)
            temp_table_name = f'dataframe_{temp_id.replace("-", "_")}'
            self._create_temp_table(temp_table_name, desired_s3_folder)
            base_features = list(self._base.columns)
            event_time_identifier_feature_dtype = self._base[
                self._event_time_identifier_feature_name
            ].dtypes
            self._event_time_identifier_feature_type = (
                FeatureGroup.DTYPE_TO_FEATURE_DEFINITION_CLS_MAP.get(
                    str(event_time_identifier_feature_dtype), None
                )
            )
            query_string = self._construct_query_string(
                FeatureGroupToBeMerged(
                    base_features,
                    self._included_feature_names if self._included_feature_names else base_features,
                    self._included_feature_names if self._included_feature_names else base_features,
                    _DEFAULT_CATALOG,
                    _DEFAULT_DATABASE,
                    temp_table_name,
                    self._record_identifier_feature_name,
                    FeatureDefinition(
                        self._event_time_identifier_feature_name,
                        self._event_time_identifier_feature_type,
                    ),
                    None,
                    TableType.DATA_FRAME,
                )
            )
            query_result = self._run_query(query_string, _DEFAULT_CATALOG, _DEFAULT_DATABASE)
            # TODO: cleanup temp table, need more clarification, keep it for now
            return query_result.get("QueryExecution", {}).get("ResultConfiguration", {}).get(
                "OutputLocation", None
            ), query_result.get("QueryExecution", {}).get("Query", None)
        if isinstance(self._base, FeatureGroup):
            base_feature_group = construct_feature_group_to_be_merged(
                self._base, self._included_feature_names
            )
            self._record_identifier_feature_name = base_feature_group.record_identifier_feature_name
            self._event_time_identifier_feature_name = (
                base_feature_group.event_time_identifier_feature.feature_name
            )
            self._event_time_identifier_feature_type = (
                base_feature_group.event_time_identifier_feature.feature_type
            )
            query_string = self._construct_query_string(base_feature_group)
            query_result = self._run_query(
                query_string,
                base_feature_group.catalog,
                base_feature_group.database,
            )
            return query_result.get("QueryExecution", {}).get("ResultConfiguration", {}).get(
                "OutputLocation", None
            ), query_result.get("QueryExecution", {}).get("Query", None)
        raise ValueError("Base must be either a FeatureGroup or a DataFrame.")

    def to_dataframe(self) -> Tuple[pd.DataFrame, str]:
        """Get query string and result in pandas.Dataframe

        Returns:
            The pandas.DataFrame object.
            The query string executed.
        """
        csv_file, query_string = self.to_csv_file()
        s3.S3Downloader.download(
            s3_uri=csv_file,
            local_path="./",
            kms_key=self._kms_key_id,
            sagemaker_session=self._sagemaker_session,
        )
        local_file_name = csv_file.split("/")[-1]
        df = pd.read_csv(local_file_name)
        os.remove(local_file_name)

        local_metadata_file_name = local_file_name + ".metadata"
        if os.path.exists(local_metadata_file_name):
            os.remove(local_file_name + ".metadata")

        if "row_recent" in df:
            df = df.drop("row_recent", axis="columns")
        return df, query_string

    def _construct_event_time_conditions(
        self,
        table_name: str,
        event_time_identifier_feature: FeatureDefinition,
    ) -> List[str]:
        """Internal method for constructing event time range sql range as string.

        Args:
            table_name (str): name of the table.
            event_time_identifier_feature (FeatureDefinition): A FeatureDefinition representing the
                event time identifier feature.
        Returns:
            The list of query strings.
        """
        event_time_conditions = []
        timestamp_cast_function_name = "from_unixtime"
        if event_time_identifier_feature.feature_type == FeatureTypeEnum.STRING:
            timestamp_cast_function_name = "from_iso8601_timestamp"
        if self._event_time_starting_timestamp:
            event_time_conditions.append(
                f"{timestamp_cast_function_name}({table_name}."
                + f'"{event_time_identifier_feature.feature_name}") >= '
                + f"from_unixtime({self._event_time_starting_timestamp.timestamp()})"
            )
        if self._event_time_ending_timestamp:
            event_time_conditions.append(
                f"{timestamp_cast_function_name}({table_name}."
                + f'"{event_time_identifier_feature.feature_name}") <= '
                + f"from_unixtime({self._event_time_ending_timestamp.timestamp()})"
            )
        return event_time_conditions

    def _construct_write_time_condition(
        self,
        table_name: str,
    ) -> str:
        """Internal method for constructing write time condition.

        Args:
            table_name (str): name of the table.
        Returns:
            string of write time condition.
        """
        write_time_condition = (
            f'{table_name}."write_time" <= '
            f"to_timestamp('{self._write_time_ending_timestamp.replace(microsecond=0)}', "
            f"'yyyy-mm-dd hh24:mi:ss')"
        )
        return write_time_condition

    def _construct_where_query_string(
        self,
        suffix: str,
        event_time_identifier_feature: FeatureDefinition,
        where_conditions: List[str],
    ) -> str:
        """Internal method for constructing SQL WHERE query string by parameters.

        Args:
            suffix (str): A temp identifier of the FeatureGroup.
            event_time_identifier_feature (FeatureDefinition): A FeatureDefinition representing the
                event time identifier feature.
            where_conditions (List[str]): A list of strings representing existing where clauses.
        Returns:
            The WHERE query string.

        Raises:
            ValueError: FeatureGroup not provided while using as_of(). Only found pandas.DataFrame.
        """
        if self._number_of_recent_records:
            if self._number_of_recent_records < 0:
                raise ValueError(
                    "Please provide non-negative integer for number_of_recent_records."
                )
        if self._number_of_records:
            if self._number_of_records < 0:
                raise ValueError("Please provide non-negative integer for number_of_records.")
        if self._include_deleted_records:
            if isinstance(self._base, pd.DataFrame):
                if len(self._feature_groups_to_be_merged) == 0:
                    raise ValueError(
                        "include_deleted_records() only works for FeatureGroup,"
                        " if there is no join operation."
                    )
        if self._include_duplicated_records:
            if isinstance(self._base, pd.DataFrame):
                if len(self._feature_groups_to_be_merged) == 0:
                    raise ValueError(
                        "include_duplicated_records() only works for FeatureGroup,"
                        " if there is no join operation."
                    )
        if self._point_in_time_accurate_join:
            if len(self._feature_groups_to_be_merged) == 0:
                raise ValueError(
                    "point_in_time_accurate_join() this operation only works when there is "
                    "more than one feature group to join."
                )
        if self._write_time_ending_timestamp:
            if isinstance(self._base, pd.DataFrame):
                if len(self._feature_groups_to_be_merged) == 0:
                    raise ValueError(
                        "as_of() only works for FeatureGroup," " if there is no join operation."
                    )
            if isinstance(self._base, FeatureGroup):
                if self._write_time_ending_timestamp:
                    where_conditions.append(self._construct_write_time_condition(f"table_{suffix}"))

        event_time_conditions = self._construct_event_time_conditions(
            f"table_{suffix}", event_time_identifier_feature
        )
        where_conditions.extend(event_time_conditions)

        if len(where_conditions) == 0:
            return ""
        return "WHERE " + "\nAND ".join(where_conditions)

    def _construct_dedup_query(self, feature_group: FeatureGroupToBeMerged, suffix: str) -> str:
        """Internal method for constructing removing duplicate records SQL query string.

        Args:
            feature_group (FeatureGroupToBeMerged): A FeatureGroupToBeMerged object which has the
                FeatureGroup metadata.
            suffix (str): A temp identifier of the FeatureGroup.
        Returns:
            The SQL query string.
        """
        record_feature_name = feature_group.record_identifier_feature_name
        event_time_identifier_feature = feature_group.event_time_identifier_feature
        event_time_feature_name = feature_group.event_time_identifier_feature.feature_name
        rank_query_string = ""
        where_conditions = []
        where_conditions_str = ""
        is_dedup_enabled = False

        if feature_group.table_type is TableType.FEATURE_GROUP:
            is_dedup_enabled = True
            rank_query_string = (
                f'ORDER BY origin_{suffix}."api_invocation_time" DESC, '
                + f'origin_{suffix}."write_time" DESC\n'
            )

            if self._write_time_ending_timestamp:
                where_conditions.append(self._construct_write_time_condition(f"origin_{suffix}"))

        event_time_conditions = self._construct_event_time_conditions(
            f"origin_{suffix}", event_time_identifier_feature
        )
        where_conditions.extend(event_time_conditions)

        if len(where_conditions) != 0:
            where_conditions_str = "WHERE " + "\nAND ".join(where_conditions) + "\n"

        dedup_where_clause = f"WHERE dedup_row_{suffix} = 1\n" if is_dedup_enabled else ""
        return (
            f"table_{suffix} AS (\n"
            + "SELECT *\n"
            + "FROM (\n"
            + "SELECT *, row_number() OVER (\n"
            + f'PARTITION BY origin_{suffix}."{record_feature_name}", '
            + f'origin_{suffix}."{event_time_feature_name}"\n'
            + rank_query_string
            + f") AS dedup_row_{suffix}\n"
            + f'FROM "{feature_group.database}"."{feature_group.table_name}" origin_{suffix}\n'
            + where_conditions_str
            + ")\n"
            + dedup_where_clause
            + ")"
        )

    def _construct_deleted_query(self, feature_group: FeatureGroupToBeMerged, suffix: str) -> str:
        """Internal method for constructing removing deleted records SQL query string.

        Args:
            feature_group (FeatureGroupToBeMerged): A FeatureGroupToBeMerged object which has the
                FeatureGroup metadata.
            suffix (str): A temp identifier of the FeatureGroup.
        Returns:
            The SQL query string.
        """
        record_feature_name = feature_group.record_identifier_feature_name
        event_time_identifier_feature = feature_group.event_time_identifier_feature
        event_time_feature_name = feature_group.event_time_identifier_feature.feature_name
        rank_query_string = f'ORDER BY origin_{suffix}."{event_time_feature_name}" DESC'
        write_time_condition = "\n"
        event_time_starting_condition = ""
        event_time_ending_condition = ""

        if feature_group.table_type is TableType.FEATURE_GROUP:
            rank_query_string += (
                f', origin_{suffix}."api_invocation_time" DESC, '
                + f'origin_{suffix}."write_time" DESC\n'
            )

            if self._write_time_ending_timestamp:
                write_time_condition += " AND "
                write_time_condition += self._construct_write_time_condition(f"origin_{suffix}")
                write_time_condition += "\n"

        if self._event_time_starting_timestamp and self._event_time_ending_timestamp:
            event_time_conditions = self._construct_event_time_conditions(
                f"origin_{suffix}", event_time_identifier_feature
            )
            event_time_starting_condition = "AND " + event_time_conditions[0] + "\n"
            event_time_ending_condition = "AND " + event_time_conditions[1] + "\n"

        return (
            f"deleted_{suffix} AS (\n"
            + "SELECT *\n"
            + "FROM (\n"
            + "SELECT *, row_number() OVER (\n"
            + f'PARTITION BY origin_{suffix}."{record_feature_name}"\n'
            + rank_query_string
            + f") AS deleted_row_{suffix}\n"
            + f'FROM "{feature_group.database}"."{feature_group.table_name}" origin_{suffix}\n'
            + "WHERE is_deleted"
            + write_time_condition
            + event_time_starting_condition
            + event_time_ending_condition
            + ")\n"
            + f"WHERE deleted_row_{suffix} = 1\n"
            + ")"
        )

    def _construct_table_included_features(
        self, feature_group: FeatureGroupToBeMerged, suffix: str
    ) -> str:
        """Internal method for constructing included features string of table.

        Args:
            feature_group (FeatureGroupToBeMerged): A FeatureGroupToBeMerged object
                which has the metadata.
            suffix (str): A temp identifier of the table.
        Returns:
            The string that includes all feature to be included of table.
        """

        included_features = ", ".join(
            [
                f'table_{suffix}."{include_feature_name}"'
                for include_feature_name in feature_group.included_feature_names
            ]
        )
        return included_features

    def _construct_table_query(self, feature_group: FeatureGroupToBeMerged, suffix: str) -> str:
        """Internal method for constructing SQL query string by parameters.

        Args:
            feature_group (FeatureGroupToBeMerged): A FeatureGroupToBeMerged object which has the
                FeatureGroup metadata.
            suffix (str): A temp identifier of the FeatureGroup.
        Returns:
            The query string.
        """
        included_features = self._construct_table_included_features(feature_group, suffix)

        # If base is a FeatureGroup then included_features_write_time will have a write_time column
        # Or included_features_write_time is same as included_features
        included_features_write_time = included_features

        if feature_group.table_type is TableType.FEATURE_GROUP:
            included_features_write_time += f', table_{suffix}."write_time"'
        record_feature_name = feature_group.record_identifier_feature_name
        event_time_feature_name = feature_group.event_time_identifier_feature.feature_name
        if self._include_duplicated_records and self._include_deleted_records:
            return (
                f"SELECT {included_features}\n"
                + f'FROM "{feature_group.database}"."{feature_group.table_name}" table_{suffix}\n'
                + self._construct_where_query_string(
                    suffix, feature_group.event_time_identifier_feature, ["NOT is_deleted"]
                )
            )
        if feature_group.table_type is TableType.FEATURE_GROUP and self._include_deleted_records:
            rank_query_string = ""
            if feature_group.table_type is TableType.FEATURE_GROUP:
                rank_query_string = (
                    f'ORDER BY origin_{suffix}."api_invocation_time" DESC, '
                    + f'origin_{suffix}."write_time" DESC\n'
                )
            return (
                f"SELECT {included_features}\n"
                + "FROM (\n"
                + "SELECT *, row_number() OVER (\n"
                + f'PARTITION BY origin_{suffix}."{record_feature_name}", '
                + f'origin_{suffix}."{event_time_feature_name}"\n'
                + rank_query_string
                + f") AS row_{suffix}\n"
                + f'FROM "{feature_group.database}"."{feature_group.table_name}" origin_{suffix}\n'
                + "WHERE NOT is_deleted"
                + f") AS table_{suffix}\n"
                + self._construct_where_query_string(
                    suffix,
                    feature_group.event_time_identifier_feature,
                    [f"row_{suffix} = 1"],
                )
            )
        rank_query_string = ""
        if feature_group.table_type is TableType.FEATURE_GROUP:
            rank_query_string = (
                f'OR (table_{suffix}."{event_time_feature_name}" = '
                + f'deleted_{suffix}."{event_time_feature_name}" '
                + f'AND table_{suffix}."api_invocation_time" > '
                + f'deleted_{suffix}."api_invocation_time")\n'
                + f'OR (table_{suffix}."{event_time_feature_name}" = '
                + f'deleted_{suffix}."{event_time_feature_name}" '
                + f'AND table_{suffix}."api_invocation_time" = '
                + f'deleted_{suffix}."api_invocation_time" '
                + f'AND table_{suffix}."write_time" > deleted_{suffix}."write_time")\n'
            )

        final_query_string = ""
        if feature_group.table_type is TableType.FEATURE_GROUP:
            if self._include_duplicated_records:
                final_query_string = (
                    f"WITH {self._construct_deleted_query(feature_group, suffix)}\n"
                    + f"SELECT {included_features}\n"
                    + "FROM (\n"
                    + f"SELECT {included_features_write_time}\n"
                    + f'FROM "{feature_group.database}"."{feature_group.table_name}"'
                    + f" table_{suffix}\n"
                    + f"LEFT JOIN deleted_{suffix}\n"
                    + f'ON table_{suffix}."{record_feature_name}" = '
                    + f'deleted_{suffix}."{record_feature_name}"\n'
                    + f'WHERE deleted_{suffix}."{record_feature_name}" IS NULL\n'
                    + "UNION ALL\n"
                    + f"SELECT {included_features_write_time}\n"
                    + f"FROM deleted_{suffix}\n"
                    + f'JOIN "{feature_group.database}"."{feature_group.table_name}"'
                    + f" table_{suffix}\n"
                    + f'ON table_{suffix}."{record_feature_name}" = '
                    + f'deleted_{suffix}."{record_feature_name}"\n'
                    + "AND (\n"
                    + f'table_{suffix}."{event_time_feature_name}" > '
                    + f'deleted_{suffix}."{event_time_feature_name}"\n'
                    + rank_query_string
                    + ")\n"
                    + f") AS table_{suffix}\n"
                    + self._construct_where_query_string(
                        suffix, feature_group.event_time_identifier_feature, []
                    )
                )
            else:
                final_query_string = (
                    f"WITH {self._construct_dedup_query(feature_group, suffix)},\n"
                    + f"{self._construct_deleted_query(feature_group, suffix)}\n"
                    + f"SELECT {included_features}\n"
                    + "FROM (\n"
                    + f"SELECT {included_features_write_time}\n"
                    + f"FROM table_{suffix}\n"
                    + f"LEFT JOIN deleted_{suffix}\n"
                    + f'ON table_{suffix}."{record_feature_name}" = '
                    + f'deleted_{suffix}."{record_feature_name}"\n'
                    + f'WHERE deleted_{suffix}."{record_feature_name}" IS NULL\n'
                    + "UNION ALL\n"
                    + f"SELECT {included_features_write_time}\n"
                    + f"FROM deleted_{suffix}\n"
                    + f"JOIN table_{suffix}\n"
                    + f'ON table_{suffix}."{record_feature_name}" = '
                    + f'deleted_{suffix}."{record_feature_name}"\n'
                    + "AND (\n"
                    + f'table_{suffix}."{event_time_feature_name}" > '
                    + f'deleted_{suffix}."{event_time_feature_name}"\n'
                    + rank_query_string
                    + ")\n"
                    + f") AS table_{suffix}\n"
                    + self._construct_where_query_string(
                        suffix, feature_group.event_time_identifier_feature, []
                    )
                )
        else:
            final_query_string = (
                f"WITH {self._construct_dedup_query(feature_group, suffix)}\n"
                + f"SELECT {included_features}\n"
                + "FROM (\n"
                + f"SELECT {included_features_write_time}\n"
                + f"FROM table_{suffix}\n"
                + f") AS table_{suffix}\n"
                + self._construct_where_query_string(
                    suffix, feature_group.event_time_identifier_feature, []
                )
            )
        return final_query_string

    def _construct_query_string(self, base: FeatureGroupToBeMerged) -> str:
        """Internal method for constructing SQL query string by parameters.

        Args:
            base (FeatureGroupToBeMerged): A FeatureGroupToBeMerged object which has the metadata.
        Returns:
            The query string.

        Raises:
            ValueError: target_feature_name_in_base is an invalid feature name.
        """
        base_table_query_string = self._construct_table_query(base, "base")
        query_string = f"WITH fg_base AS ({base_table_query_string})"
        if len(self._feature_groups_to_be_merged) > 0:
            with_subquery_string = "".join(
                [
                    f",\nfg_{i} AS ({self._construct_table_query(feature_group, str(i))})"
                    for i, feature_group in enumerate(self._feature_groups_to_be_merged)
                ]
            )
            query_string += with_subquery_string

        selected_features = ""
        selected_features += ", ".join(map("fg_base.{0}".format, base.projected_feature_names))
        if len(self._feature_groups_to_be_merged) > 0:
            for i, feature_group in enumerate(self._feature_groups_to_be_merged):
                selected_features += ", "
                selected_features += ", ".join(
                    [
                        f'fg_{i}."{feature_name}" as "{feature_name}.{(i+1)}"'
                        for feature_name in feature_group.projected_feature_names
                    ]
                )

        selected_features_final = ""
        selected_features_final += ", ".join(base.projected_feature_names)
        if len(self._feature_groups_to_be_merged) > 0:
            for i, feature_group in enumerate(self._feature_groups_to_be_merged):
                selected_features_final += ", "
                selected_features_final += ", ".join(
                    [
                        '"{0}.{1}"'.format(feature_name, (i + 1))
                        for feature_name in feature_group.projected_feature_names
                    ]
                )

        query_string += (
            f"\nSELECT {selected_features_final}\n"
            + "FROM (\n"
            + f"SELECT {selected_features}, row_number() OVER (\n"
            + f'PARTITION BY fg_base."{base.record_identifier_feature_name}"\n'
            + f'ORDER BY fg_base."{base.event_time_identifier_feature.feature_name}" DESC'
        )

        recent_record_where_clause = ""
        if self._number_of_recent_records is not None and self._number_of_recent_records >= 0:
            recent_record_where_clause = f"WHERE row_recent <= {self._number_of_recent_records}"

        join_subquery_strings = []
        if len(self._feature_groups_to_be_merged) > 0:
            for i, feature_group in enumerate(self._feature_groups_to_be_merged):
                if not feature_group.target_feature_name_in_base:
                    feature_group.target_feature_name_in_base = self._record_identifier_feature_name
                else:
                    if feature_group.target_feature_name_in_base not in base.features:
                        raise ValueError(
                            f"Feature {feature_group.target_feature_name_in_base} not found in base"
                        )
                query_string += (
                    f', fg_{i}."{feature_group.event_time_identifier_feature.feature_name}" DESC'
                )
                join_subquery_strings.append(self._construct_join_condition(feature_group, str(i)))

        query_string += (
            "\n) AS row_recent\n"
            + "FROM fg_base"
            + "".join(join_subquery_strings)
            + "\n)\n"
            + f"{recent_record_where_clause}"
        )

        if self._number_of_records is not None and self._number_of_records >= 0:
            query_string += f"\nLIMIT {self._number_of_records}"
        return query_string

    def _construct_join_condition(self, feature_group: FeatureGroupToBeMerged, suffix: str) -> str:
        """Internal method for constructing SQL JOIN query string by parameters.

        Args:
            feature_group (FeatureGroupToBeMerged): A FeatureGroupToBeMerged object which has the
                FeatureGroup metadata.
            suffix (str): A temp identifier of the FeatureGroup.
        Returns:
            The JOIN query string.
        """

        feature_name_in_target = (
            feature_group.feature_name_in_target
            if feature_group.feature_name_in_target is not None
            else feature_group.record_identifier_feature_name
        )

        join_condition_string = (
            f"\n{feature_group.join_type.value} fg_{suffix}\n"
            + f'ON fg_base."{feature_group.target_feature_name_in_base}"'
            + f" {feature_group.join_comparator.value} "
            + f'fg_{suffix}."{feature_name_in_target}"'
        )
        base_timestamp_cast_function_name = "from_unixtime"
        if self._event_time_identifier_feature_type == FeatureTypeEnum.STRING:
            base_timestamp_cast_function_name = "from_iso8601_timestamp"
        timestamp_cast_function_name = "from_unixtime"
        if feature_group.event_time_identifier_feature.feature_type == FeatureTypeEnum.STRING:
            timestamp_cast_function_name = "from_iso8601_timestamp"
        if self._point_in_time_accurate_join:
            join_condition_string += (
                f"\nAND {base_timestamp_cast_function_name}(fg_base."
                + f'"{self._event_time_identifier_feature_name}") >= '
                + f"{timestamp_cast_function_name}(fg_{suffix}."
                + f'"{feature_group.event_time_identifier_feature.feature_name}")'
            )
        return join_condition_string

    def _create_temp_table(self, temp_table_name: str, desired_s3_folder: str):
        """Internal method for creating a temp Athena table for the base pandas.Dataframe.

        Args:
            temp_table_name (str): The Athena table name of base pandas.DataFrame.
            desired_s3_folder (str): The S3 URI of the folder of the data.
        """
        columns_string = ", ".join(
            [self._construct_athena_table_column_string(column) for column in self._base.columns]
        )
        serde_properties = '"separatorChar" = ",", "quoteChar" = "`", "escapeChar" = "\\\\"'
        query_string = (
            f"CREATE EXTERNAL TABLE {temp_table_name} ({columns_string}) "
            + "ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' "
            + f"WITH SERDEPROPERTIES ({serde_properties}) "
            + f"LOCATION '{desired_s3_folder}';"
        )
        self._run_query(query_string, _DEFAULT_CATALOG, _DEFAULT_DATABASE)

    def _construct_athena_table_column_string(self, column: str) -> str:
        """Internal method for constructing string of Athena column.

        Args:
            column (str): The column name from pandas.Dataframe.
        Returns:
            The Athena column string.

        Raises:
            RuntimeError: The type of pandas.Dataframe column is not support yet.
        """
        dataframe_type = self._base[column].dtypes
        if str(dataframe_type) not in self._DATAFRAME_TYPE_TO_COLUMN_TYPE_MAP.keys():
            raise RuntimeError(f"The dataframe type {dataframe_type} is not supported yet.")
        return f"{column} {self._DATAFRAME_TYPE_TO_COLUMN_TYPE_MAP.get(str(dataframe_type), None)}"

    def _run_query(self, query_string: str, catalog: str, database: str) -> Dict[str, Any]:
        """Internal method for execute Athena query, wait for query finish and get query result.

        Args:
            query_string (str): The SQL query statements to be executed.
            catalog (str): The name of the data catalog used in the query execution.
            database (str): The name of the database used in the query execution.
        Returns:
            The query result.

        Raises:
            RuntimeError: Athena query failed.
        """
        query = self._sagemaker_session.start_query_execution(
            catalog=catalog,
            database=database,
            query_string=query_string,
            output_location=self._output_path,
            kms_key=self._kms_key_id,
        )
        query_id = query.get("QueryExecutionId", None)
        self._sagemaker_session.wait_for_athena_query(query_execution_id=query_id)
        query_result = self._sagemaker_session.get_query_execution(query_execution_id=query_id)
        query_state = query_result.get("QueryExecution", {}).get("Status", {}).get("State", None)

        if query_state != "SUCCEEDED":
            raise RuntimeError(f"Failed to execute query {query_id}.")
        return query_result
