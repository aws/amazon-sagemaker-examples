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
"""The input configs for FeatureStore.

A feature store serves as the single source of truth to store, retrieve,
remove, track, share, discover, and control access to features.

You can configure two types of feature stores, an online features store
and an offline feature store.

The online features store is a low latency, high availability cache for a
feature group that enables real-time lookup of records. Only the latest record is stored.

The offline feature store use when low (sub-second) latency reads are not needed.
This is the case when you want to store and serve features for exploration, model training,
and batch inference. The offline store uses your Amazon Simple Storage Service (Amazon S3)
bucket for storage. A prefixing scheme based on event time is used to store your data in Amazon S3.
"""
from __future__ import absolute_import

import abc
from typing import Dict, Any, List
from enum import Enum

import attr


class Config(abc.ABC):
    """Base config object for FeatureStore.

    Configs must implement the to_dict method.
    """

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Get the dictionary from attributes.

        Returns:
            dict contains the attributes.
        """

    @classmethod
    def construct_dict(cls, **kwargs) -> Dict[str, Any]:
        """Construct the dictionary based on the args.

        args:
            kwargs: args to be used to construct the dict.

        Returns:
            dict represents the given kwargs.
        """
        result = dict()
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, Config):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result


@attr.s
class OnlineStoreSecurityConfig(Config):
    """OnlineStoreSecurityConfig for FeatureStore.

    Attributes:
        kms_key_id (str): KMS key id.
    """

    kms_key_id: str = attr.ib(factory=str)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes."""
        return Config.construct_dict(KmsKeyId=self.kms_key_id)


@attr.s
class TtlDuration(Config):
    """TtlDuration for records in online FeatureStore.

    Attributes:
        unit (str): time unit.
        value (int): time value.
    """

    unit: str = attr.ib()
    value: int = attr.ib()

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            Unit=self.unit,
            Value=self.value,
        )


@attr.s
class OnlineStoreConfig(Config):
    """OnlineStoreConfig for FeatureStore.

    Attributes:
        enable_online_store (bool): whether to enable the online store.
        online_store_security_config (OnlineStoreSecurityConfig): configuration of security setting.
        ttl_duration (TtlDuration): Default time to live duration for records.
    """

    enable_online_store: bool = attr.ib(default=True)
    online_store_security_config: OnlineStoreSecurityConfig = attr.ib(default=None)
    ttl_duration: TtlDuration = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            EnableOnlineStore=self.enable_online_store,
            SecurityConfig=self.online_store_security_config,
            TtlDuration=self.ttl_duration,
        )


@attr.s
class OnlineStoreConfigUpdate(Config):
    """OnlineStoreConfigUpdate for FeatureStore.

    Attributes:
        ttl_duration (TtlDuration): Default time to live duration for records.
    """

    ttl_duration: TtlDuration = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            TtlDuration=self.ttl_duration,
        )


@attr.s
class S3StorageConfig(Config):
    """S3StorageConfig for FeatureStore.

    Attributes:
        s3_uri (str): S3 URI.
        kms_key_id (str): KMS key id.
    """

    s3_uri: str = attr.ib()
    kms_key_id: str = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            S3Uri=self.s3_uri,
            KmsKeyId=self.kms_key_id,
        )


@attr.s
class DataCatalogConfig(Config):
    """DataCatalogConfig for FeatureStore.

    Attributes:
        table_name (str): name of the table.
        catalog (str): name of the catalog.
        database (str): name of the database.
    """

    table_name: str = attr.ib(factory=str)
    catalog: str = attr.ib(factory=str)
    database: str = attr.ib(factory=str)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            TableName=self.table_name,
            Catalog=self.catalog,
            Database=self.database,
        )


class TableFormatEnum(Enum):
    """Enum of table formats.

    The offline store table formats can be Glue or Iceberg.
    """

    GLUE = "Glue"
    ICEBERG = "Iceberg"


@attr.s
class OfflineStoreConfig(Config):
    """OfflineStoreConfig for FeatureStore.

    Attributes:
        s3_storage_config (S3StorageConfig): configuration of S3 storage.
        disable_glue_table_creation (bool): whether to disable the Glue table creation.
        data_catalog_config (DataCatalogConfig): configuration of the data catalog.
        table_format (TableFormatEnum): format of the offline store table.
    """

    s3_storage_config: S3StorageConfig = attr.ib()
    disable_glue_table_creation: bool = attr.ib(default=False)
    data_catalog_config: DataCatalogConfig = attr.ib(default=None)
    table_format: TableFormatEnum = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            DisableGlueTableCreation=self.disable_glue_table_creation,
            S3StorageConfig=self.s3_storage_config,
            DataCatalogConfig=self.data_catalog_config,
            TableFormat=self.table_format.value if self.table_format else None,
        )


@attr.s
class FeatureValue(Config):
    """FeatureValue for FeatureStore.

    Attributes:
        feature_name (str): name of the Feature.
        value_as_string (str): value of the Feature in string form.
    """

    feature_name: str = attr.ib(default=None)
    value_as_string: str = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            FeatureName=self.feature_name,
            ValueAsString=self.value_as_string,
        )


@attr.s
class FeatureParameter(Config):
    """FeatureParameter for FeatureStore.

    Attributes:
        key (str): key of the parameter.
        value (str): value of the parameter.
    """

    key: str = attr.ib(default=None)
    value: str = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            Key=self.key,
            Value=self.value,
        )


class ResourceEnum(Enum):
    """Enum of resources.

    The data type of resource can be ``FeatureGroup`` or ``FeatureMetadata``.
    """

    def __str__(self):
        """Override str method to return enum value."""
        return str(self.value)

    FEATURE_GROUP = "FeatureGroup"
    FEATURE_METADATA = "FeatureMetadata"


class SearchOperatorEnum(Enum):
    """Enum of search operators.

    The data type of search operator can be ``And`` or ``Or``.
    """

    def __str__(self):
        """Override str method to return enum value."""
        return str(self.value)

    AND = "And"
    OR = "Or"


class SortOrderEnum(Enum):
    """Enum of sort orders.

    The data type of sort order can be ``Ascending`` or ``Descending``.
    """

    def __str__(self):
        """Override str method to return enum value."""
        return str(self.value)

    ASCENDING = "Ascending"
    DESCENDING = "Descending"


class FilterOperatorEnum(Enum):
    """Enum of filter operators.

    The data type of filter operator can be ``Equals``, ``NotEquals``, ``GreaterThan``,
    ``GreaterThanOrEqualTo``, ``LessThan``, ``LessThanOrEqualTo``, ``Contains``, ``Exists``,
    ``NotExists``, or ``In``.
    """

    def __str__(self):
        """Override str method to return enum value."""
        return str(self.value)

    EQUALS = "Equals"
    NOT_EQUALS = "NotEquals"
    GREATER_THAN = "GreaterThan"
    GREATER_THAN_OR_EQUAL_TO = "GreaterThanOrEqualTo"
    LESS_THAN = "LessThan"
    LESS_THAN_OR_EQUAL_TO = "LessThanOrEqualTo"
    CONTAINS = "Contains"
    EXISTS = "Exists"
    NOT_EXISTS = "NotExists"
    IN = "In"


@attr.s
class Filter(Config):
    """Filter for FeatureStore search.

    Attributes:
        name (str): A resource property name.
        value (str): A value used with ``Name`` and ``Operator`` to determine which resources
            satisfy the filter's condition.
        operator (FilterOperatorEnum): A Boolean binary operator that is used to evaluate the
        filter. If specify ``Value`` without ``Operator``, Amazon SageMaker uses ``Equals``
        (default: None).
    """

    name: str = attr.ib()
    value: str = attr.ib()
    operator: FilterOperatorEnum = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """
        return Config.construct_dict(
            Name=self.name,
            Value=self.value,
            Operator=None if not self.operator else str(self.operator),
        )


@attr.s
class Identifier(Config):
    """Identifier of batch get record API.

    Attributes:
        feature_group_name (str): name of a feature group.
        record_identifiers_value_as_string (List[str]): string value of record identifier.
        feature_names (List[str]): list of feature names (default: None).
    """

    feature_group_name: str = attr.ib()
    record_identifiers_value_as_string: List[str] = attr.ib()
    feature_names: List[str] = attr.ib(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on the attributes provided.

        Returns:
            dict represents the attributes.
        """

        return Config.construct_dict(
            FeatureGroupName=self.feature_group_name,
            RecordIdentifiersValueAsString=self.record_identifiers_value_as_string,
            FeatureNames=None if not self.feature_names else self.feature_names,
        )


class DeletionModeEnum(Enum):
    """Enum of deletion modes.

    The deletion mode for deleting records can be SoftDelete or HardDelete.
    """

    SOFT_DELETE = "SoftDelete"
    HARD_DELETE = "HardDelete"


class ExpirationTimeResponseEnum(Enum):
    """Enum of toggling the response of ExpiresAt.

    The ExpirationTimeResponse for toggling the response of ExpiresAt can be Disabled or Enabled.
    """

    DISABLED = "Disabled"
    ENABLED = "Enabled"
