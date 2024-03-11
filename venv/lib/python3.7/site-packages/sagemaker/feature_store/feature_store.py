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
"""Feature Store.

Amazon SageMaker Feature Store is a fully managed, purpose-built repository to store, share, and
manage features for machine learning (ML) models.
"""
from __future__ import absolute_import

import datetime
from typing import Any, Dict, Sequence, Union

import attr
import pandas as pd

from sagemaker import Session
from sagemaker.feature_store.dataset_builder import DatasetBuilder
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.inputs import (
    Filter,
    ResourceEnum,
    SearchOperatorEnum,
    SortOrderEnum,
    Identifier,
)


@attr.s
class FeatureStore:
    """FeatureStore definition.

    This class instantiates a FeatureStore object that comprises a SageMaker session instance.

    Attributes:
        sagemaker_session (Session): session instance to perform boto calls.
    """

    sagemaker_session: Session = attr.ib(default=Session)

    def create_dataset(
        self,
        base: Union[FeatureGroup, pd.DataFrame],
        output_path: str,
        record_identifier_feature_name: str = None,
        event_time_identifier_feature_name: str = None,
        included_feature_names: Sequence[str] = None,
        kms_key_id: str = None,
    ) -> DatasetBuilder:
        """Create a Dataset Builder for generating a Dataset.

        Args:
            base (Union[FeatureGroup, DataFrame]): A base which can be either a FeatureGroup or a
                pandas.DataFrame and will be used to merge other FeatureGroups and generate a
                Dataset.
            output_path (str): An S3 URI which stores the output .csv file.
            record_identifier_feature_name (str): A string representing the record identifier
                feature if base is a DataFrame (default: None).
            event_time_identifier_feature_name (str): A string representing the event time
                identifier feature if base is a DataFrame (default: None).
            included_feature_names (List[str]): A list of features to be included in the output
                (default: None).
            kms_key_id (str): An KMS key id. If set, will be used to encrypt the result file
                (default: None).

        Raises:
            ValueError: Base is a Pandas DataFrame but no record identifier feature name nor event
                time identifier feature name is provided.
        """
        if isinstance(base, pd.DataFrame):
            if record_identifier_feature_name is None or event_time_identifier_feature_name is None:
                raise ValueError(
                    "You must provide a record identifier feature name and an event time "
                    + "identifier feature name if specify DataFrame as base."
                )
        return DatasetBuilder(
            self.sagemaker_session,
            base,
            output_path,
            record_identifier_feature_name,
            event_time_identifier_feature_name,
            included_feature_names,
            kms_key_id,
        )

    def list_feature_groups(
        self,
        name_contains: str = None,
        feature_group_status_equals: str = None,
        offline_store_status_equals: str = None,
        creation_time_after: datetime.datetime = None,
        creation_time_before: datetime.datetime = None,
        sort_order: str = None,
        sort_by: str = None,
        max_results: int = None,
        next_token: str = None,
    ) -> Dict[str, Any]:
        """List all FeatureGroups satisfying given filters.

        Args:
            name_contains (str): A string that partially matches one or more FeatureGroups' names.
                Filters FeatureGroups by name.
            feature_group_status_equals (str): A FeatureGroup status.
                Filters FeatureGroups by FeatureGroup status.
            offline_store_status_equals (str): An OfflineStore status.
                Filters FeatureGroups by OfflineStore status.
            creation_time_after (datetime.datetime): Use this parameter to search for FeatureGroups
                created after a specific date and time.
            creation_time_before (datetime.datetime): Use this parameter to search for FeatureGroups
                created before a specific date and time.
            sort_order (str): The order in which FeatureGroups are listed.
            sort_by (str): The value on which the FeatureGroup list is sorted.
            max_results (int): The maximum number of results returned by ListFeatureGroups.
            next_token (str): A token to resume pagination of ListFeatureGroups results.

        Returns:
            Response dict from service.
        """
        return self.sagemaker_session.list_feature_groups(
            name_contains=name_contains,
            feature_group_status_equals=feature_group_status_equals,
            offline_store_status_equals=offline_store_status_equals,
            creation_time_after=creation_time_after,
            creation_time_before=creation_time_before,
            sort_order=sort_order,
            sort_by=sort_by,
            max_results=max_results,
            next_token=next_token,
        )

    def batch_get_record(
        self,
        identifiers: Sequence[Identifier],
        expiration_time_response: str = None,
    ) -> Dict[str, Any]:
        """Get record in batch from FeatureStore

        Args:
            identifiers (Sequence[Identifier]): A list of identifiers to uniquely identify records
                in FeatureStore.
            expiration_time_response (str): the field of expiration time response
                to toggle returning of expiresAt.

        Returns:
            Response dict from service.
        """
        batch_get_record_identifiers = [identifier.to_dict() for identifier in identifiers]
        return self.sagemaker_session.batch_get_record(
            identifiers=batch_get_record_identifiers,
            expiration_time_response=expiration_time_response,
        )

    def search(
        self,
        resource: ResourceEnum,
        filters: Sequence[Filter] = None,
        operator: SearchOperatorEnum = None,
        sort_by: str = None,
        sort_order: SortOrderEnum = None,
        next_token: str = None,
        max_results: int = None,
    ) -> Dict[str, Any]:
        """Search for FeatureGroups or FeatureMetadata satisfying given filters.

        Args:
            resource (ResourceEnum): The name of the Amazon SageMaker resource to search for.
                Valid values are ``FeatureGroup`` or ``FeatureMetadata``.
            filters (Sequence[Filter]): A list of filter objects (Default: None).
            operator (SearchOperatorEnum): A Boolean operator used to evaluate the filters.
                Valid values are ``And`` or ``Or``. The default is ``And`` (Default: None).
            sort_by (str): The name of the resource property used to sort the ``SearchResults``.
                The default is ``LastModifiedTime``.
            sort_order (SortOrderEnum): How ``SearchResults`` are ordered.
                Valid values are ``Ascending`` or ``Descending``. The default is ``Descending``.
            next_token (str): If more than ``MaxResults`` resources match the specified
                filters, the response includes a ``NextToken``. The ``NextToken`` can be passed to
                the next ``SearchRequest`` to continue retrieving results (Default: None).
            max_results (int): The maximum number of results to return (Default: None).

        Returns:
            Response dict from service.
        """
        search_expression = {}
        if filters:
            search_expression["Filters"] = [filter.to_dict() for filter in filters]
        if operator:
            search_expression["Operator"] = str(operator)

        return self.sagemaker_session.search(
            resource=str(resource),
            search_expression=search_expression,
            sort_by=sort_by,
            sort_order=None if not sort_order else str(sort_order),
            next_token=next_token,
            max_results=max_results,
        )
