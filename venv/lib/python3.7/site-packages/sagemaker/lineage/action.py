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
"""This module contains code to create and manage SageMaker ``Actions``."""
from __future__ import absolute_import

from typing import Optional, Iterator, List
from datetime import datetime

from sagemaker.session import Session
from sagemaker.apiutils import _base_types
from sagemaker.lineage import _api_types, _utils
from sagemaker.lineage._api_types import ActionSource, ActionSummary
from sagemaker.lineage.artifact import Artifact

from sagemaker.lineage.query import (
    LineageQuery,
    LineageFilter,
    LineageSourceEnum,
    LineageEntityEnum,
    LineageQueryDirectionEnum,
)


class Action(_base_types.Record):
    """An Amazon SageMaker action, which is part of a SageMaker lineage.

    Examples:
        .. code-block:: python

            from sagemaker.lineage import action

            my_action = action.Action.create(
                action_name='MyAction',
                action_type='EndpointDeployment',
                source_uri='s3://...')

            my_action.properties["added"] = "property"
            my_action.save()

            for actn in action.Action.list():
                print(actn)

            my_action.delete()

    Attributes:
        action_arn (str): The ARN of the action.
        action_name (str): The name of the action.
        action_type (str): The type of the action.
        description (str): A description of the action.
        status (str): The status of the action.
        source (obj): The source of the action with a URI and type.
        properties (dict): Dictionary of properties.
        tags (List[dict[str, str]]): A list of tags to associate with the action.
        creation_time (datetime): When the action was created.
        created_by (obj): Contextual info on which account created the action.
        last_modified_time (datetime): When the action was last modified.
        last_modified_by (obj): Contextual info on which account created the action.
    """

    action_arn: str = None
    action_name: str = None
    action_type: str = None
    description: str = None
    status: str = None
    source: ActionSource = None
    properties: dict = None
    properties_to_remove: list = None
    tags: list = None
    creation_time: datetime = None
    created_by: str = None
    last_modified_time: datetime = None
    last_modified_by: str = None

    _boto_create_method: str = "create_action"
    _boto_load_method: str = "describe_action"
    _boto_update_method: str = "update_action"
    _boto_delete_method: str = "delete_action"

    _boto_update_members = [
        "action_name",
        "description",
        "status",
        "properties",
        "properties_to_remove",
    ]

    _boto_delete_members = ["action_name"]

    _custom_boto_types = {"source": (_api_types.ActionSource, False)}

    def save(self) -> "Action":
        """Save the state of this Action to SageMaker.

        Returns:
            Action: A SageMaker ``Action``object.
        """
        return self._invoke_api(self._boto_update_method, self._boto_update_members)

    def delete(self, disassociate: bool = False):
        """Delete the action.

        Args:
            disassociate (bool): When set to true, disassociate incoming and outgoing association.

        """
        if disassociate:
            _utils._disassociate(
                source_arn=self.action_arn, sagemaker_session=self.sagemaker_session
            )
            _utils._disassociate(
                destination_arn=self.action_arn,
                sagemaker_session=self.sagemaker_session,
            )

        self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    @classmethod
    def load(cls, action_name: str, sagemaker_session=None) -> "Action":
        """Load an existing action and return an ``Action`` object representing it.

        Args:
            action_name (str): Name of the action
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Action: A SageMaker ``Action`` object
        """
        result = cls._construct(
            cls._boto_load_method,
            action_name=action_name,
            sagemaker_session=sagemaker_session,
        )
        return result

    def set_tag(self, tag=None):
        """Add a tag to the object.

        Args:

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.action_arn, tags=[tag])

    def set_tags(self, tags=None):
        """Add tags to the object.

        Args:
            tags ([{key:value}]): list of key value pairs.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.action_arn, tags=tags)

    @classmethod
    def create(
        cls,
        action_name: str = None,
        source_uri: str = None,
        source_type: str = None,
        action_type: str = None,
        description: str = None,
        status: str = None,
        properties: dict = None,
        tags: dict = None,
        sagemaker_session: Session = None,
    ) -> "Action":
        """Create an action and return an ``Action`` object representing it.

        Args:
            action_name (str): Name of the action
            source_uri (str): Source URI of the action
            source_type (str): Source type of the action
            action_type (str): The type of the action
            description (str): Description of the action
            status (str): Status of the action.
            properties (dict): key/value properties
            tags (dict): AWS tags for the action
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Action: A SageMaker ``Action`` object.
        """
        return super(Action, cls)._construct(
            cls._boto_create_method,
            action_name=action_name,
            source=_api_types.ActionSource(source_uri=source_uri, source_type=source_type),
            action_type=action_type,
            description=description,
            status=status,
            properties=properties,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def list(
        cls,
        source_uri: Optional[str] = None,
        action_type: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        sagemaker_session: Session = None,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
    ) -> Iterator[ActionSummary]:
        """Return a list of action summaries.

        Args:
            source_uri (str, optional): A source URI.
            action_type (str, optional): An action type.
            created_before (datetime.datetime, optional): Return actions created before this
                instant.
            created_after (datetime.datetime, optional): Return actions created after this instant.
            sort_by (str, optional): Which property to sort results by.
                One of 'SourceArn', 'CreatedBefore', 'CreatedAfter'
            sort_order (str, optional): One of 'Ascending', or 'Descending'.
            max_results (int, optional): maximum number of actions to retrieve
            next_token (str, optional): token for next page of results
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            collections.Iterator[ActionSummary]: An iterator
                over ``ActionSummary`` objects.
        """
        return super(Action, cls)._list(
            "list_actions",
            _api_types.ActionSummary.from_boto,
            "ActionSummaries",
            source_uri=source_uri,
            action_type=action_type,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            sagemaker_session=sagemaker_session,
            max_results=max_results,
            next_token=next_token,
        )

    def artifacts(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.BOTH
    ) -> List[Artifact]:
        """Use a lineage query to retrieve all artifacts that use this action.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Artifacts: Artifacts.
        """
        query_filter = LineageFilter(entities=[LineageEntityEnum.ARTIFACT])
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.action_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]


class ModelPackageApprovalAction(Action):
    """An Amazon SageMaker model package approval action, which is part of a SageMaker lineage."""

    def datasets(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> List[Artifact]:
        """Use a lineage query to retrieve all upstream datasets that use this action.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Artifacts: Artifacts representing a dataset.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.ARTIFACT], sources=[LineageSourceEnum.DATASET]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.action_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]

    def model_package(self):
        """Get model package from model package approval action.

        Returns:
            Model package.
        """
        source_uri = self.source.source_uri
        if source_uri is None:
            return None

        model_package_name = source_uri.split("/")[1]
        return self.sagemaker_session.sagemaker_client.describe_model_package(
            ModelPackageName=model_package_name
        )

    def endpoints(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.DESCENDANTS
    ) -> List:
        """Use a lineage query to retrieve downstream endpoint contexts that use this action.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Contexts: Contexts representing an endpoint.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.CONTEXT], sources=[LineageSourceEnum.ENDPOINT]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.action_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]
