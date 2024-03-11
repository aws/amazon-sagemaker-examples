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

"""This module contains code related to Amazon SageMaker Collections in the Model Registry.

Use these methods to help you create and maintain your Collections.
"""

from __future__ import absolute_import
import json
import time
from typing import List


from botocore.exceptions import ClientError
from sagemaker.session import Session


class Collection(object):
    """Sets up an Amazon SageMaker Collection."""

    def __init__(self, sagemaker_session):
        """Initializes a Collection instance.

        A Collection is a logical grouping of Model Groups.

        Args:
            sagemaker_session (sagemaker.session.Session): A Session object which
                manages interactions between Amazon SageMaker APIs and other
                AWS services needed. If unspecified, a session is created using
                the default AWS configuration chain.
        """

        self.sagemaker_session = sagemaker_session or Session()

    def _check_access_error(self, err: ClientError):
        """Checks if the error is related to the access error and provide the relevant message.

        Args:
            err: The client error to check.
        """

        error_code = err.response["Error"]["Code"]
        if error_code == "AccessDeniedException":
            raise Exception(
                f"{error_code}: This account needs to attach a custom policy "
                "to the user role to gain access to Collections. Refer - "
                "https://docs.aws.amazon.com/sagemaker/latest/dg/modelcollections-permissions.html"
            )

    def _add_model_group(self, model_package_group, tag_rule_key, tag_rule_value):
        """Adds a Model Group to a Collection.

        Args:
            model_package_group (str): The name of the Model Group.
            tag_rule_key (str): The tag key of the destination collection.
            tag_rule_value (str): The tag value of the destination collection.
        """
        model_group_details = self.sagemaker_session.sagemaker_client.describe_model_package_group(
            ModelPackageGroupName=model_package_group
        )
        self.sagemaker_session.sagemaker_client.add_tags(
            ResourceArn=model_group_details["ModelPackageGroupArn"],
            Tags=[
                {
                    "Key": tag_rule_key,
                    "Value": tag_rule_value,
                }
            ],
        )

    def _remove_model_group(self, model_package_group, tag_rule_key):
        """Removes a Model Group from a Collection.

        Args:
            model_package_group (str): The name of the Model Group
            tag_rule_key (str): The tag key of the Collection from which to remove the Model Group.
        """
        model_group_details = self.sagemaker_session.sagemaker_client.describe_model_package_group(
            ModelPackageGroupName=model_package_group
        )
        self.sagemaker_session.sagemaker_client.delete_tags(
            ResourceArn=model_group_details["ModelPackageGroupArn"], TagKeys=[tag_rule_key]
        )

    def create(self, collection_name: str, parent_collection_name: str = None):
        """Creates a Collection.

        Args:
            collection_name (str): The name of the Collection to create.
            parent_collection_name (str): The name of the parent Collection.
                Is ``None`` if the Collection is created at the root level.
        """

        tag_rule_key = f"sagemaker:collection-path:{int(time.time() * 1000)}"
        tags_on_collection = {
            "sagemaker:collection": "true",
            "sagemaker:collection-path:root": "true",
        }
        tag_rule_values = [collection_name]

        if parent_collection_name is not None:
            parent_tag_rules = self._get_collection_tag_rule(collection_name=parent_collection_name)
            parent_tag_rule_key = parent_tag_rules["tag_rule_key"]
            parent_tag_value = parent_tag_rules["tag_rule_value"]
            tags_on_collection = {
                parent_tag_rule_key: parent_tag_value,
                "sagemaker:collection": "true",
            }
            tag_rule_values = [f"{parent_tag_value}/{collection_name}"]
        try:
            resource_filters = [
                "AWS::SageMaker::ModelPackageGroup",
                "AWS::ResourceGroups::Group",
            ]

            tag_filters = [
                {
                    "Key": tag_rule_key,
                    "Values": tag_rule_values,
                }
            ]
            resource_query = {
                "Query": json.dumps(
                    {"ResourceTypeFilters": resource_filters, "TagFilters": tag_filters}
                ),
                "Type": "TAG_FILTERS_1_0",
            }
            collection_create_response = self.sagemaker_session.create_group(
                collection_name, resource_query, tags_on_collection
            )
            return {
                "Name": collection_create_response["Group"]["Name"],
                "Arn": collection_create_response["Group"]["GroupArn"],
            }
        except ClientError as e:
            message = e.response["Error"]["Message"]
            error_code = e.response["Error"]["Code"]

            if error_code == "BadRequestException" and "group already exists" in message:
                raise ValueError("Collection with the given name already exists")
            self._check_access_error(err=e)
            raise

    def delete(self, collections: List[str]):
        """Deletes a list of Collections.

        Args:
            collections (List[str]): A list of Collections to delete.
                Only deletes a Collection if it is empty.
        """

        if len(collections) > 10:
            raise ValueError("Can delete upto 10 collections at a time")

        delete_collection_failures = []
        deleted_collection = []
        collection_filter = [
            {
                "Name": "resource-type",
                "Values": ["AWS::ResourceGroups::Group", "AWS::SageMaker::ModelPackageGroup"],
            },
        ]

        # loops over the list of collection and deletes one at a time.
        for collection in collections:
            try:
                collection_details = self.sagemaker_session.list_group_resources(
                    group=collection, filters=collection_filter
                )
            except ClientError as e:
                self._check_access_error(err=e)
                delete_collection_failures.append(
                    {"collection": collection, "message": e.response["Error"]["Message"]}
                )
                continue
            if collection_details.get("Resources") and len(collection_details["Resources"]) > 0:
                delete_collection_failures.append(
                    {"collection": collection, "message": "Validation error: Collection not empty"}
                )
            else:
                try:
                    self.sagemaker_session.delete_resource_group(group=collection)
                    deleted_collection.append(collection)
                except ClientError as e:
                    self._check_access_error(err=e)
                    delete_collection_failures.append(
                        {"collection": collection, "message": e.response["Error"]["Message"]}
                    )
        return {
            "deleted_collections": deleted_collection,
            "delete_collection_failures": delete_collection_failures,
        }

    def _get_collection_tag_rule(self, collection_name: str):
        """Returns the tag rule key and value for a Collection."""

        if collection_name is not None:
            try:
                group_query = self.sagemaker_session.get_resource_group_query(group=collection_name)
            except ClientError as e:
                error_code = e.response["Error"]["Code"]

                if error_code == "NotFoundException":
                    raise ValueError(f"Cannot find collection: {collection_name}")
                self._check_access_error(err=e)
                raise
            if group_query.get("GroupQuery"):
                tag_rule_query = json.loads(
                    group_query["GroupQuery"].get("ResourceQuery", {}).get("Query", "")
                )
                tag_rule = tag_rule_query.get("TagFilters", [])[0]
                if not tag_rule:
                    raise "Unsupported parent_collection_name"
                tag_rule_value = tag_rule["Values"][0]
                tag_rule_key = tag_rule["Key"]

            return {
                "tag_rule_key": tag_rule_key,
                "tag_rule_value": tag_rule_value,
            }
        raise ValueError("Collection name is required")

    def add_model_groups(self, collection_name: str, model_groups: List[str]):
        """Adds a list of Model Groups to a Collection.

        Args:
            collection_name (str): The name of the Collection.
            model_groups (List[str]): The names of the Model Groups to add to the Collection.
        """
        if len(model_groups) > 10:
            raise Exception("Model groups can have a maximum length of 10")
        tag_rules = self._get_collection_tag_rule(collection_name=collection_name)
        tag_rule_key = tag_rules["tag_rule_key"]
        tag_rule_value = tag_rules["tag_rule_value"]

        add_groups_success = []
        add_groups_failure = []
        if tag_rule_key is not None and tag_rule_value is not None:
            for model_group in model_groups:
                try:
                    self._add_model_group(
                        model_package_group=model_group,
                        tag_rule_key=tag_rule_key,
                        tag_rule_value=tag_rule_value,
                    )
                    add_groups_success.append(model_group)
                except ClientError as e:
                    self._check_access_error(err=e)
                    message = e.response["Error"]["Message"]
                    add_groups_failure.append(
                        {
                            "model_group": model_group,
                            "failure_reason": message,
                        }
                    )
        return {
            "added_groups": add_groups_success,
            "failure": add_groups_failure,
        }

    def remove_model_groups(self, collection_name: str, model_groups: List[str]):
        """Removes a list of Model Groups from a Collection.

        Args:
            collection_name (str): The name of the Collection.
            model_groups (List[str]): The names of the Model Groups to remove.
        """

        if len(model_groups) > 10:
            raise Exception("Model groups can have a maximum length of 10")
        tag_rules = self._get_collection_tag_rule(collection_name=collection_name)

        tag_rule_key = tag_rules["tag_rule_key"]
        tag_rule_value = tag_rules["tag_rule_value"]

        remove_groups_success = []
        remove_groups_failure = []
        if tag_rule_key is not None and tag_rule_value is not None:
            for model_group in model_groups:
                try:
                    self._remove_model_group(
                        model_package_group=model_group,
                        tag_rule_key=tag_rule_key,
                    )
                    remove_groups_success.append(model_group)
                except ClientError as e:
                    self._check_access_error(err=e)
                    message = e.response["Error"]["Message"]
                    remove_groups_failure.append(
                        {
                            "model_group": model_group,
                            "failure_reason": message,
                        }
                    )
        return {
            "removed_groups": remove_groups_success,
            "failure": remove_groups_failure,
        }

    def move_model_group(
        self, source_collection_name: str, model_group: str, destination_collection_name: str
    ):
        """Moves a Model Group from one Collection to another.

        Args:
            source_collection_name (str): The name of the source Collection.
            model_group (str): The name of the Model Group to move.
            destination_collection_name (str): The name of the destination Collection.
        """
        remove_details = self.remove_model_groups(
            collection_name=source_collection_name, model_groups=[model_group]
        )
        if len(remove_details["failure"]) == 1:
            raise Exception(remove_details["failure"][0]["failure"])

        added_details = self.add_model_groups(
            collection_name=destination_collection_name, model_groups=[model_group]
        )

        if len(added_details["failure"]) == 1:
            # adding the Model Group back to the source collection in case of an add failure
            self.add_model_groups(
                collection_name=source_collection_name, model_groups=[model_group]
            )
            raise Exception(added_details["failure"][0]["failure"])

        return {
            "moved_success": model_group,
        }

    def _convert_tag_collection_response(self, tag_collections: List[str]):
        """Converts a Collection response from the tag api to a Collection list response.

        Args:
            tag_collections List[dict]: The Collection list response from the tag api.
        """
        collection_details = []
        for collection in tag_collections:
            collection_arn = collection["ResourceARN"]
            collection_name = collection_arn.split("group/")[1]
            collection_details.append(
                {
                    "Name": collection_name,
                    "Arn": collection_arn,
                    "Type": "Collection",
                }
            )
        return collection_details

    def _convert_group_resource_response(
        self, group_resource_details: List[dict], is_model_group: bool = False
    ):
        """Converts a Collection response from the resource group api to a Collection list response.

        Args:
            group_resource_details (List[dict]): The Collection list response from the
                resource group api.
            is_model_group (bool): Indicates if the response is of Collection or Model Group type.
        """
        collection_details = []
        if group_resource_details["Resources"]:
            for resource_group in group_resource_details["Resources"]:
                collection_arn = resource_group["Identifier"]["ResourceArn"]
                collection_name = collection_arn.split("group/")[1]
                collection_details.append(
                    {
                        "Name": collection_name,
                        "Arn": collection_arn,
                        "Type": resource_group["Identifier"]["ResourceType"]
                        if is_model_group
                        else "Collection",
                    }
                )
        return collection_details

    def _get_full_list_resource(self, collection_name, collection_filter):
        """Iterates the full resource group list and returns the appended paginated response.

        Args:
            collection_name (str): The name of the Collection from which to get details.
            collection_filter (dict): Filter details to pass to get the resource list.
        """
        list_group_response = self.sagemaker_session.list_group_resources(
            group=collection_name, filters=collection_filter
        )
        next_token = list_group_response.get("NextToken")
        while next_token is not None:

            paginated_group_response = self.sagemaker_session.list_group_resources(
                group=collection_name,
                filters=collection_filter,
                next_token=next_token,
            )
            list_group_response["Resources"] = (
                list_group_response["Resources"] + paginated_group_response["Resources"]
            )
            list_group_response["ResourceIdentifiers"] = (
                list_group_response["ResourceIdentifiers"]
                + paginated_group_response["ResourceIdentifiers"]
            )
            next_token = paginated_group_response.get("NextToken")

        return list_group_response

    def list_collection(self, collection_name: str = None):
        """Lists the contents of the specified Collection.

        If there is no Collection with the name ``collection_name``, lists all the
        Collections at the root level.

        Args:
            collection_name (str): The name of the Collection whose contents are listed.
        """
        collection_content = []
        if collection_name is None:
            tag_filters = [
                {
                    "Key": "sagemaker:collection-path:root",
                    "Values": ["true"],
                },
            ]
            resource_type_filters = ["resource-groups:group"]
            tag_collections = self.sagemaker_session.get_tagging_resources(
                tag_filters=tag_filters, resource_type_filters=resource_type_filters
            )

            return self._convert_tag_collection_response(tag_collections)

        collection_filter = [
            {
                "Name": "resource-type",
                "Values": ["AWS::ResourceGroups::Group"],
            },
        ]
        list_group_response = self._get_full_list_resource(
            collection_name=collection_name, collection_filter=collection_filter
        )
        collection_content = self._convert_group_resource_response(list_group_response)

        collection_filter = [
            {
                "Name": "resource-type",
                "Values": ["AWS::SageMaker::ModelPackageGroup"],
            },
        ]
        list_group_response = self._get_full_list_resource(
            collection_name=collection_name, collection_filter=collection_filter
        )

        collection_content = collection_content + self._convert_group_resource_response(
            list_group_response, True
        )

        return collection_content
