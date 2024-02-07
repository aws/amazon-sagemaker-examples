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
"""Provides utilities for custom boto type objects."""
from __future__ import absolute_import

from sagemaker.apiutils import _boto_functions, _utils


class ApiObject(object):
    """A Python class representation of a boto API object.

    Converts boto dicts of 'UpperCamelCase' names to dicts into/from a Python object with standard
    python members. Clients invoke to_boto on an instance of ApiObject to transform the ApiObject
    into a boto representation. Clients invoke from_boto on a sub-class of ApiObject to
    instantiate an instance of that class from a boto representation.
    """

    # A map from boto 'UpperCamelCase' name to member name. If a boto name does not appear in
    # this dict then it is converted to lower_snake_case.
    _custom_boto_names = {}

    # A map from name to an ApiObject subclass. Allows ApiObjects to contain ApiObject members.
    _custom_boto_types = {}

    def __init__(self, **kwargs):
        """Init ApiObject."""
        self.__dict__.update(kwargs)

    @classmethod
    def _boto_ignore(cls):
        """Response fields to ignore by default."""
        return ["ResponseMetadata"]

    @classmethod
    def from_boto(cls, boto_dict, **kwargs):
        """Construct an instance of this ApiObject from a boto response.

        Args:
            boto_dict (dict): A dictionary of a boto response.
            **kwargs: Arbitrary keyword arguments
        """
        if boto_dict is None:
            return None

        boto_dict = {k: v for k, v in boto_dict.items() if k not in cls._boto_ignore()}
        custom_boto_names_to_member_names = {a: b for b, a in cls._custom_boto_names.items()}
        cls_kwargs = _boto_functions.from_boto(
            boto_dict, custom_boto_names_to_member_names, cls._custom_boto_types
        )
        cls_kwargs.update(kwargs)
        return cls(**cls_kwargs)

    @classmethod
    def to_boto(cls, obj):
        """Convert an object to a boto representation.

        Args:
            obj (dict): The object to convert to boto.
        """
        if not isinstance(obj, dict):
            var_dict = vars(obj)
        else:
            var_dict = obj
        return _boto_functions.to_boto(var_dict, cls._custom_boto_names, cls._custom_boto_types)

    def __eq__(self, other):
        """Return True if this ApiObject equals other."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Return True if this ApiObject does not equal other."""
        return not self.__eq__(other)

    def __hash__(self):
        """Return a hashcode for this ApiObject."""
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        """Return a string representation of this ApiObject."""
        return "{}({})".format(
            type(self).__name__,
            ",".join(["{}={}".format(k, repr(v)) for k, v in vars(self).items()]),
        )


class Record(ApiObject):
    """A boto based Active Record class based on convention of CRUD operations."""

    # update / delete / list method names
    _boto_update_method = None
    _boto_delete_method = None
    _boto_list_method = None

    # List of member names to convert to boto representations and pass to the update method.
    _boto_update_members = []

    # List of member names to convert to boto representations and pass to the delete method.
    _boto_delete_members = []

    def __init__(self, sagemaker_session=None, **kwargs):
        """Init Record."""
        self.sagemaker_session = sagemaker_session
        super(Record, self).__init__(**kwargs)

    @classmethod
    def _list(
        cls,
        boto_list_method,
        list_item_factory,
        boto_list_items_name,
        boto_next_token_name="NextToken",
        sagemaker_session=None,
        **kwargs
    ):
        """List objects from the SageMaker API."""
        sagemaker_session = sagemaker_session or _utils.default_session()
        sagemaker_client = sagemaker_session.sagemaker_client
        next_token = None
        try:
            while True:
                list_request_kwargs = _boto_functions.to_boto(
                    kwargs, cls._custom_boto_names, cls._custom_boto_types
                )
                if next_token:
                    list_request_kwargs[boto_next_token_name] = next_token
                list_method = getattr(sagemaker_client, boto_list_method)
                list_method_response = list_method(**list_request_kwargs)
                list_items = list_method_response.get(boto_list_items_name, [])
                next_token = list_method_response.get(boto_next_token_name)
                for item in list_items:
                    yield list_item_factory(item)
                if not next_token:
                    break
        except StopIteration:
            return

    @classmethod
    def _search(
        cls,
        search_resource,
        search_item_factory,
        boto_next_token_name="NextToken",
        sagemaker_session=None,
        **kwargs
    ):
        """Search for objects with the SageMaker API."""
        sagemaker_session = sagemaker_session or _utils.default_session()
        sagemaker_client = sagemaker_session.sagemaker_client

        next_token = None
        try:
            while True:
                search_request_kwargs = _boto_functions.to_boto(
                    kwargs, cls._custom_boto_names, cls._custom_boto_types
                )
                search_request_kwargs["Resource"] = search_resource
                if next_token:
                    search_request_kwargs[boto_next_token_name] = next_token
                search_method = getattr(sagemaker_client, "search")
                search_method_response = search_method(**search_request_kwargs)
                search_items = search_method_response.get("Results", [])
                next_token = search_method_response.get(boto_next_token_name)
                for item in search_items:
                    # _TrialComponent class in experiments module is not public currently
                    class_name = cls.__name__.lstrip("_")
                    if class_name in item:
                        yield search_item_factory(item[class_name])
                if not next_token:
                    break
        except StopIteration:
            return

    @classmethod
    def _construct(cls, boto_method_name, sagemaker_session=None, **kwargs):
        """Create and invoke a SageMaker API call request."""
        sagemaker_session = sagemaker_session or _utils.default_session()
        instance = cls(sagemaker_session, **kwargs)
        return instance._invoke_api(boto_method_name, kwargs)

    def _set_tags(self, resource_arn=None, tags=None):
        """Set tags on this ApiObject.

        Args:
            resource_arn (str): The arn of the Record
            tags (dict): An array of Tag objects that set to Record

        Returns:
            A list of key, value pair objects. i.e. [{"key":"value"}]
        """
        tag_list = self.sagemaker_session.sagemaker_client.add_tags(
            ResourceArn=resource_arn, Tags=tags
        )["Tags"]
        return tag_list

    def with_boto(self, boto_dict):
        """Update this ApiObject with a boto response.

        Args:
            boto_dict (dict): A dictionary of a boto response.
        """
        custom_boto_names_to_member_names = {a: b for b, a in self._custom_boto_names.items()}
        self.__dict__.update(
            **_boto_functions.from_boto(
                boto_dict, custom_boto_names_to_member_names, self._custom_boto_types
            )
        )
        return self

    def _invoke_api(self, boto_method, boto_method_members):
        """Invoke a SageMaker API."""
        api_values = {k: v for k, v in vars(self).items() if k in boto_method_members}
        api_kwargs = self.to_boto(api_values)
        api_method = getattr(self.sagemaker_session.sagemaker_client, boto_method)
        api_boto_response = api_method(**api_kwargs)
        return self.with_boto(api_boto_response)
