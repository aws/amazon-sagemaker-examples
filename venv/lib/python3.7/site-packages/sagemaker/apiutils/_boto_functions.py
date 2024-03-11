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
"""Provides utilities for converting between python style and boto style."""
from __future__ import absolute_import

import re


def to_camel_case(snake_case):
    """Convert a snake case string to camel case.

    Args:
        snake_case (str): String to convert to camel case.

    Returns:
        str: String converted to camel case.
    """
    return "".join([x.title() for x in snake_case.split("_")])


def to_snake_case(name):
    """Convert a camel case string to snake case.

    Args:
        name (str): String to convert to snake case.

    Returns:
        str: String converted to snake case.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def from_boto(boto_dict, boto_name_to_member_name, member_name_to_type):
    """Convert an UpperCamelCase boto response to a snake case representation.

    Args:
        boto_dict (dict[str, ?]): A boto response dictionary.
        boto_name_to_member_name (dict[str, str]):  A map from boto name to snake_case name.
            If a given boto name is not in the map then a default mapping is applied.
        member_name_to_type (dict[str, (_base_types.ApiObject, boolean)]): A map from snake case
            name to a type description tuple. The first element of the tuple, a subclass of
            ApiObject, is the type of the mapped object. The second element indicates whether the
            mapped element is a collection or singleton.

    Returns:
        dict: Boto response in snake case.
    """
    from_boto_values = {}
    for boto_name, boto_value in boto_dict.items():
        # Convert the boto_name to a snake-case name by preferentially looking up the boto name in
        # boto_name_to_member_name before defaulting to the snake case representation
        member_name = boto_name_to_member_name.get(boto_name, to_snake_case(boto_name))

        # If the member name maps to a subclass of _base_types.ApiObject
        # (i.e. it's in member_name_to_type), then transform its boto dictionary using that type:
        if member_name in member_name_to_type:
            api_type, is_collection = member_name_to_type[member_name]
            if is_collection:
                if isinstance(boto_value, dict):
                    member_value = {
                        key: api_type.from_boto(value) for key, value in boto_value.items()
                    }
                else:
                    member_value = [api_type.from_boto(item) for item in boto_value]
            else:
                member_value = api_type.from_boto(boto_value)
        # If the member name does not have a custom type definition then simply assign it the
        # boto value.  Appropriate if the type is simple and requires not further conversion (e.g.
        # a number or list of strings).
        else:
            member_value = boto_value
        from_boto_values[member_name] = member_value
    return from_boto_values


def to_boto(member_vars, member_name_to_boto_name, member_name_to_type):
    """Convert a dict of of snake case names to values into a boto UpperCamelCase representation.

    Args:
        member_vars dict[str, ?]: A map from snake case name to value.
        member_name_to_boto_name dict[str, ?]: A map from snake_case name to boto name.

     Returns:
         dict: boto dict converted to snake case

    """
    to_boto_values = {}
    # Strip out all entries in member_vars that have a None value. None values are treated as
    # not having a value
    # set, required as API operations can have optional parameters that may not take a null value.
    member_vars = {k: v for k, v in member_vars.items() if v is not None}

    # Iterate over each snake_case name and its value and map to a camel case name. If the value
    # is an ApiObject subclass then recursively map its entries.
    for member_name, member_value in member_vars.items():
        boto_name = member_name_to_boto_name.get(member_name, to_camel_case(member_name))
        api_type, is_api_collection_type = member_name_to_type.get(member_name, (None, None))
        if is_api_collection_type and isinstance(member_value, dict):
            boto_value = {
                k: api_type.to_boto(v) if api_type else v for k, v in member_value.items()
            }
        elif is_api_collection_type and isinstance(member_value, list):
            boto_value = [api_type.to_boto(v) if api_type else v for v in member_value]
        else:
            boto_value = api_type.to_boto(member_value) if api_type else member_value
        to_boto_values[boto_name] = boto_value
    return to_boto_values
