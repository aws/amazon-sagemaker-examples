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
"""Functions for checking AST nodes for matches."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers import parsing


def matches_any(node, name_to_namespaces_dict):
    """Determines if the ``ast.Call`` node matches any of the provided names and namespaces.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        name_to_namespaces_dict (dict[str, tuple]): a mapping of names to appropriate namespaces.

    Returns:
        bool: if the node matches any of the names and namespaces.
    """
    return any(
        matches_name_or_namespaces(node, name, namespaces)
        for name, namespaces in name_to_namespaces_dict.items()
    )


def matches_name_or_namespaces(node, name, namespaces):
    """Determines if the ``ast.Call`` node matches the function name in the right namespace.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        name (str): the function name.
        namespaces (tuple): the possible namespaces to match to.

    Returns:
        bool: if the node matches the name and any of the namespaces.
    """
    if matches_name(node, name):
        return True

    if not matches_attr(node, name):
        return False

    return any(matches_namespace(node, namespace) for namespace in namespaces)


def matches_name(node, name):
    """Determines if the ``ast.Call`` node points to an ``ast.Name`` node with a matching name.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        name (str): the function name.

    Returns:
        bool: if ``node.func`` is an ``ast.Name`` node with a matching name.
    """
    return isinstance(node.func, ast.Name) and node.func.id == name


def matches_attr(node, name):
    """Determines if the ``ast.Call`` node points to an ``ast.Attribute`` node with a matching name.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        name (str): the function name.

    Returns:
        bool: if ``node.func`` is an ``ast.Attribute`` node with a matching name.
    """
    return isinstance(node.func, ast.Attribute) and node.func.attr == name


def matches_namespace(node, namespace):
    """Determines if the ``ast.Call`` node corresponds to a matching namespace.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        namespace (str): the namespace.

    Returns:
        bool: if the node's namespaces matches the given namespace.
    """
    names = namespace.split(".")
    name, value = names.pop(), node.func.value
    while isinstance(value, ast.Attribute) and len(names) > 0:
        if value.attr != name:
            return False
        name, value = names.pop(), value.value

    return isinstance(value, ast.Name) and value.id == name


def has_arg(node, arg):
    """Checks if the call has the given argument.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        arg (str): the name of the argument.

    Returns:
        bool: if the node has the given argument.
    """
    try:
        return parsing.arg_value(node, arg) is not None
    except KeyError:
        return False
