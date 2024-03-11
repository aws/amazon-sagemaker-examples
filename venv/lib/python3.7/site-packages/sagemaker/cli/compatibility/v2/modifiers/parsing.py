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
"""Functions for parsing AST nodes."""
from __future__ import absolute_import

import pasta


def arg_from_keywords(node, arg):
    """Retrieves a keyword argument from the node's keywords.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        arg (str): the name of the argument.

    Returns:
        ast.keyword: the keyword argument if it is present. Otherwise, this returns ``None``.
    """
    for kw in node.keywords:
        if kw.arg == arg:
            return kw

    return None


def arg_value(node, arg):
    """Retrieves a keyword argument's value from the node's keywords.

    Args:
        node (ast.Call): a node that represents a function call. For more,
            see https://docs.python.org/3/library/ast.html#abstract-grammar.
        arg (str): the name of the argument.

    Returns:
        obj: the keyword argument's value.

    Raises:
        KeyError: if the node's keywords do not contain the argument.
    """
    keyword = arg_from_keywords(node, arg)
    if keyword is None:
        raise KeyError("arg '{}' not found in call: {}".format(arg, pasta.dump(node)))

    return getattr(keyword.value, keyword.value._fields[0], None) if keyword.value else None
