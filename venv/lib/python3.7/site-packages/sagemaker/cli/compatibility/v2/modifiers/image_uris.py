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
"""Classes to modify image uri retrieve methods for Python SDK v2.0 and later."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

GET_IMAGE_URI_NAME = "get_image_uri"
GET_IMAGE_URI_NAMESPACES = (
    "sagemaker",
    "sagemaker.amazon_estimator",
    "sagemaker.amazon.amazon_estimator",
    "amazon_estimator",
    "amazon.amazon_estimator",
)


class ImageURIRetrieveRefactor(Modifier):
    """A class to refactor *get_image_uri() method."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node calls a function of interest.

        This looks for the following calls:

        - ``sagemaker.get_image_uri``
        - ``sagemaker.amazon_estimator.get_image_uri``
        - ``get_image_uri``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates a class of interest.
        """
        return matching.matches_name_or_namespaces(
            node, GET_IMAGE_URI_NAME, GET_IMAGE_URI_NAMESPACES
        )

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to call ``image_uris.retrieve`` instead.

        And also switch the first two parameters from (region, repo) to (framework, region).

        Args:
            node (ast.Call): a node that represents a *image_uris.retrieve call.
        """
        original_args = [None] * 3
        for kw in node.keywords:
            if kw.arg == "repo_name":
                original_args[0] = ast.Str(kw.value.s)
            elif kw.arg == "repo_region":
                original_args[1] = ast.Str(kw.value.s)
            elif kw.arg == "repo_version":
                original_args[2] = ast.Str(kw.value.s)

        if len(node.args) > 0:
            original_args[1] = ast.Str(node.args[0].s)
        if len(node.args) > 1:
            original_args[0] = ast.Str(node.args[1].s)
        if len(node.args) > 2:
            original_args[2] = ast.Str(node.args[2].s)

        args = []
        for arg in original_args:
            if arg:
                args.append(arg)

        func = node.func
        has_sagemaker = False
        while hasattr(func, "value"):
            if hasattr(func.value, "id") and func.value.id == "sagemaker":
                has_sagemaker = True
                break
            func = func.value

        if has_sagemaker:
            node.func = ast.Attribute(
                value=ast.Attribute(attr="image_uris", value=ast.Name(id="sagemaker")),
                attr="retrieve",
            )
        else:
            node.func = ast.Attribute(value=ast.Name(id="image_uris"), attr="retrieve")
        node.args = args
        node.keywords = []
        return node


class ImageURIRetrieveImportFromRenamer(Modifier):
    """A class to update import statements of ``get_image_uri``."""

    def node_should_be_modified(self, node):
        """Checks if the import statement imports ``get_image_uri`` from the correct module.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the import statement imports ``get_image_uri`` from the correct module.
        """
        return (
            node is not None
            and node.module in GET_IMAGE_URI_NAMESPACES
            and any(name.name == GET_IMAGE_URI_NAME for name in node.names)
        )

    def modify_node(self, node):
        """Changes the ``ast.ImportFrom`` node's name from ``get_image_uri`` to ``image_uris``.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        for name in node.names:
            if name.name == GET_IMAGE_URI_NAME:
                name.name = "image_uris"
            if node.module in GET_IMAGE_URI_NAMESPACES:
                node.module = "sagemaker"
        return node
