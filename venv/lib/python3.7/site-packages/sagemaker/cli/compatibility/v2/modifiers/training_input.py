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
"""Classes to modify TrainingInput code to be compatible with version 2.0 and later."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

S3_INPUT_NAME = "s3_input"
S3_INPUT_NAMESPACES = ("sagemaker", "sagemaker.inputs", "sagemaker.session")


class TrainingInputConstructorRefactor(Modifier):
    """A class to refactor *s3_input class."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a class of interest.

        This looks for the following calls:

        - ``sagemaker.s3_input``
        - ``sagemaker.session.s3_input``
        - ``s3_input``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates a class of interest.
        """
        return matching.matches_name_or_namespaces(node, S3_INPUT_NAME, S3_INPUT_NAMESPACES)

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to call ``TrainingInput`` instead.

        Args:
            node (ast.Call): a node that represents a *TrainingInput constructor.
        """
        if matching.matches_name(node, S3_INPUT_NAME):
            node.func.id = "TrainingInput"
        elif matching.matches_attr(node, S3_INPUT_NAME):
            node.func.attr = "TrainingInput"
            _rename_namespace(node, "session")
        return node


def _rename_namespace(node, name):
    """Rename namespace ``session`` to ``inputs``."""
    if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == name:
        node.func.value.attr = "inputs"
    elif isinstance(node.func.value, ast.Name) and node.func.value.id == name:
        node.func.value.id = "inputs"


class TrainingInputImportFromRenamer(Modifier):
    """A class to update import statements of ``s3_input``."""

    def node_should_be_modified(self, node):
        """Checks if the import statement imports ``s3_input`` from the correct module.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the import statement imports ``s3_input`` from the correct module.
        """
        return node.module in S3_INPUT_NAMESPACES and any(
            name.name == S3_INPUT_NAME for name in node.names
        )

    def modify_node(self, node):
        """Changes the ``ast.ImportFrom`` node's name from ``s3_input`` to ``TrainingInput``.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        for name in node.names:
            if name.name == S3_INPUT_NAME:
                name.name = "TrainingInput"
            if node.module == "sagemaker.session":
                node.module = "sagemaker.inputs"
        return node


class ShuffleConfigModuleRenamer(Modifier):
    """A class to change ``ShuffleConfig`` usage to use ``sagemaker.inputs.ShuffleConfig``."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a class of interest.

        This looks for the following calls:

        - ``sagemaker.session.ShuffleConfig``
        - ``session.ShuffleConfig``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates a class of interest.
        """
        if isinstance(node.func, ast.Name):
            return False

        return matching.matches_name_or_namespaces(
            node, "ShuffleConfig", ("sagemaker.session", "session")
        )

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to call ``sagemaker.inputs.ShuffleConfig``.

        Args:
            node (ast.Call): a node that represents a ``sagemaker.session.ShuffleConfig``
                constructor.

        Returns:
            ast.Call: the original node, with its namespace changed to use the ``inputs`` module.
        """
        _rename_namespace(node, "session")
        return node


class ShuffleConfigImportFromRenamer(Modifier):
    """A class to update import statements of ``ShuffleConfig``."""

    def node_should_be_modified(self, node):
        """Checks if the import statement imports ``sagemaker.session.ShuffleConfig``.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the import statement imports ``sagemaker.session.ShuffleConfig``.
        """
        return node.module == "sagemaker.session" and any(
            name.name == "ShuffleConfig" for name in node.names
        )

    def modify_node(self, node):
        """Changes the ``ast.ImportFrom`` node's namespace to ``sagemaker.inputs``.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            ast.ImportFrom: the original node, with its module modified to ``"sagemaker.inputs"``.
        """
        node.module = "sagemaker.inputs"
        return node
