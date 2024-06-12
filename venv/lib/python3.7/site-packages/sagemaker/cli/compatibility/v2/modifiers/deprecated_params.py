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
"""Classes to remove deprecated parameters."""
from __future__ import absolute_import

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

TF_NAMESPACES = ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator")


class TensorFlowScriptModeParameterRemover(Modifier):
    """A class to remove ``script_mode`` from TensorFlow estimators (because it's the only mode)."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a TensorFlow estimator.

        TensorFlow estimator would use``script_mode`` set. This looks for the following formats:

        - ``TensorFlow``
        - ``sagemaker.tensorflow.TensorFlow``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is instantiating a TensorFlow estimator with ``script_mode``.
        """
        is_tf_constructor = matching.matches_name_or_namespaces(node, "TensorFlow", TF_NAMESPACES)
        return is_tf_constructor and self._has_script_mode_param(node)

    def _has_script_mode_param(self, node):
        """Checks if the ``ast.Call`` node's keywords include ``script_mode``."""
        for kw in node.keywords:
            if kw.arg == "script_mode":
                return True

        return False

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node's keywords to remove ``script_mode``.

        Args:
            node (ast.Call): a node that represents a TensorFlow constructor.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        for kw in node.keywords:
            if kw.arg == "script_mode":
                node.keywords.remove(kw)
        return node
