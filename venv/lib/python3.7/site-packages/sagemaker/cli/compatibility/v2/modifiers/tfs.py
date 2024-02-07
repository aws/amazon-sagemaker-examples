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
"""Classes to modify TensorFlow Serving code to be compatible with version 2.0 and later."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

CLASS_NAMES = ("Model", "Predictor")
TFS_CLASSES = {name: ("sagemaker.tensorflow.serving",) for name in CLASS_NAMES}


class TensorFlowServingConstructorRenamer(Modifier):
    """A class to rename TensorFlow Serving classes."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a TensorFlow Serving class.

        This looks for the following calls:

        - ``sagemaker.tensorflow.serving.Model``
        - ``sagemaker.tensorflow.serving.Predictor``
        - ``Predictor``

        Because ``Model`` can refer to either ``sagemaker.tensorflow.serving.Model``
        or :class:`~sagemaker.model.Model`, ``Model`` on its own is not sufficient
        for indicating a TFS Model object.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates a TensorFlow Serving class.
        """
        if isinstance(node.func, ast.Name):
            return node.func.id == "Predictor"

        return matching.matches_any(node, TFS_CLASSES)

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to use the classes for TensorFlow Serving.

        Modification is available in version 2.0 and later of the Python SDK:

        - ``sagemaker.tensorflow.TensorFlowModel``
        - ``sagemaker.tensorflow.TensorFlowPredictor``

        Args:
            node (ast.Call): a node that represents a TensorFlow Serving constructor.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        if isinstance(node.func, ast.Name):
            node.func.id = self._new_cls_name(node.func.id)
        else:
            node.func.attr = self._new_cls_name(node.func.attr)
            node.func.value = node.func.value.value
        return node

    def _new_cls_name(self, cls_name):
        """Returns the updated class name."""
        return "TensorFlow{}".format(cls_name)


class TensorFlowServingImportFromRenamer(Modifier):
    """A class to update import statements starting with ``from sagemaker.tensorflow.serving``."""

    def node_should_be_modified(self, node):
        """Checks if the import statement imports from the ``sagemaker.tensorflow.serving`` module.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.ImportFrom`` uses the ``sagemaker.tensorflow.serving`` module.
        """
        return node.module == "sagemaker.tensorflow.serving"

    def modify_node(self, node):
        """Changes and updates the imported class names as applicable.

        Changes the ``ast.ImportFrom`` node's module to ``sagemaker.tensorflow`` and updates the
        imported class names to ``TensorFlowModel`` and ``TensorFlowPredictor``, as applicable.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        node.module = "sagemaker.tensorflow"

        for cls in node.names:
            cls.name = "TensorFlow{}".format(cls.name)
        return node


class TensorFlowServingImportRenamer(Modifier):
    """A class to update ``import sagemaker.tensorflow.serving``."""

    def check_and_modify_node(self, node):
        """Checks if the ``ast.Import`` node imports the ``sagemaker.tensorflow.serving`` module.

        If ``ast.Import`` node imports the ``sagemaker.tensorflow.serving`` module,
        this function changes it to ``sagemaker.tensorflow``.

        Args:
            node (ast.Import): a node that represents an import statement. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        for module in node.names:
            if module.name == "sagemaker.tensorflow.serving":
                module.name = "sagemaker.tensorflow"
        return node
