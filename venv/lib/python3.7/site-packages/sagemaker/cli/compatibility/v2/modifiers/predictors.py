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
"""Classes to modify Predictor code to be compatible with version 2.0 and later."""
from __future__ import absolute_import

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

BASE_PREDICTOR = "RealTimePredictor"
PREDICTORS = {
    "FactorizationMachinesPredictor": ("sagemaker", "sagemaker.amazon.factorization_machines"),
    "IPInsightsPredictor": ("sagemaker", "sagemaker.amazon.ipinsights"),
    "KMeansPredictor": ("sagemaker", "sagemaker.amazon.kmeans"),
    "KNNPredictor": ("sagemaker", "sagemaker.amazon.knn"),
    "LDAPredictor": ("sagemaker", "sagemaker.amazon.lda"),
    "LinearLearnerPredictor": ("sagemaker", "sagemaker.amazon.linear_learner"),
    "NTMPredictor": ("sagemaker", "sagemaker.amazon.ntm"),
    "PCAPredictor": ("sagemaker", "sagemaker.amazon.pca"),
    "RandomCutForestPredictor": ("sagemaker", "sagemaker.amazon.randomcutforest"),
    "RealTimePredictor": ("sagemaker", "sagemaker.predictor"),
    "SparkMLPredictor": ("sagemaker.sparkml", "sagemaker.sparkml.model"),
}


class PredictorConstructorRefactor(Modifier):
    """A class to refactor *Predictor class and refactor endpoint attribute."""

    def node_should_be_modified(self, node):
        """Checks if the ``ast.Call`` node instantiates a class of interest.

        This looks for the following calls:

        - ``sagemaker.<my>.<namespace>.<MyPredictor>``
        - ``sagemaker.<namespace>.<MyPredictor>``
        - ``<MyPredictor>``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` instantiates a class of interest.
        """
        return matching.matches_any(node, PREDICTORS)

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to call ``Predictor`` instead.

        Also renames ``endpoint`` attribute to ``endpoint_name``.

        Args:
            node (ast.Call): a node that represents a *Predictor constructor.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        _rename_class(node)
        _rename_endpoint(node)
        return node


def _rename_class(node):
    """Renames the RealTimePredictor base class to Predictor"""
    if matching.matches_name(node, BASE_PREDICTOR):
        node.func.id = "Predictor"
    elif matching.matches_attr(node, BASE_PREDICTOR):
        node.func.attr = "Predictor"


def _rename_endpoint(node):
    """Renames keyword endpoint argument to endpoint_name"""
    for keyword in node.keywords:
        if keyword.arg == "endpoint":
            keyword.arg = "endpoint_name"
            break


class PredictorImportFromRenamer(Modifier):
    """A class to update import statements of ``RealTimePredictor``."""

    def node_should_be_modified(self, node):
        """Checks if the import statement imports ``RealTimePredictor`` from the correct module.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the import statement imports ``RealTimePredictor`` from the correct module.
        """
        return node.module in PREDICTORS[BASE_PREDICTOR] and any(
            name.name == BASE_PREDICTOR for name in node.names
        )

    def modify_node(self, node):
        """Changes the ``ast.ImportFrom`` node's name from ``RealTimePredictor`` to ``Predictor``.

        Args:
            node (ast.ImportFrom): a node that represents a ``from ... import ... `` statement.
                For more, see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        for name in node.names:
            if name.name == BASE_PREDICTOR:
                name.name = "Predictor"
        return node
