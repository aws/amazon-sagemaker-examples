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
"""An ast.NodeTransformer subclass for updating SageMaker Python SDK code."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2 import modifiers

FUNCTION_CALL_MODIFIERS = [
    modifiers.renamed_params.EstimatorImageURIRenamer(),
    modifiers.renamed_params.ModelImageURIRenamer(),
    modifiers.framework_version.FrameworkVersionEnforcer(),
    modifiers.tf_legacy_mode.TensorFlowLegacyModeConstructorUpgrader(),
    modifiers.tf_legacy_mode.TensorBoardParameterRemover(),
    modifiers.deprecated_params.TensorFlowScriptModeParameterRemover(),
    modifiers.tfs.TensorFlowServingConstructorRenamer(),
    modifiers.predictors.PredictorConstructorRefactor(),
    modifiers.airflow.ModelConfigArgModifier(),
    modifiers.airflow.ModelConfigImageURIRenamer(),
    modifiers.renamed_params.DistributionParameterRenamer(),
    modifiers.renamed_params.S3SessionRenamer(),
    modifiers.renamed_params.EstimatorCreateModelImageURIRenamer(),
    modifiers.renamed_params.SessionCreateModelImageURIRenamer(),
    modifiers.renamed_params.SessionCreateEndpointImageURIRenamer(),
    modifiers.training_params.TrainPrefixRemover(),
    modifiers.training_input.TrainingInputConstructorRefactor(),
    modifiers.training_input.ShuffleConfigModuleRenamer(),
    modifiers.serde.SerdeConstructorRenamer(),
    modifiers.serde.SerdeKeywordRemover(),
    modifiers.image_uris.ImageURIRetrieveRefactor(),
]

IMPORT_MODIFIERS = [modifiers.tfs.TensorFlowServingImportRenamer()]

NAME_MODIFIERS = [modifiers.serde.SerdeObjectRenamer()]

MODULE_MODIFIERS = [
    modifiers.serde.SerializerImportInserter(),
    modifiers.serde.DeserializerImportInserter(),
]

IMPORT_FROM_MODIFIERS = [
    modifiers.predictors.PredictorImportFromRenamer(),
    modifiers.tfs.TensorFlowServingImportFromRenamer(),
    modifiers.training_input.TrainingInputImportFromRenamer(),
    modifiers.training_input.ShuffleConfigImportFromRenamer(),
    modifiers.serde.SerdeImportFromAmazonCommonRenamer(),
    modifiers.serde.SerdeImportFromPredictorRenamer(),
    modifiers.image_uris.ImageURIRetrieveImportFromRenamer(),
]


class ASTTransformer(ast.NodeTransformer):
    """An ``ast.NodeTransformer`` subclass that walks the abstract syntax tree.

    It modifies nodes to upgrade the given SageMaker Python SDK code.
    """

    def visit_Call(self, node):
        """Visits an ``ast.Call`` node and returns a modified node or None.

        See https://docs.python.org/3/library/ast.html#ast.NodeTransformer.

        Args:
            node (ast.Call): a node that represents a function call.

        Returns:
            ast.AST: if the returned node is None, the original node is removed
                from its location. Otherwise, the original node is replaced with
                the returned node.
        """
        for function_checker in FUNCTION_CALL_MODIFIERS:
            node = function_checker.check_and_modify_node(node)
        return ast.fix_missing_locations(node) if node else None

    def visit_Name(self, node):
        """Visits an ``ast.Name`` node and returns a modified node or None.

        See https://docs.python.org/3/library/ast.html#ast.NodeTransformer.

        Args:
            node (ast.Name): a node that represents an identifier.

        Returns:
            ast.AST: if the returned node is None, the original node is removed
                from its location. Otherwise, the original node is replaced with
                the returned node.
        """
        for name_checker in NAME_MODIFIERS:
            node = name_checker.check_and_modify_node(node)
        return ast.fix_missing_locations(node) if node else None

    def visit_Import(self, node):
        """Visits an ``ast.Import`` node and returns a modified node or None.

        See https://docs.python.org/3/library/ast.html#ast.NodeTransformer.

        Args:
            node (ast.Import): a node that represents an import statement.

        Returns:
            ast.AST: if the returned node is None, the original node is removed
                from its location. Otherwise, the original node is replaced with
                the returned node.
        """
        for import_checker in IMPORT_MODIFIERS:
            node = import_checker.check_and_modify_node(node)
        return ast.fix_missing_locations(node) if node else None

    def visit_Module(self, node):
        """Visits an ``ast.Module`` node and returns a modified node or None.

        See https://docs.python.org/3/library/ast.html#ast.NodeTransformer.

        The ``ast.NodeTransformer`` walks the abstract syntax tree and modifies
        all other nodes before modifying the ``ast.Module`` node.

        Args:
            node (ast.Module): a node that represents a Python module.

        Returns:
            ast.AST: if the returned node is None, the original node is removed
                from its location. Otherwise, the original node is replaced with
                the returned node.
        """
        self.generic_visit(node)
        for module_checker in MODULE_MODIFIERS:
            node = module_checker.check_and_modify_node(node)
        return ast.fix_missing_locations(node) if node else None

    def visit_ImportFrom(self, node):
        """Visits an ``ast.ImportFrom`` node and returns a modified node or None.

        See https://docs.python.org/3/library/ast.html#ast.NodeTransformer.

        Args:
            node (ast.ImportFrom): a node that represents an import statement.

        Returns:
            ast.AST: if the returned node is None, the original node is removed
                from its location. Otherwise, the original node is replaced with
                the returned node.
        """
        for import_checker in IMPORT_FROM_MODIFIERS:
            node = import_checker.check_and_modify_node(node)
        return ast.fix_missing_locations(node) if node else None
