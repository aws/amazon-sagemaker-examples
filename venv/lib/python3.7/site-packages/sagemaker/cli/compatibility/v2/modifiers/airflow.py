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
"""A class to handle argument changes for Airflow functions."""
from __future__ import absolute_import

import ast

from sagemaker.cli.compatibility.v2.modifiers import matching, renamed_params
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

FUNCTION_NAMES = ("model_config", "model_config_from_estimator")
NAMESPACES = ("sagemaker.workflow.airflow", "workflow.airflow", "airflow")
FUNCTIONS = {name: NAMESPACES for name in FUNCTION_NAMES}


class ModelConfigArgModifier(Modifier):
    """A class to handle argument changes for Airflow model config functions."""

    def node_should_be_modified(self, node):
        """Function to check Airflow model config and if it contains positional arguments.

        Checks if the ``ast.Call`` node creates an Airflow model config and
        contains positional arguments. This looks for the following formats:

        - ``model_config``
        - ``airflow.model_config``
        - ``workflow.airflow.model_config``
        - ``sagemaker.workflow.airflow.model_config``

        where ``model_config`` is either ``model_config`` or ``model_config_from_estimator``.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is either a ``model_config`` call or
                a ``model_config_from_estimator`` call and has positional arguments.
        """
        return matching.matches_any(node, FUNCTIONS) and len(node.args) > 0

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node's arguments.

        The first argument, the instance type, is turned into a keyword arg,
        leaving the second argument, the model, to be the first argument.

        Args:
            node (ast.Call): a node that represents either a ``model_config`` call or
                a ``model_config_from_estimator`` call.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        instance_type = node.args.pop(0)
        node.keywords.append(ast.keyword(arg="instance_type", value=instance_type))
        return node


class ModelConfigImageURIRenamer(renamed_params.ParamRenamer):
    """A class to rename the ``image`` attribute to ``image_uri`` in Airflow model config functions.

    This looks for the following formats:

    - ``model_config``
    - ``airflow.model_config``
    - ``workflow.airflow.model_config``
    - ``sagemaker.workflow.airflow.model_config``

    where ``model_config`` is either ``model_config`` or ``model_config_from_estimator``.
    """

    @property
    def calls_to_modify(self):
        """A dictionary mapping Airflow model config functions to their respective namespaces."""
        return FUNCTIONS

    @property
    def old_param_name(self):
        """The previous name for the image URI argument."""
        return "image"

    @property
    def new_param_name(self):
        """The new name for the image URI argument."""
        return "image_uri"
