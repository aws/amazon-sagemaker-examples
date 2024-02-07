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
"""A class to ensure that ``framework_version`` is defined when constructing framework classes."""
from __future__ import absolute_import

import ast

from packaging.version import InvalidVersion, Version

from sagemaker.cli.compatibility.v2.modifiers import matching, parsing
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

FRAMEWORK_ARG = "framework_version"
IMAGE_ARG = "image_uri"
PY_ARG = "py_version"

FRAMEWORK_DEFAULTS = {
    "Chainer": "4.1.0",
    "MXNet": "1.2.0",
    "PyTorch": "0.4.0",
    "SKLearn": "0.20.0",
    "TensorFlow": "1.11.0",
}

FRAMEWORK_CLASSES = list(FRAMEWORK_DEFAULTS.keys())

ESTIMATORS = {
    fw: ("sagemaker.{}".format(fw.lower()), "sagemaker.{}.estimator".format(fw.lower()))
    for fw in FRAMEWORK_CLASSES
}
# TODO: check for sagemaker.tensorflow.serving.Model
MODELS = {
    "{}Model".format(fw): (
        "sagemaker.{}".format(fw.lower()),
        "sagemaker.{}.model".format(fw.lower()),
    )
    for fw in FRAMEWORK_CLASSES
}


class FrameworkVersionEnforcer(Modifier):
    """Ensures that ``framework_version`` is defined when instantiating a framework estimator."""

    def node_should_be_modified(self, node):
        """Checks if the ast.Call node instantiates a framework estimator or model.

        It doesn't specify the ``framework_version`` and ``py_version`` parameter,
        as appropriate.

        This looks for the following formats:

        - ``TensorFlow``
        - ``sagemaker.tensorflow.TensorFlow``

        where "TensorFlow" can be Chainer, MXNet, PyTorch, SKLearn, or TensorFlow.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` is instantiating a framework class that
                should specify ``framework_version``, but doesn't.
        """
        if matching.matches_any(node, ESTIMATORS) or matching.matches_any(node, MODELS):
            return _version_args_needed(node)

        return False

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node's keywords to include ``framework_version``.

        The ``framework_version`` value is determined by the framework:

        - Chainer: "4.1.0"
        - MXNet: "1.2.0"
        - PyTorch: "0.4.0"
        - SKLearn: "0.20.0"
        - TensorFlow: "1.11.0"

        The ``py_version`` value is determined by the framework, framework_version, and if it is a
        model, whether the model accepts a py_version

        Args:
            node (ast.Call): a node that represents the constructor of a framework class.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        framework, is_model = _framework_from_node(node)

        # if framework_version is not supplied, get default and append keyword
        if matching.has_arg(node, FRAMEWORK_ARG):
            framework_version = parsing.arg_value(node, FRAMEWORK_ARG)
        else:
            framework_version = FRAMEWORK_DEFAULTS[framework]
            node.keywords.append(ast.keyword(arg=FRAMEWORK_ARG, value=ast.Str(s=framework_version)))

        # if py_version is not supplied, get a conditional default, and if not None, append keyword
        if not matching.has_arg(node, PY_ARG):
            py_version = _py_version_defaults(framework, framework_version, is_model)
            if py_version:
                node.keywords.append(ast.keyword(arg=PY_ARG, value=ast.Str(s=py_version)))
        return node


def _py_version_defaults(framework, framework_version, is_model=False):
    """Gets the py_version required for the framework_version and if it's a model

    Args:
        framework (str): name of the framework
        framework_version (str): version of the framework
        is_model (bool): whether it is a constructor for a model or not

    Returns:
        str: the default py version, as appropriate. None if no default py_version
    """
    if framework in ("Chainer", "PyTorch"):
        return "py3"
    if framework == "SKLearn" and not is_model:
        return "py3"
    if framework == "MXNet":
        return "py2"
    if framework == "TensorFlow" and not is_model:
        return _tf_py_version_default(framework_version)
    return None


def _tf_py_version_default(framework_version):
    """Gets the py_version default based on framework_version for TensorFlow."""
    if not framework_version:
        return "py2"

    try:
        version = Version(framework_version)
    except InvalidVersion:
        return "py2"

    if version < Version("1.12"):
        return "py2"
    if version < Version("2.2"):
        return "py3"
    return "py37"


def _framework_from_node(node):
    """Retrieves the framework class name based on the function call, and if it was a model

    Args:
        node (ast.Call): a node that represents the constructor of a framework class.
            This can represent either <Framework> or sagemaker.<framework>.<Framework>.

    Returns:
        str, bool: the (capitalized) framework class name, and if it is a model class
    """
    if isinstance(node.func, ast.Name):
        framework = node.func.id
    elif isinstance(node.func, ast.Attribute):
        framework = node.func.attr
    else:
        framework = ""

    is_model = framework.endswith("Model")
    if is_model:
        framework = framework[: framework.find("Model")]

    return framework, is_model


def _version_args_needed(node):
    """Determines if image_arg or version_arg was supplied

    Applies similar logic as ``validate_version_or_image_args``
    """
    # if image_arg is present, no need to supply version arguments
    if matching.has_arg(node, IMAGE_ARG):
        return False

    # if framework_version is None, need args
    if matching.has_arg(node, FRAMEWORK_ARG):
        framework_version = parsing.arg_value(node, FRAMEWORK_ARG)
    else:
        return True

    # check if we expect py_version and we don't get it -- framework and model dependent
    framework, is_model = _framework_from_node(node)
    expecting_py_version = _py_version_defaults(framework, framework_version, is_model)
    if expecting_py_version:
        return not matching.has_arg(node, PY_ARG)

    return False
