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
"""Classes to handle renames for version 2.0 and later of the SageMaker Python SDK."""
from __future__ import absolute_import

import ast
from abc import abstractmethod

from sagemaker.cli.compatibility.v2.modifiers import matching, parsing
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier


class ParamRenamer(Modifier):
    """Abstract class to take in an AST node to check if it needs to be renamed.

    It checks if AST node is a function call with
    an argument that needs to be renamed, and rename the argument if needed.
    """

    @property
    @abstractmethod
    def calls_to_modify(self):
        """A dictionary mapping function names to possible namespaces."""

    @property
    @abstractmethod
    def old_param_name(self):
        """The parameter name used in previous versions of the SageMaker Python SDK."""

    @property
    @abstractmethod
    def new_param_name(self):
        """The parameter name used in version 2.0 and later of the SageMaker Python SDK."""

    def node_should_be_modified(self, node):
        """Checks node for matches and also contains certain parameter, returns boolean.

        If the node matches any of the relevant functions and
        contains the parameter to be renamed.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` matches the relevant function calls and
                contains the parameter to be renamed.
        """
        return matching.matches_any(node, self.calls_to_modify) and matching.has_arg(
            node, self.old_param_name
        )

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to rename the attribute.

        Args:
            node (ast.Call): a node that represents the relevant function call.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        keyword = parsing.arg_from_keywords(node, self.old_param_name)
        keyword.arg = self.new_param_name
        return node


class MethodParamRenamer(ParamRenamer):
    """Abstract class to handle parameter renames for methods that belong to objects.

    This differs from ``ParamRenamer`` in that a node for a standalone function call
    (i.e. where ``node.func`` is an ``ast.Name`` rather than an ``ast.Attribute``) is not modified.
    """

    def node_should_be_modified(self, node):
        """Checks node for matches and also contains certain parameter, returns boolean.

        If the ``ast.Call`` node matches any of the relevant functions and
        contains the parameter to be renamed returns True.

        This looks for a call of the form ``<object>.<method>``, and
        assumes the method cannot be called on its own.

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` matches the relevant function calls and
                contains the parameter to be renamed.
        """
        if isinstance(node.func, ast.Name):
            return False

        return super(MethodParamRenamer, self).node_should_be_modified(node)


class DistributionParameterRenamer(ParamRenamer):
    """A class to rename the ``distributions`` attribute to ``distrbution``.

     Rename takes place in MXNet and TensorFlow estimators.

    This looks for the following calls:

    - ``<Framework>``
    - ``sagemaker.<framework>.<Framework>``
    - ``sagemaker.<framework>.estimator.<Framework>``

    where ``<Framework>`` is either ``TensorFlow`` or ``MXNet``.
    """

    @property
    def calls_to_modify(self):
        """A dictionary mapping ``MXNet`` and ``TensorFlow`` to their respective namespaces."""
        return {
            "TensorFlow": ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator"),
            "MXNet": ("sagemaker.mxnet", "sagemaker.mxnet.estimator"),
        }

    @property
    def old_param_name(self):
        """The previous name for the distribution argument."""
        return "distributions"

    @property
    def new_param_name(self):
        """The new name for the distribution argument."""
        return "distribution"


class S3SessionRenamer(MethodParamRenamer):
    """A class to rename the ``session`` attribute to ``sagemaker_session``.

    Rename takes place in ``S3Uploader`` and ``S3Downloader``.

    This looks for the following calls:

    - ``sagemaker.s3.S3Uploader.<function>``
    - ``s3.S3Uploader.<function>``
    - ``S3Uploader.<function>``

    where ``S3Uploader`` is either ``S3Uploader`` or ``S3Downloader``, and where
    ``<function>`` is any of the functions belonging to those two classes.
    """

    @property
    def calls_to_modify(self):
        """A dictionary mapping S3 utility functions to their respective namespaces."""
        return {
            "download": ("sagemaker.s3.S3Downloader", "s3.S3Downloader", "S3Downloader"),
            "list": ("sagemaker.s3.S3Downloader", "s3.S3Downloader", "S3Downloader"),
            "read_file": ("sagemaker.s3.S3Downloader", "s3.S3Downloader", "S3Downloader"),
            "upload": ("sagemaker.s3.S3Uploader", "s3.S3Uploader", "S3Uploader"),
            "upload_string_as_file_body": (
                "sagemaker.s3.S3Uploader",
                "s3.S3Uploader",
                "S3Uploader",
            ),
        }

    @property
    def old_param_name(self):
        """The previous name for the SageMaker session argument."""
        return "session"

    @property
    def new_param_name(self):
        """The new name for the SageMaker session argument."""
        return "sagemaker_session"


class EstimatorImageURIRenamer(ParamRenamer):
    """A class to rename the ``image_name`` attribute to ``image_uri`` in estimators."""

    @property
    def calls_to_modify(self):
        """A dictionary mapping estimators with the ``image_name`` attribute.

        It maps to their respective namespaces.
        """
        return {
            "Chainer": ("sagemaker.chainer", "sagemaker.chainer.estimator"),
            "Estimator": ("sagemaker.estimator",),
            "Framework": ("sagemaker.estimator",),
            "MXNet": ("sagemaker.mxnet", "sagemaker.mxnet.estimator"),
            "PyTorch": ("sagemaker.pytorch", "sagemaker.pytorch.estimator"),
            "RLEstimator": ("sagemaker.rl", "sagemaker.rl.estimator"),
            "SKLearn": ("sagemaker.sklearn", "sagemaker.sklearn.estimator"),
            "TensorFlow": ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator"),
            "XGBoost": ("sagemaker.xgboost", "sagemaker.xgboost.estimator"),
        }

    @property
    def old_param_name(self):
        """The previous name for the image URI argument."""
        return "image_name"

    @property
    def new_param_name(self):
        """The new name for the image URI argument."""
        return "image_uri"


class ModelImageURIRenamer(ParamRenamer):
    """A class to rename the ``image`` attribute to ``image_uri`` in models."""

    @property
    def calls_to_modify(self):
        """A dictionary mapping models with the ``image`` attribute.

        It maps to their respective namespaces.
        """
        return {
            "ChainerModel": ("sagemaker.chainer", "sagemaker.chainer.model"),
            "Model": ("sagemaker.model",),
            "MultiDataModel": ("sagemaker.multidatamodel",),
            "FrameworkModel": ("sagemaker.model",),
            "MXNetModel": ("sagemaker.mxnet", "sagemaker.mxnet.model"),
            "PyTorchModel": ("sagemaker.pytorch", "sagemaker.pytorch.model"),
            "SKLearnModel": ("sagemaker.sklearn", "sagemaker.sklearn.model"),
            "TensorFlowModel": ("sagemaker.tensorflow", "sagemaker.tensorflow.model"),
            "XGBoostModel": ("sagemaker.xgboost", "sagemaker.xgboost.model"),
        }

    @property
    def old_param_name(self):
        """The previous name for the image URI argument."""
        return "image"

    @property
    def new_param_name(self):
        """The new name for the image URI argument."""
        return "image_uri"


class EstimatorCreateModelImageURIRenamer(MethodParamRenamer):
    """A class to rename ``image`` to ``image_uri`` in estimator ``create_model()`` methods."""

    @property
    def calls_to_modify(self):
        """A mapping of ``create_model`` to common variable names for estimators."""
        return {
            "create_model": (
                "estimator",
                "chainer",
                "mxnet",
                "mx",
                "pytorch",
                "rl",
                "sklearn",
                "tensorflow",
                "tf",
                "xgboost",
                "xgb",
            )
        }

    @property
    def old_param_name(self):
        """The previous name for the image URI argument."""
        return "image"

    @property
    def new_param_name(self):
        """The new name for the the image URI argument."""
        return "image_uri"


class SessionCreateModelImageURIRenamer(MethodParamRenamer):
    """A class to rename ``primary_container_image`` to ``image_uri``.

    This looks for the following calls:

    - ``sagemaker_session.create_model_from_job()``
    - ``sess.create_model_from_job()``
    """

    @property
    def calls_to_modify(self):
        """A mapping of ``create_model_from_job`` to common variable names for Session."""
        return {
            "create_model_from_job": ("sagemaker_session", "sess"),
        }

    @property
    def old_param_name(self):
        """The previous name for the image URI argument."""
        return "primary_container_image"

    @property
    def new_param_name(self):
        """The new name for the the image URI argument."""
        return "image_uri"


class SessionCreateEndpointImageURIRenamer(MethodParamRenamer):
    """A class to rename ``deployment_image`` to ``image_uri``.

    This looks for the following calls:

    - ``sagemaker_session.endpoint_from_job()``
    - ``sess.endpoint_from_job()``
    - ``sagemaker_session.endpoint_from_model_data()``
    - ``sess.endpoint_from_model_data()``
    """

    @property
    def calls_to_modify(self):
        """A mapping of the ``endpoint_from_*`` functions to common variable names for Session."""
        return {
            "endpoint_from_job": ("sagemaker_session", "sess"),
            "endpoint_from_model_data": ("sagemaker_session", "sess"),
        }

    @property
    def old_param_name(self):
        """The previous name for the image URI argument."""
        return "deployment_image"

    @property
    def new_param_name(self):
        """The new name for the the image URI argument."""
        return "image_uri"
