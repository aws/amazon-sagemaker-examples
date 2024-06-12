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
"""Classes to handle training renames for version 2.0 and later of the SageMaker Python SDK."""
from __future__ import absolute_import

from sagemaker.cli.compatibility.v2.modifiers import matching
from sagemaker.cli.compatibility.v2.modifiers.modifier import Modifier

ESTIMATORS = {
    "AlgorithmEstimator": ("sagemaker", "sagemaker.algorithm"),
    "AmazonAlgorithmEstimatorBase": ("sagemaker.amazon.amazon_estimator",),
    "Chainer": ("sagemaker.chainer", "sagemaker.chainer.estimator"),
    "Estimator": ("sagemaker.estimator",),
    "EstimatorBase": ("sagemaker.estimator",),
    "FactorizationMachines": ("sagemaker", "sagemaker.amazon.factorization_machines"),
    "Framework": ("sagemaker.estimator",),
    "IPInsights": ("sagemaker", "sagemaker.amazon.ipinsights"),
    "KMeans": ("sagemaker", "sagemaker.amazon.kmeans"),
    "KNN": ("sagemaker", "sagemaker.amazon.knn"),
    "LDA": ("sagemaker", "sagemaker.amazon.lda"),
    "LinearLearner": ("sagemaker", "sagemaker.amazon.linear_learner"),
    "MXNet": ("sagemaker.mxnet", "sagemaker.mxnet.estimator"),
    "NTM": ("sagemaker", "sagemaker.amazon.ntm"),
    "Object2Vec": ("sagemaker", "sagemaker.amazon.object2vec"),
    "PCA": ("sagemaker", "sagemaker.amazon.pca"),
    "PyTorch": ("sagemaker.pytorch", "sagemaker.pytorch.estimator"),
    "RandomCutForest": ("sagemaker", "sagemaker.amazon.randomcutforest"),
    "RLEstimator": ("sagemaker.rl", "sagemaker.rl.estimator"),
    "SKLearn": ("sagemaker.sklearn", "sagemaker.sklearn.estimator"),
    "TensorFlow": ("sagemaker.tensorflow", "sagemaker.tensorflow.estimator"),
    "XGBoost": ("sagemaker.xgboost", "sagemaker.xgboost.estimator"),
}

PARAMS = (
    "train_instance_count",
    "train_instance_type",
    "train_max_run",
    "train_max_wait",
    "train_use_spot_instances",
    "train_volume_size",
    "train_volume_kms_key",
)


class TrainPrefixRemover(Modifier):
    """A class to remove the redundant 'train' prefix in estimator parameters."""

    def node_should_be_modified(self, node):
        """Checks if the node is an estimator constructor and contains any relevant parameters.

        This looks for the following parameters:

        - ``train_instance_count``
        - ``train_instance_type``
        - ``train_max_run``
        - ``train_max_wait``
        - ``train_use_spot_instances``
        - ``train_volume_kms_key``
        - ``train_volume_size``

        Args:
            node (ast.Call): a node that represents a function call. For more,
                see https://docs.python.org/3/library/ast.html#abstract-grammar.

        Returns:
            bool: If the ``ast.Call`` matches the relevant function calls and
                contains the parameter to be renamed.
        """
        return matching.matches_any(node, ESTIMATORS) and self._has_train_parameter(node)

    def _has_train_parameter(self, node):
        """Checks if at least one of the node's keywords is prefixed with 'train'."""
        for kw in node.keywords:
            if kw.arg in PARAMS:
                return True

        return False

    def modify_node(self, node):
        """Modifies the ``ast.Call`` node to remove the 'train' prefix from its keywords.

        Args:
            node (ast.Call): a node that represents an estimator constructor.

        Returns:
            ast.AST: the original node, which has been potentially modified.
        """
        for kw in node.keywords:
            if kw.arg in PARAMS:
                kw.arg = kw.arg.replace("train_", "")
        return node
