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
"""Placeholder docstring"""
from __future__ import absolute_import

from typing import Union, Optional

from sagemaker import image_uris
from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import ge, le, isin
from sagemaker.predictor import Predictor
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.utils import pop_out_unused_kwarg
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable


def _list_check_subset(valid_super_list):
    """Provides a function to check validity of list subset.

    Args:
        valid_super_list:
    """
    valid_superset = set(valid_super_list)

    def validate(value):
        if not isinstance(value, str):
            return False

        val_list = [s.strip() for s in value.split(",")]
        return set(val_list).issubset(valid_superset)

    return validate


class Object2Vec(AmazonAlgorithmEstimatorBase):
    """A general-purpose neural embedding algorithm that is highly customizable.

    It can learn low-dimensional dense embeddings of high-dimensional objects. The embeddings
    are learned in a way that preserves the semantics of the relationship between pairs of
    objects in the original space in the embedding space.
    """

    repo_name: str = "object2vec"
    repo_version: str = "1"
    MINI_BATCH_SIZE: int = 32

    enc_dim: hp = hp("enc_dim", (ge(4), le(10000)), "An integer in [4, 10000]", int)
    mini_batch_size: hp = hp("mini_batch_size", (ge(1), le(10000)), "An integer in [1, 10000]", int)
    epochs: hp = hp("epochs", (ge(1), le(100)), "An integer in [1, 100]", int)
    early_stopping_patience: hp = hp(
        "early_stopping_patience", (ge(1), le(5)), "An integer in [1, 5]", int
    )
    early_stopping_tolerance: hp = hp(
        "early_stopping_tolerance", (ge(1e-06), le(0.1)), "A float in [1e-06, 0.1]", float
    )
    dropout: hp = hp("dropout", (ge(0.0), le(1.0)), "A float in [0.0, 1.0]", float)
    weight_decay: hp = hp(
        "weight_decay", (ge(0.0), le(10000.0)), "A float in [0.0, 10000.0]", float
    )
    bucket_width: hp = hp("bucket_width", (ge(0), le(100)), "An integer in [0, 100]", int)
    num_classes: hp = hp("num_classes", (ge(2), le(30)), "An integer in [2, 30]", int)
    mlp_layers: hp = hp("mlp_layers", (ge(1), le(10)), "An integer in [1, 10]", int)
    mlp_dim: hp = hp("mlp_dim", (ge(2), le(10000)), "An integer in [2, 10000]", int)
    mlp_activation: hp = hp(
        "mlp_activation", isin("tanh", "relu", "linear"), 'One of "tanh", "relu", "linear"', str
    )
    output_layer: hp = hp(
        "output_layer",
        isin("softmax", "mean_squared_error"),
        'One of "softmax", "mean_squared_error"',
        str,
    )
    optimizer: hp = hp(
        "optimizer",
        isin("adagrad", "adam", "rmsprop", "sgd", "adadelta"),
        'One of "adagrad", "adam", "rmsprop", "sgd", "adadelta"',
        str,
    )
    learning_rate: hp = hp("learning_rate", (ge(1e-06), le(1.0)), "A float in [1e-06, 1.0]", float)

    negative_sampling_rate: hp = hp(
        "negative_sampling_rate", (ge(0), le(100)), "An integer in [0, 100]", int
    )
    comparator_list: hp = hp(
        "comparator_list",
        _list_check_subset(["hadamard", "concat", "abs_diff"]),
        'Comma-separated of hadamard, concat, abs_diff. E.g. "hadamard,abs_diff"',
        str,
    )
    tied_token_embedding_weight: hp = hp(
        "tied_token_embedding_weight", (), "Either True or False", bool
    )
    token_embedding_storage_type: hp = hp(
        "token_embedding_storage_type",
        isin("dense", "row_sparse"),
        'One of "dense", "row_sparse"',
        str,
    )

    enc0_network: hp = hp(
        "enc0_network",
        isin("hcnn", "bilstm", "pooled_embedding"),
        'One of "hcnn", "bilstm", "pooled_embedding"',
        str,
    )
    enc1_network: hp = hp(
        "enc1_network",
        isin("hcnn", "bilstm", "pooled_embedding", "enc0"),
        'One of "hcnn", "bilstm", "pooled_embedding", "enc0"',
        str,
    )
    enc0_cnn_filter_width: hp = hp(
        "enc0_cnn_filter_width", (ge(1), le(9)), "An integer in [1, 9]", int
    )
    enc1_cnn_filter_width: hp = hp(
        "enc1_cnn_filter_width", (ge(1), le(9)), "An integer in [1, 9]", int
    )
    enc0_max_seq_len: hp = hp("enc0_max_seq_len", (ge(1), le(5000)), "An integer in [1, 5000]", int)
    enc1_max_seq_len: hp = hp("enc1_max_seq_len", (ge(1), le(5000)), "An integer in [1, 5000]", int)
    enc0_token_embedding_dim: hp = hp(
        "enc0_token_embedding_dim", (ge(2), le(1000)), "An integer in [2, 1000]", int
    )
    enc1_token_embedding_dim: hp = hp(
        "enc1_token_embedding_dim", (ge(2), le(1000)), "An integer in [2, 1000]", int
    )
    enc0_vocab_size: hp = hp(
        "enc0_vocab_size", (ge(2), le(3000000)), "An integer in [2, 3000000]", int
    )
    enc1_vocab_size: hp = hp(
        "enc1_vocab_size", (ge(2), le(3000000)), "An integer in [2, 3000000]", int
    )
    enc0_layers: hp = hp("enc0_layers", (ge(1), le(4)), "An integer in [1, 4]", int)
    enc1_layers: hp = hp("enc1_layers", (ge(1), le(4)), "An integer in [1, 4]", int)
    enc0_freeze_pretrained_embedding: hp = hp(
        "enc0_freeze_pretrained_embedding", (), "Either True or False", bool
    )
    enc1_freeze_pretrained_embedding: hp = hp(
        "enc1_freeze_pretrained_embedding", (), "Either True or False", bool
    )

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        epochs: Optional[int] = None,
        enc0_max_seq_len: Optional[int] = None,
        enc0_vocab_size: Optional[int] = None,
        enc_dim: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_tolerance: Optional[float] = None,
        dropout: Optional[float] = None,
        weight_decay: Optional[float] = None,
        bucket_width: Optional[int] = None,
        num_classes: Optional[int] = None,
        mlp_layers: Optional[int] = None,
        mlp_dim: Optional[int] = None,
        mlp_activation: Optional[str] = None,
        output_layer: Optional[str] = None,
        optimizer: Optional[str] = None,
        learning_rate: Optional[float] = None,
        negative_sampling_rate: Optional[int] = None,
        comparator_list: Optional[str] = None,
        tied_token_embedding_weight: Optional[bool] = None,
        token_embedding_storage_type: Optional[str] = None,
        enc0_network: Optional[str] = None,
        enc1_network: Optional[str] = None,
        enc0_cnn_filter_width: Optional[int] = None,
        enc1_cnn_filter_width: Optional[int] = None,
        enc1_max_seq_len: Optional[int] = None,
        enc0_token_embedding_dim: Optional[int] = None,
        enc1_token_embedding_dim: Optional[int] = None,
        enc1_vocab_size: Optional[int] = None,
        enc0_layers: Optional[int] = None,
        enc1_layers: Optional[int] = None,
        enc0_freeze_pretrained_embedding: Optional[bool] = None,
        enc1_freeze_pretrained_embedding: Optional[bool] = None,
        **kwargs
    ):
        """Object2Vec is :class:`Estimator` used for anomaly detection.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
        There is an utility
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.record_set`
        that can be used to upload data to S3 and creates
        :class:`~sagemaker.amazon.amazon_estimator.RecordSet` to be passed to
        the `fit` call.

        After this Estimator is fit, model data is stored in S3. The model
        may be deployed to an Amazon SageMaker Endpoint by invoking
        :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
        deploying an Endpoint, deploy returns a
        :class:`~sagemaker.amazon.Predictor` object that can be used for
        inference calls using the trained model hosted in the SageMaker
        Endpoint.

        Object2Vec Estimators can be configured by setting hyperparameters.
        The available hyperparameters for Object2Vec are documented below.

        For further information on the AWS Object2Vec algorithm, please
        consult AWS technical documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/object2vec.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if accessing AWS resource.
            instance_count (int or PipelineVariable): Number of Amazon EC2 instances to use
                for training.
            instance_type (str or PipelineVariable): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            epochs (int): Total number of epochs for SGD training
            enc0_max_seq_len (int): Maximum sequence length
            enc0_vocab_size (int): Vocabulary size of tokens
            enc_dim (int): Optional. Dimension of the output of the embedding
                layer
            mini_batch_size (int): Optional. mini batch size for SGD training
            early_stopping_patience (int): Optional. The allowed number of
                consecutive epochs without improvement before early stopping is
                applied
            early_stopping_tolerance (float): Optional. The value used to
                determine whether the algorithm has made improvement between two
                consecutive epochs for early stopping
            dropout (float): Optional. Dropout probability on network layers
            weight_decay (float): Optional. Weight decay parameter during
                optimization
            bucket_width (int): Optional. The allowed difference between data
                sequence length when bucketing is enabled
            num_classes (int): Optional. Number of classes for classification
                training (ignored for regression problems)
            mlp_layers (int): Optional. Number of MLP layers in the network
            mlp_dim (int): Optional. Dimension of the output of MLP layer
            mlp_activation (str): Optional. Type of activation function for the
                MLP layer
            output_layer (str): Optional. Type of output layer
            optimizer (str): Optional. Type of optimizer for training
            learning_rate (float): Optional. Learning rate for SGD training
            negative_sampling_rate (int): Optional. Negative sampling rate
            comparator_list (str): Optional. Customization of comparator
                operator
            tied_token_embedding_weight (bool): Optional. Tying of token
                embedding layer weight
            token_embedding_storage_type (str): Optional. Type of token
                embedding storage
            enc0_network (str): Optional. Network model of encoder "enc0"
            enc1_network (str): Optional. Network model of encoder "enc1"
            enc0_cnn_filter_width (int): Optional. CNN filter width
            enc1_cnn_filter_width (int): Optional. CNN filter width
            enc1_max_seq_len (int): Optional. Maximum sequence length
            enc0_token_embedding_dim (int): Optional. Output dimension of token
                embedding layer
            enc1_token_embedding_dim (int): Optional. Output dimension of token
                embedding layer
            enc1_vocab_size (int): Optional. Vocabulary size of tokens
            enc0_layers (int): Optional. Number of layers in encoder
            enc1_layers (int): Optional. Number of layers in encoder
            enc0_freeze_pretrained_embedding (bool): Optional. Freeze pretrained
                embedding weights
            enc1_freeze_pretrained_embedding (bool): Optional. Freeze pretrained
                embedding weights
            **kwargs: base class keyword argument values.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.amazon_estimator.AmazonAlgorithmEstimatorBase` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """

        super(Object2Vec, self).__init__(role, instance_count, instance_type, **kwargs)
        self.enc_dim = enc_dim
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tolerance = early_stopping_tolerance
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.bucket_width = bucket_width
        self.num_classes = num_classes
        self.mlp_layers = mlp_layers
        self.mlp_dim = mlp_dim
        self.mlp_activation = mlp_activation
        self.output_layer = output_layer
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.negative_sampling_rate = negative_sampling_rate
        self.comparator_list = comparator_list
        self.tied_token_embedding_weight = tied_token_embedding_weight
        self.token_embedding_storage_type = token_embedding_storage_type

        self.enc0_network = enc0_network
        self.enc1_network = enc1_network
        self.enc0_cnn_filter_width = enc0_cnn_filter_width
        self.enc1_cnn_filter_width = enc1_cnn_filter_width
        self.enc0_max_seq_len = enc0_max_seq_len
        self.enc1_max_seq_len = enc1_max_seq_len
        self.enc0_token_embedding_dim = enc0_token_embedding_dim
        self.enc1_token_embedding_dim = enc1_token_embedding_dim
        self.enc0_vocab_size = enc0_vocab_size
        self.enc1_vocab_size = enc1_vocab_size
        self.enc0_layers = enc0_layers
        self.enc1_layers = enc1_layers
        self.enc0_freeze_pretrained_embedding = enc0_freeze_pretrained_embedding
        self.enc1_freeze_pretrained_embedding = enc1_freeze_pretrained_embedding

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT, **kwargs):
        """Return a :class:`~sagemaker.amazon.Object2VecModel`.

        It references the latest s3 model data produced by this Estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional kwargs passed to the Object2VecModel constructor.
        """
        return Object2VecModel(
            self.model_data,
            self.role,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            **kwargs
        )

    def _prepare_for_training(self, records, mini_batch_size=None, job_name=None):
        """Placeholder docstring"""
        if mini_batch_size is None:
            mini_batch_size = self.MINI_BATCH_SIZE

        super(Object2Vec, self)._prepare_for_training(
            records, mini_batch_size=mini_batch_size, job_name=job_name
        )


class Object2VecModel(Model):
    """Reference Object2Vec s3 model data.

    Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and returns a
    Predictor that calculates anomaly scores for datapoints.
    """

    def __init__(
        self,
        model_data: Union[str, PipelineVariable],
        role: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        **kwargs
    ):
        """Initialization for Object2VecModel class.

        Args:
            model_data (str or PipelineVariable): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            **kwargs: Keyword arguments passed to the ``FrameworkModel``
                initializer.
        """
        sagemaker_session = sagemaker_session or Session()
        image_uri = image_uris.retrieve(
            Object2Vec.repo_name,
            sagemaker_session.boto_region_name,
            version=Object2Vec.repo_version,
        )
        pop_out_unused_kwarg("predictor_cls", kwargs, Predictor.__name__)
        pop_out_unused_kwarg("image_uri", kwargs, image_uri)
        super(Object2VecModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=Predictor,
            sagemaker_session=sagemaker_session,
            **kwargs
        )
