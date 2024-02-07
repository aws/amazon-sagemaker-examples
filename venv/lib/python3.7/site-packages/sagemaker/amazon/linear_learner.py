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

import logging
from typing import Union, Optional

from sagemaker import image_uris
from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase
from sagemaker.amazon.common import RecordSerializer, RecordDeserializer
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import isin, gt, lt, ge, le
from sagemaker.predictor import Predictor
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.utils import pop_out_unused_kwarg
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger(__name__)


class LinearLearner(AmazonAlgorithmEstimatorBase):
    """A supervised learning algorithms used for solving classification or regression problems.

    For input, you give the model labeled examples (x, y). x is a high-dimensional vector and
    y is a numeric label. For binary classification problems, the label must be either 0 or 1.
    For multiclass classification problems, the labels must be from 0 to num_classes - 1. For
    regression problems, y is a real number. The algorithm learns a linear function, or, for
    classification problems, a linear threshold function, and maps a vector x to an approximation
    of the label y.
    """

    repo_name: str = "linear-learner"
    repo_version: str = "1"

    DEFAULT_MINI_BATCH_SIZE: int = 1000

    binary_classifier_model_selection_criteria: hp = hp(
        "binary_classifier_model_selection_criteria",
        isin(
            "accuracy",
            "f1",
            "f_beta",
            "precision_at_target_recall",
            "recall_at_target_precision",
            "cross_entropy_loss",
            "loss_function",
        ),
        data_type=str,
    )
    target_recall: hp = hp("target_recall", (gt(0), lt(1)), "A float in (0,1)", float)
    target_precision: hp = hp("target_precision", (gt(0), lt(1)), "A float in (0,1)", float)
    positive_example_weight_mult: hp = hp(
        "positive_example_weight_mult", (), "A float greater than 0 or 'auto' or 'balanced'", str
    )
    epochs: hp = hp("epochs", gt(0), "An integer greater-than 0", int)
    predictor_type: hp = hp(
        "predictor_type",
        isin("binary_classifier", "regressor", "multiclass_classifier"),
        'One of "binary_classifier" or "multiclass_classifier" or "regressor"',
        str,
    )
    use_bias: hp = hp("use_bias", (), "Either True or False", bool)
    num_models: hp = hp("num_models", gt(0), "An integer greater-than 0", int)
    num_calibration_samples: hp = hp(
        "num_calibration_samples", gt(0), "An integer greater-than 0", int
    )
    init_method: hp = hp(
        "init_method", isin("uniform", "normal"), 'One of "uniform" or "normal"', str
    )
    init_scale: hp = hp("init_scale", gt(0), "A float greater-than 0", float)
    init_sigma: hp = hp("init_sigma", gt(0), "A float greater-than 0", float)
    init_bias: hp = hp("init_bias", (), "A number", float)
    optimizer: hp = hp(
        "optimizer",
        isin("sgd", "adam", "rmsprop", "auto"),
        'One of "sgd", "adam", "rmsprop" or "auto',
        str,
    )
    loss: hp = hp(
        "loss",
        isin(
            "logistic",
            "squared_loss",
            "absolute_loss",
            "hinge_loss",
            "eps_insensitive_squared_loss",
            "eps_insensitive_absolute_loss",
            "quantile_loss",
            "huber_loss",
            "softmax_loss",
            "auto",
        ),
        '"logistic", "squared_loss", "absolute_loss", "hinge_loss", "eps_insensitive_squared_loss",'
        ' "eps_insensitive_absolute_loss", "quantile_loss", "huber_loss", "softmax_loss" or "auto"',
        str,
    )
    wd: hp = hp("wd", ge(0), "A float greater-than or equal to 0", float)
    l1: hp = hp("l1", ge(0), "A float greater-than or equal to 0", float)
    momentum: hp = hp("momentum", (ge(0), lt(1)), "A float in [0,1)", float)
    learning_rate: hp = hp("learning_rate", gt(0), "A float greater-than 0", float)
    beta_1: hp = hp("beta_1", (ge(0), lt(1)), "A float in [0,1)", float)
    beta_2: hp = hp("beta_2", (ge(0), lt(1)), "A float in [0,1)", float)
    bias_lr_mult: hp = hp("bias_lr_mult", gt(0), "A float greater-than 0", float)
    bias_wd_mult: hp = hp("bias_wd_mult", ge(0), "A float greater-than or equal to 0", float)
    use_lr_scheduler: hp = hp("use_lr_scheduler", (), "A boolean", bool)
    lr_scheduler_step: hp = hp("lr_scheduler_step", gt(0), "An integer greater-than 0", int)
    lr_scheduler_factor: hp = hp("lr_scheduler_factor", (gt(0), lt(1)), "A float in (0,1)", float)
    lr_scheduler_minimum_lr: hp = hp(
        "lr_scheduler_minimum_lr", gt(0), "A float greater-than 0", float
    )
    normalize_data: hp = hp("normalize_data", (), "A boolean", bool)
    normalize_label: hp = hp("normalize_label", (), "A boolean", bool)
    unbias_data: hp = hp("unbias_data", (), "A boolean", bool)
    unbias_label: hp = hp("unbias_label", (), "A boolean", bool)
    num_point_for_scaler: hp = hp("num_point_for_scaler", gt(0), "An integer greater-than 0", int)
    margin: hp = hp("margin", ge(0), "A float greater-than or equal to 0", float)
    quantile: hp = hp("quantile", (gt(0), lt(1)), "A float in (0,1)", float)
    loss_insensitivity: hp = hp("loss_insensitivity", gt(0), "A float greater-than 0", float)
    huber_delta: hp = hp("huber_delta", ge(0), "A float greater-than or equal to 0", float)
    early_stopping_patience: hp = hp(
        "early_stopping_patience", gt(0), "An integer greater-than 0", int
    )
    early_stopping_tolerance: hp = hp(
        "early_stopping_tolerance", gt(0), "A float greater-than 0", float
    )
    num_classes: hp = hp("num_classes", (gt(0), le(1000000)), "An integer in [1,1000000]", int)
    accuracy_top_k: hp = hp(
        "accuracy_top_k", (gt(0), le(1000000)), "An integer in [1,1000000]", int
    )
    f_beta: hp = hp("f_beta", gt(0), "A float greater-than 0", float)
    balance_multiclass_weights: hp = hp("balance_multiclass_weights", (), "A boolean", bool)

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        predictor_type: Optional[str] = None,
        binary_classifier_model_selection_criteria: Optional[str] = None,
        target_recall: Optional[float] = None,
        target_precision: Optional[float] = None,
        positive_example_weight_mult: Optional[float] = None,
        epochs: Optional[int] = None,
        use_bias: Optional[bool] = None,
        num_models: Optional[int] = None,
        num_calibration_samples: Optional[int] = None,
        init_method: Optional[str] = None,
        init_scale: Optional[float] = None,
        init_sigma: Optional[float] = None,
        init_bias: Optional[float] = None,
        optimizer: Optional[str] = None,
        loss: Optional[str] = None,
        wd: Optional[float] = None,
        l1: Optional[float] = None,
        momentum: Optional[float] = None,
        learning_rate: Optional[float] = None,
        beta_1: Optional[float] = None,
        beta_2: Optional[float] = None,
        bias_lr_mult: Optional[float] = None,
        bias_wd_mult: Optional[float] = None,
        use_lr_scheduler: Optional[bool] = None,
        lr_scheduler_step: Optional[int] = None,
        lr_scheduler_factor: Optional[float] = None,
        lr_scheduler_minimum_lr: Optional[float] = None,
        normalize_data: Optional[bool] = None,
        normalize_label: Optional[bool] = None,
        unbias_data: Optional[bool] = None,
        unbias_label: Optional[bool] = None,
        num_point_for_scaler: Optional[int] = None,
        margin: Optional[float] = None,
        quantile: Optional[float] = None,
        loss_insensitivity: Optional[float] = None,
        huber_delta: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_tolerance: Optional[float] = None,
        num_classes: Optional[int] = None,
        accuracy_top_k: Optional[int] = None,
        f_beta: Optional[float] = None,
        balance_multiclass_weights: Optional[bool] = None,
        **kwargs
    ):
        """An :class:`Estimator` for binary classification and regression.

        Amazon SageMaker Linear Learner provides a solution for both
        classification and regression problems, allowing for exploring different
        training objectives simultaneously and choosing the best solution from a
        validation set. It allows the user to explore a large number of models
        and choose the best, which optimizes either continuous objectives such
        as mean square error, cross entropy loss, absolute error, etc., or
        discrete objectives suited for classification such as F1 measure,
        precision@recall, accuracy. The implementation provides a significant
        speedup over naive hyperparameter optimization techniques and an added
        convenience, when compared with solutions providing a solution only to
        continuous objectives.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit_ndarray`
        or
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
        The former allows a LinearLearner model to be fit on a 2-dimensional
        numpy array. The latter requires Amazon
        :class:`~sagemaker.amazon.record_pb2.Record` protobuf serialized data to
        be stored in S3.

        To learn more about the Amazon protobuf Record class and how to
        prepare bulk data in this format, please consult AWS technical
        documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        After this Estimator is fit, model data is stored in S3. The model
        may be deployed to an Amazon SageMaker Endpoint by invoking
        :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
        deploying an Endpoint, ``deploy`` returns a
        :class:`~sagemaker.amazon.linear_learner.LinearLearnerPredictor` object
        that can be used to make class or regression predictions, using the
        trained model.

        LinearLearner Estimators can be configured by setting
        hyperparameters. The available hyperparameters for LinearLearner are
        documented below. For further information on the AWS LinearLearner
        algorithm, please consult AWS technical documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html

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
            predictor_type (str): The type of predictor to learn. Either
                "binary_classifier" or "multiclass_classifier" or "regressor".
            binary_classifier_model_selection_criteria (str): One of 'accuracy',
                'f1', 'f_beta', 'precision_at_target_recall', 'recall_at_target_precision',
                'cross_entropy_loss', 'loss_function'
            target_recall (float): Target recall. Only applicable if
                binary_classifier_model_selection_criteria is
                precision_at_target_recall.
            target_precision (float): Target precision. Only applicable if
                binary_classifier_model_selection_criteria is
                recall_at_target_precision.
            positive_example_weight_mult (float): The importance weight of
                positive examples is multiplied by this constant. Useful for
                skewed datasets. Only applies for classification tasks.
            epochs (int): The maximum number of passes to make over the training
                data.
            use_bias (bool): Whether to include a bias field
            num_models (int): Number of models to train in parallel. If not set,
                the number of parallel models to train will be decided by the
                algorithm itself. One model will be trained according to the
                given training parameter (regularization, optimizer, loss) and
                the rest by close by parameters.
            num_calibration_samples (int): Number of observations to use from
                validation dataset for doing model calibration (finding the best threshold).
            init_method (str): Function to use to set the initial model weights.
                One of "uniform" or "normal"
            init_scale (float): For "uniform" init, the range of values.
            init_sigma (float): For "normal" init, the standard-deviation.
            init_bias (float): Initial weight for bias term
            optimizer (str): One of 'sgd', 'adam', 'rmsprop' or 'auto'
            loss (str): One of 'logistic', 'squared_loss', 'absolute_loss',
                'hinge_loss', 'eps_insensitive_squared_loss', 'eps_insensitive_absolute_loss',
                'quantile_loss', 'huber_loss' or
            'softmax_loss' or 'auto'.
            wd (float): L2 regularization parameter i.e. the weight decay
                parameter. Use 0 for no L2 regularization.
            l1 (float): L1 regularization parameter. Use 0 for no L1
                regularization.
            momentum (float): Momentum parameter of sgd optimizer.
            learning_rate (float): The SGD learning rate
            beta_1 (float): Exponential decay rate for first moment estimates.
                Only applies for adam optimizer.
            beta_2 (float): Exponential decay rate for second moment estimates.
                Only applies for adam optimizer.
            bias_lr_mult (float): Allows different learning rate for the bias
                term. The actual learning rate for the bias is learning rate times bias_lr_mult.
            bias_wd_mult (float): Allows different regularization for the bias
                term. The actual L2 regularization weight for the bias is wd times bias_wd_mult.
                By default there is no regularization on the bias term.
            use_lr_scheduler (bool): If true, we use a scheduler for the
                learning rate.
            lr_scheduler_step (int): The number of steps between decreases of
                the learning rate. Only applies to learning rate scheduler.
            lr_scheduler_factor (float): Every lr_scheduler_step the learning
                rate will decrease by this quantity. Only applies for learning
                rate scheduler.
            lr_scheduler_minimum_lr (float): The learning rate will never
                decrease to a value lower than this. Only applies for learning rate scheduler.
            normalize_data (bool): Normalizes the features before training to
                have standard deviation of 1.0.
            normalize_label (bool): Normalizes the regression label to have a
                standard deviation of 1.0. If set for classification, it will be
                ignored.
            unbias_data (bool): If true, features are modified to have mean 0.0.
            unbias_label (bool): If true, labels are modified to have mean 0.0.
            num_point_for_scaler (int): The number of data points to use for
                calculating the normalizing and unbiasing terms.
            margin (float): the margin for hinge_loss.
            quantile (float): Quantile for quantile loss. For quantile q, the
                model will attempt to produce predictions such that true_label < prediction with
                probability q.
            loss_insensitivity (float): Parameter for epsilon insensitive loss
                type. During training and metric evaluation, any error smaller than this is
                considered to be zero.
            huber_delta (float): Parameter for Huber loss. During training and
                metric evaluation, compute L2 loss for errors smaller than delta and L1 loss for
                errors larger than delta.
            early_stopping_patience (int): the number of epochs to wait before ending training
                if no improvement is made. The improvement is training loss if validation data is
                not provided, or else it is the validation loss or the binary classification model
                selection criteria like accuracy, f1-score etc. To disable early stopping,
                set early_stopping_patience to a value larger than epochs.
            early_stopping_tolerance (float): Relative tolerance to measure an
                improvement in loss. If the ratio of the improvement in loss divided by the
                previous best loss is smaller than this value, early stopping will
                consider the improvement to be zero.
            num_classes (int): The number of classes for the response variable.
                Required when predictor_type is multiclass_classifier and ignored otherwise. The
                classes are assumed to be labeled 0, ..., num_classes - 1.
            accuracy_top_k (int): The value of k when computing the Top K
                Accuracy metric for multiclass classification. An example is scored as correct
                if the model assigns one of the top k scores to the true
                label.
            f_beta (float): The value of beta to use when calculating F score
                metrics for binary or multiclass classification. Also used if
                binary_classifier_model_selection_criteria is f_beta.
            balance_multiclass_weights (bool): Whether to use class weights
                which give each class equal importance in the loss function. Only used when
                predictor_type is multiclass_classifier.
            **kwargs: base class keyword argument values.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.amazon_estimator.AmazonAlgorithmEstimatorBase` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        super(LinearLearner, self).__init__(role, instance_count, instance_type, **kwargs)
        self.predictor_type = predictor_type
        self.binary_classifier_model_selection_criteria = binary_classifier_model_selection_criteria
        self.target_recall = target_recall
        self.target_precision = target_precision
        self.positive_example_weight_mult = positive_example_weight_mult
        self.epochs = epochs
        self.use_bias = use_bias
        self.num_models = num_models
        self.num_calibration_samples = num_calibration_samples
        self.init_method = init_method
        self.init_scale = init_scale
        self.init_sigma = init_sigma
        self.init_bias = init_bias
        self.optimizer = optimizer
        self.loss = loss
        self.wd = wd
        self.l1 = l1
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.bias_lr_mult = bias_lr_mult
        self.bias_wd_mult = bias_wd_mult
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_minimum_lr = lr_scheduler_minimum_lr
        self.normalize_data = normalize_data
        self.normalize_label = normalize_label
        self.unbias_data = unbias_data
        self.unbias_label = unbias_label
        self.num_point_for_scaler = num_point_for_scaler
        self.margin = margin
        self.quantile = quantile
        self.loss_insensitivity = loss_insensitivity
        self.huber_delta = huber_delta
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tolerance = early_stopping_tolerance
        self.num_classes = num_classes
        self.accuracy_top_k = accuracy_top_k
        self.f_beta = f_beta
        self.balance_multiclass_weights = balance_multiclass_weights

        if self.predictor_type == "multiclass_classifier" and (
            num_classes is None or int(num_classes) < 3
        ):
            raise ValueError(
                "For predictor_type 'multiclass_classifier', 'num_classes' should be set to a "
                "value greater than 2."
            )

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT, **kwargs):
        """Return a :class:`~sagemaker.amazon.LinearLearnerModel`.

        It references the latest s3 model data produced by this Estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional kwargs passed to the LinearLearnerModel constructor.
        """
        return LinearLearnerModel(
            self.model_data,
            self.role,
            self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            **kwargs
        )

    def _prepare_for_training(self, records, mini_batch_size=None, job_name=None):
        """Placeholder docstring"""
        num_records = None
        if isinstance(records, list):
            for record in records:
                if record.channel == "train":
                    num_records = record.num_records
                    break
            if num_records is None:
                raise ValueError("Must provide train channel.")
        else:
            num_records = records.num_records

        # mini_batch_size can't be greater than number of records or training job fails
        mini_batch_size = mini_batch_size or self._get_default_mini_batch_size(num_records)

        super(LinearLearner, self)._prepare_for_training(
            records, mini_batch_size=mini_batch_size, job_name=job_name
        )


class LinearLearnerPredictor(Predictor):
    """Performs binary-classification or regression prediction from input vectors.

    The implementation of
    :meth:`~sagemaker.predictor.Predictor.predict` in this
    `Predictor` requires a numpy ``ndarray`` as input. The array should
    contain the same number of columns as the feature-dimension of the data used
    to fit the model this Predictor performs inference on.

    :func:`predict` returns a list of
    :class:`~sagemaker.amazon.record_pb2.Record` objects (assuming the default
    recordio-protobuf ``deserializer`` is used), one for each row in
    the input ``ndarray``. The prediction is stored in the ``"predicted_label"``
    key of the ``Record.label`` field.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=RecordSerializer(),
        deserializer=RecordDeserializer(),
    ):
        """Initialization for LinearLearnerPredictor.

        Args:
            endpoint_name (str): Name of the Amazon SageMaker endpoint to which
                requests are sent.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to x-recordio-protobuf format.
            deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
                Default parses responses from x-recordio-protobuf format.
        """
        super(LinearLearnerPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class LinearLearnerModel(Model):
    """Reference LinearLearner s3 model data.

    Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and returns a
    :class:`LinearLearnerPredictor`
    """

    def __init__(
        self,
        model_data: Union[str, PipelineVariable],
        role: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        **kwargs
    ):
        """Initialization for LinearLearnerModel.

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
            LinearLearner.repo_name,
            sagemaker_session.boto_region_name,
            version=LinearLearner.repo_version,
        )
        pop_out_unused_kwarg("predictor_cls", kwargs, LinearLearnerPredictor.__name__)
        pop_out_unused_kwarg("image_uri", kwargs, image_uri)
        super(LinearLearnerModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=LinearLearnerPredictor,
            sagemaker_session=sagemaker_session,
            **kwargs
        )
