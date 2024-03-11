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

from typing import Optional, Union, List

from sagemaker import image_uris
from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase
from sagemaker.amazon.common import RecordSerializer, RecordDeserializer
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import ge, le
from sagemaker.predictor import Predictor
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.utils import pop_out_unused_kwarg
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable


class RandomCutForest(AmazonAlgorithmEstimatorBase):
    """An unsupervised algorithm for detecting anomalous data points within a data set.

    These are observations which diverge from otherwise well-structured or patterned data.
    Anomalies can manifest as unexpected spikes in time series data, breaks in periodicity,
    or unclassifiable data points.
    """

    repo_name: str = "randomcutforest"
    repo_version: str = "1"
    MINI_BATCH_SIZE: int = 1000

    eval_metrics: hp = hp(
        name="eval_metrics",
        validation_message='A comma separated list of "accuracy" or "precision_recall_fscore"',
        data_type=list,
    )

    num_trees: hp = hp("num_trees", (ge(50), le(1000)), "An integer in [50, 1000]", int)
    num_samples_per_tree: hp = hp(
        "num_samples_per_tree", (ge(1), le(2048)), "An integer in [1, 2048]", int
    )
    feature_dim: hp = hp("feature_dim", (ge(1), le(10000)), "An integer in [1, 10000]", int)

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        num_samples_per_tree: Optional[int] = None,
        num_trees: Optional[int] = None,
        eval_metrics: Optional[List] = None,
        **kwargs
    ):
        """An `Estimator` class implementing a Random Cut Forest.

        Typically used for anomaly detection, this Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
        It requires Amazon :class:`~sagemaker.amazon.record_pb2.Record` protobuf
        serialized data to be stored in S3. There is an utility
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.record_set`
        that can be used to upload data to S3 and creates
        :class:`~sagemaker.amazon.amazon_estimator.RecordSet` to be passed to
        the `fit` call.

        To learn more about the Amazon protobuf Record class and how to
        prepare bulk data in this format, please consult AWS technical
        documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        After this Estimator is fit, model data is stored in S3. The model
        may be deployed to an Amazon SageMaker Endpoint by invoking
        :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
        deploying an Endpoint, deploy returns a
        :class:`~sagemaker.amazon.ntm.RandomCutForestPredictor` object that can
        be used for inference calls using the trained model hosted in the
        SageMaker Endpoint.

        RandomCutForest Estimators can be configured by setting
        hyperparameters. The available hyperparameters for RandomCutForest are
        documented below.

        For further information on the AWS Random Cut Forest algorithm,
        please consult AWS technical documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html

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
            num_samples_per_tree (int): Optional. The number of samples used to
                build each tree in the forest. The total number of samples drawn
                from the train dataset is num_trees * num_samples_per_tree.
            num_trees (int): Optional. The number of trees used in the forest.
            eval_metrics (list): Optional. JSON list of metrics types to be used
                for reporting the score for the model. Allowed values are
                "accuracy", "precision_recall_fscore": positive and negative
                precision, recall, and f1 scores. If test data is provided, the
                score shall be reported in terms of all requested metrics.
            **kwargs: base class keyword argument values.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.amazon_estimator.AmazonAlgorithmEstimatorBase` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """

        super(RandomCutForest, self).__init__(role, instance_count, instance_type, **kwargs)
        self.num_samples_per_tree = num_samples_per_tree
        self.num_trees = num_trees
        self.eval_metrics = eval_metrics

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT, **kwargs):
        """Return a :class:`~sagemaker.amazon.RandomCutForestModel`.

        It references the latest s3 model data produced by this Estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional kwargs passed to the RandomCutForestModel constructor.
        """
        return RandomCutForestModel(
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
        elif mini_batch_size != self.MINI_BATCH_SIZE:
            raise ValueError(
                "Random Cut Forest uses a fixed mini_batch_size of {}".format(self.MINI_BATCH_SIZE)
            )

        super(RandomCutForest, self)._prepare_for_training(
            records, mini_batch_size=mini_batch_size, job_name=job_name
        )


class RandomCutForestPredictor(Predictor):
    """Assigns an anomaly score to each of the datapoints provided.

    The implementation of
    :meth:`~sagemaker.predictor.Predictor.predict` in this
    `Predictor` requires a numpy ``ndarray`` as input. The array should
    contain the same number of columns as the feature-dimension of the data used
    to fit the model this Predictor performs inference on.

    :meth:`predict()` returns a list of
    :class:`~sagemaker.amazon.record_pb2.Record` objects (assuming the default
    recordio-protobuf ``deserializer`` is used), one for each row in
    the input. Each row's score is stored in the key ``score`` of the
    ``Record.label`` field.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=RecordSerializer(),
        deserializer=RecordDeserializer(),
    ):
        """Initialization for RandomCutForestPredictor class.

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
        super(RandomCutForestPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class RandomCutForestModel(Model):
    """Reference RandomCutForest s3 model data.

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
        """Initialization for RandomCutForestModel class.

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
            RandomCutForest.repo_name,
            sagemaker_session.boto_region_name,
            version=RandomCutForest.repo_version,
        )
        pop_out_unused_kwarg("predictor_cls", kwargs, RandomCutForestPredictor.__name__)
        pop_out_unused_kwarg("image_uri", kwargs, image_uri)
        super(RandomCutForestModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=RandomCutForestPredictor,
            sagemaker_session=sagemaker_session,
            **kwargs
        )
