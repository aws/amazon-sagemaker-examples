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
from sagemaker.amazon.validation import gt
from sagemaker.predictor import Predictor
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.utils import pop_out_unused_kwarg
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow import is_pipeline_variable

logger = logging.getLogger(__name__)


class LDA(AmazonAlgorithmEstimatorBase):
    """An unsupervised learning algorithm attempting to describe data as distinct categories.

    LDA is most commonly used to discover a
    user-specified number of topics shared by documents within a text corpus. Here each
    observation is a document, the features are the presence (or occurrence count) of each
    word, and the categories are the topics.
    """

    repo_name: str = "lda"
    repo_version: str = "1"

    num_topics: hp = hp("num_topics", gt(0), "An integer greater than zero", int)
    alpha0: hp = hp("alpha0", gt(0), "A positive float", float)
    max_restarts: hp = hp("max_restarts", gt(0), "An integer greater than zero", int)
    max_iterations: hp = hp("max_iterations", gt(0), "An integer greater than zero", int)
    tol: hp = hp("tol", gt(0), "A positive float", float)

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        num_topics: Optional[int] = None,
        alpha0: Optional[float] = None,
        max_restarts: Optional[int] = None,
        max_iterations: Optional[int] = None,
        tol: Optional[float] = None,
        **kwargs
    ):
        """Latent Dirichlet Allocation (LDA) is :class:`Estimator` used for unsupervised learning.

        Amazon SageMaker Latent Dirichlet Allocation is an unsupervised
        learning algorithm that attempts to describe a set of observations as a
        mixture of distinct categories. LDA is most commonly used to discover a
        user-specified number of topics shared by documents within a text
        corpus. Here each observation is a document, the features are the
        presence (or occurrence count) of each word, and the categories are the
        topics.

        This Estimator may be fit via calls to
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
        :class:`~sagemaker.amazon.lda.LDAPredictor` object that can be used for
        inference calls using the trained model hosted in the SageMaker
        Endpoint.

        LDA Estimators can be configured by setting hyperparameters. The
        available hyperparameters for LDA are documented below.

        For further information on the AWS LDA algorithm, please consult AWS
        technical documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/lda.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if accessing AWS resource.
            instance_type (str or PipelineVariable): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            num_topics (int): The number of topics for LDA to find within the
                data.
            alpha0 (float): Optional. Initial guess for the concentration
                parameter
            max_restarts (int): Optional. The number of restarts to perform
                during the Alternating Least Squares (ALS) spectral
                decomposition phase of the algorithm.
            max_iterations (int): Optional. The maximum number of iterations to
                perform during the ALS phase of the algorithm.
            tol (float): Optional. Target error tolerance for the ALS phase of
                the algorithm.
            **kwargs: base class keyword argument values.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.amazon_estimator.AmazonAlgorithmEstimatorBase` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        # this algorithm only supports single instance training
        instance_count = kwargs.pop("instance_count", 1)
        if is_pipeline_variable(instance_count) or instance_count != 1:
            logger.warning(
                "LDA only supports single instance training. Defaulting to 1 %s.", instance_type
            )

        super(LDA, self).__init__(role, 1, instance_type, **kwargs)
        self.num_topics = num_topics
        self.alpha0 = alpha0
        self.max_restarts = max_restarts
        self.max_iterations = max_iterations
        self.tol = tol

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT, **kwargs):
        """Return a :class:`~sagemaker.amazon.LDAModel`.

        It references the latest s3 model data produced by this Estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for
                VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional kwargs passed to the LDAModel constructor.
        """
        return LDAModel(
            self.model_data,
            self.role,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            **kwargs
        )

    def _prepare_for_training(  # pylint: disable=signature-differs
        self, records, mini_batch_size, job_name=None
    ):
        # mini_batch_size is required, prevent explicit calls with None
        """Placeholder docstring"""
        if mini_batch_size is None:
            raise ValueError("mini_batch_size must be set")

        super(LDA, self)._prepare_for_training(
            records, mini_batch_size=mini_batch_size, job_name=job_name
        )


class LDAPredictor(Predictor):
    """Transforms input vectors to lower-dimesional representations.

    The implementation of
    :meth:`~sagemaker.predictor.Predictor.predict` in this
    `Predictor` requires a numpy ``ndarray`` as input. The array should
    contain the same number of columns as the feature-dimension of the data used
    to fit the model this Predictor performs inference on.

    :meth:`predict()` returns a list of
    :class:`~sagemaker.amazon.record_pb2.Record` objects (assuming the default
    recordio-protobuf ``deserializer`` is used), one for each row in
    the input ``ndarray``. The lower dimension vector result is stored in the
    ``projection`` key of the ``Record.label`` field.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=RecordSerializer(),
        deserializer=RecordDeserializer(),
    ):
        """Creates "LDAPredictor" object to be used for transforming input vectors.

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
        super(LDAPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class LDAModel(Model):
    """Reference LDA s3 model data.

    Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return a
    Predictor that transforms vectors to a lower-dimensional representation.
    """

    def __init__(
        self,
        model_data: Union[str, PipelineVariable],
        role: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        **kwargs
    ):
        """Initialization for LDAModel class.

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
            LDA.repo_name,
            sagemaker_session.boto_region_name,
            version=LDA.repo_version,
        )
        pop_out_unused_kwarg("predictor_cls", kwargs, LDAPredictor.__name__)
        pop_out_unused_kwarg("image_uri", kwargs, image_uri)
        super(LDAModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=LDAPredictor,
            sagemaker_session=sagemaker_session,
            **kwargs
        )
