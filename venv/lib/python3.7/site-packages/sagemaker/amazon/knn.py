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
from sagemaker.amazon.common import RecordSerializer, RecordDeserializer
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import ge, isin
from sagemaker.predictor import Predictor
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.utils import pop_out_unused_kwarg
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable


class KNN(AmazonAlgorithmEstimatorBase):
    """An index-based algorithm. It uses a non-parametric method for classification or regression.

    For classification problems, the algorithm queries the k points that are closest to the sample
    point and returns the most frequently used label of their class as the predicted label. For
    regression problems, the algorithm queries the k closest points to the sample point and returns
    the average of their feature values as the predicted value.
    """

    repo_name: str = "knn"
    repo_version: str = "1"

    k: hp = hp("k", (ge(1)), "An integer greater than 0", int)
    sample_size: hp = hp("sample_size", (ge(1)), "An integer greater than 0", int)
    predictor_type: hp = hp(
        "predictor_type", isin("classifier", "regressor"), 'One of "classifier" or "regressor"', str
    )
    dimension_reduction_target: hp = hp(
        "dimension_reduction_target",
        (ge(1)),
        "An integer greater than 0 and less than feature_dim",
        int,
    )
    dimension_reduction_type: hp = hp(
        "dimension_reduction_type", isin("sign", "fjlt"), 'One of "sign" or "fjlt"', str
    )
    index_metric: hp = hp(
        "index_metric",
        isin("COSINE", "INNER_PRODUCT", "L2"),
        'One of "COSINE", "INNER_PRODUCT", "L2"',
        str,
    )
    index_type: hp = hp(
        "index_type",
        isin("faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ"),
        'One of "faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ"',
        str,
    )
    faiss_index_ivf_nlists: hp = hp(
        "faiss_index_ivf_nlists", (), '"auto" or an integer greater than 0', str
    )
    faiss_index_pq_m: hp = hp("faiss_index_pq_m", (ge(1)), "An integer greater than 0", int)

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        k: Optional[int] = None,
        sample_size: Optional[int] = None,
        predictor_type: Optional[str] = None,
        dimension_reduction_type: Optional[str] = None,
        dimension_reduction_target: Optional[int] = None,
        index_type: Optional[str] = None,
        index_metric: Optional[str] = None,
        faiss_index_ivf_nlists: Optional[str] = None,
        faiss_index_pq_m: Optional[int] = None,
        **kwargs
    ):
        """k-nearest neighbors (KNN) is :class:`Estimator` used for classification and regression.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
        It requires Amazon :class:`~sagemaker.amazon.record_pb2.Record` protobuf
        serialized data to be stored in S3. There is an utility
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.record_set`
        that can be used to upload data to S3 and creates
        :class:`~sagemaker.amazon.amazon_estimator.RecordSet` to be passed to
        the `fit` call. To learn more about the Amazon protobuf Record class and
        how to prepare bulk data in this format, please consult AWS technical
        documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html After
        this Estimator is fit, model data is stored in S3. The model may be
        deployed to an Amazon SageMaker Endpoint by invoking
        :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
        deploying an Endpoint, deploy returns a
        :class:`~sagemaker.amazon.knn.KNNPredictor` object that can be used for
        inference calls using the trained model hosted in the SageMaker
        Endpoint. KNN Estimators can be configured by setting hyperparameters.
        The available hyperparameters for KNN are documented below. For further
        information on the AWS KNN algorithm, please consult AWS technical
        documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/knn.html

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if accessing AWS resource.
            instance_count: (int or PipelineVariable): Number of Amazon EC2 instances to use
                for training.
            instance_type (str or PipelineVariable): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            k (int): Required. Number of nearest neighbors.
            sample_size (int): Required. Number of data points to be sampled
                from the training data set.
            predictor_type (str): Required. Type of inference to use on the
                data's labels, allowed values are 'classifier' and 'regressor'.
            dimension_reduction_type (str): Optional. Type of dimension
                reduction technique to use. Valid values: "sign", "fjlt"
            dimension_reduction_target (int): Optional. Target dimension to
                reduce to. Required when dimension_reduction_type is specified.
            index_type (str): Optional. Type of index to use. Valid values are
                "faiss.Flat", "faiss.IVFFlat", "faiss.IVFPQ".
            index_metric (str): Optional. Distance metric to measure between
                points when finding nearest neighbors. Valid values are
                "COSINE", "INNER_PRODUCT", "L2"
            faiss_index_ivf_nlists (str): Optional. Number of centroids to
                construct in the index if index_type is "faiss.IVFFlat" or
                "faiss.IVFPQ".
            faiss_index_pq_m (int): Optional. Number of vector sub-components to
                construct in the index, if index_type is "faiss.IVFPQ".
            **kwargs: base class keyword argument values.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.amazon_estimator.AmazonAlgorithmEstimatorBase` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """

        super(KNN, self).__init__(role, instance_count, instance_type, **kwargs)
        self.k = k
        self.sample_size = sample_size
        self.predictor_type = predictor_type
        self.dimension_reduction_type = dimension_reduction_type
        self.dimension_reduction_target = dimension_reduction_target
        self.index_type = index_type
        self.index_metric = index_metric
        self.faiss_index_ivf_nlists = faiss_index_ivf_nlists
        self.faiss_index_pq_m = faiss_index_pq_m
        if dimension_reduction_type and not dimension_reduction_target:
            raise ValueError(
                '"dimension_reduction_target" is required when "dimension_reduction_type" is set.'
            )

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT, **kwargs):
        """Return a :class:`~sagemaker.amazon.KNNModel`.

        It references the latest s3 model data produced by this Estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional kwargs passed to the KNNModel constructor.
        """
        return KNNModel(
            self.model_data,
            self.role,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            **kwargs
        )

    def _prepare_for_training(self, records, mini_batch_size=None, job_name=None):
        """Placeholder docstring"""
        super(KNN, self)._prepare_for_training(
            records, mini_batch_size=mini_batch_size, job_name=job_name
        )


class KNNPredictor(Predictor):
    """Performs classification or regression prediction from input vectors.

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
        """Function to initialize KNNPredictor.

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
        super(KNNPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class KNNModel(Model):
    """Reference S3 model data created by KNN estimator.

    Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint
    and returns :class:`KNNPredictor`.
    """

    def __init__(
        self,
        model_data: Union[str, PipelineVariable],
        role: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        **kwargs
    ):
        """Function to initialize KNNModel.

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
            KNN.repo_name,
            sagemaker_session.boto_region_name,
            version=KNN.repo_version,
        )
        pop_out_unused_kwarg("predictor_cls", kwargs, KNNPredictor.__name__)
        pop_out_unused_kwarg("image_uri", kwargs, image_uri)
        super(KNNModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=KNNPredictor,
            sagemaker_session=sagemaker_session,
            **kwargs
        )
