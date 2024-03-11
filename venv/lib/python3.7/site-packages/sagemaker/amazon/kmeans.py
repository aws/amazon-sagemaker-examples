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

from typing import Union, Optional, List

from sagemaker import image_uris
from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase
from sagemaker.amazon.common import RecordSerializer, RecordDeserializer
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.amazon.validation import gt, isin, ge, le
from sagemaker.predictor import Predictor
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.utils import pop_out_unused_kwarg
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable


class KMeans(AmazonAlgorithmEstimatorBase):
    """An unsupervised learning algorithm that attempts to find discrete groupings within data.

    As the result of KMeans, members of a group are as similar as possible to one another and as
    different as possible from members of other groups. You define the attributes that you want
    the algorithm to use to determine similarity.
    """

    repo_name: str = "kmeans"
    repo_version: str = "1"

    k: hp = hp("k", gt(1), "An integer greater-than 1", int)
    init_method: hp = hp(
        "init_method", isin("random", "kmeans++"), 'One of "random", "kmeans++"', str
    )
    max_iterations: hp = hp("local_lloyd_max_iter", gt(0), "An integer greater-than 0", int)
    tol: hp = hp("local_lloyd_tol", (ge(0), le(1)), "An float in [0, 1]", float)
    num_trials: hp = hp("local_lloyd_num_trials", gt(0), "An integer greater-than 0", int)
    local_init_method: hp = hp(
        "local_lloyd_init_method", isin("random", "kmeans++"), 'One of "random", "kmeans++"', str
    )
    half_life_time_size: hp = hp(
        "half_life_time_size", ge(0), "An integer greater-than-or-equal-to 0", int
    )
    epochs: hp = hp("epochs", gt(0), "An integer greater-than 0", int)
    center_factor: hp = hp("extra_center_factor", gt(0), "An integer greater-than 0", int)
    eval_metrics: hp = hp(
        name="eval_metrics",
        validation_message='A comma separated list of "msd" or "ssd"',
        data_type=list,
    )

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        k: Optional[int] = None,
        init_method: Optional[str] = None,
        max_iterations: Optional[int] = None,
        tol: Optional[float] = None,
        num_trials: Optional[int] = None,
        local_init_method: Optional[str] = None,
        half_life_time_size: Optional[int] = None,
        epochs: Optional[int] = None,
        center_factor: Optional[int] = None,
        eval_metrics: Optional[List[Union[str, PipelineVariable]]] = None,
        **kwargs
    ):
        """A k-means clustering class :class:`~sagemaker.amazon.AmazonAlgorithmEstimatorBase`.

        Finds k clusters of data in an unlabeled dataset.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit_ndarray`
        or
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
        The former allows a KMeans model to be fit on a 2-dimensional numpy
        array. The latter requires Amazon
        :class:`~sagemaker.amazon.record_pb2.Record` protobuf serialized data to
        be stored in S3.

        To learn more about the Amazon protobuf Record class and how to
        prepare bulk data in this format, please consult AWS technical
        documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html.

        After this Estimator is fit, model data is stored in S3. The model
        may be deployed to an Amazon SageMaker Endpoint by invoking
        :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
        deploying an Endpoint, ``deploy`` returns a
        :class:`~sagemaker.amazon.kmeans.KMeansPredictor` object that can be
        used to k-means cluster assignments, using the trained k-means model
        hosted in the SageMaker Endpoint.

        KMeans Estimators can be configured by setting hyperparameters. The
        available hyperparameters for KMeans are documented below. For further
        information on the AWS KMeans algorithm, please consult AWS technical
        documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html.

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
            k (int): The number of clusters to produce.
            init_method (str): How to initialize cluster locations. One of
                'random' or 'kmeans++'.
            max_iterations (int): Maximum iterations for Lloyds EM procedure in
                the local kmeans used in finalize stage.
            tol (float): Tolerance for change in ssd for early stopping in local
                kmeans.
            num_trials (int): Local version is run multiple times and the one
                with the best loss is chosen. This determines how many times.
            local_init_method (str): Initialization method for local version.
                One of 'random', 'kmeans++'
            half_life_time_size (int): The points can have a decayed weight.
                When a point is observed its weight, with regard to the
                computation of the cluster mean is 1. This weight will decay
                exponentially as we observe more points. The exponent
                coefficient is chosen such that after observing
                ``half_life_time_size`` points after the mentioned point, its
                weight will become 1/2. If set to 0, there will be no decay.
            epochs (int): Number of passes done over the training data.
            center_factor (int): The algorithm will create
                ``num_clusters * extra_center_factor`` as it runs and reduce the
                number of centers to ``k`` when finalizing
            eval_metrics (list[str] or list[PipelineVariable]): JSON list of metrics types
                to be used for reporting the score for the model. Allowed values are "msd"
                Means Square Error, "ssd": Sum of square distance. If test data
                is provided, the score shall be reported in terms of all
                requested metrics.
            **kwargs: base class keyword argument values.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.amazon_estimator.AmazonAlgorithmEstimatorBase` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        super(KMeans, self).__init__(role, instance_count, instance_type, **kwargs)
        self.k = k
        self.init_method = init_method
        self.max_iterations = max_iterations
        self.tol = tol
        self.num_trials = num_trials
        self.local_init_method = local_init_method
        self.half_life_time_size = half_life_time_size
        self.epochs = epochs
        self.center_factor = center_factor
        self.eval_metrics = eval_metrics

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT, **kwargs):
        """Return a :class:`~sagemaker.amazon.kmeans.KMeansModel`.

        It references the latest s3 model data produced by this Estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for
                VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional kwargs passed to the KMeansModel constructor.
        """
        return KMeansModel(
            self.model_data,
            self.role,
            self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            **kwargs
        )

    def _prepare_for_training(self, records, mini_batch_size=5000, job_name=None):
        """Placeholder docstring"""
        super(KMeans, self)._prepare_for_training(
            records, mini_batch_size=mini_batch_size, job_name=job_name
        )

    def hyperparameters(self):
        """Return the SageMaker hyperparameters for training this KMeans Estimator."""
        hp_dict = dict(force_dense="True")  # KMeans requires this hp to fit on Record objects
        hp_dict.update(super(KMeans, self).hyperparameters())
        return hp_dict


class KMeansPredictor(Predictor):
    """Assigns input vectors to their closest cluster in a KMeans model.

    The implementation of
    :meth:`~sagemaker.predictor.Predictor.predict` in this
    `Predictor` requires a numpy ``ndarray`` as input. The array should
    contain the same number of columns as the feature-dimension of the data used
    to fit the model this Predictor performs inference on.

    ``predict()`` returns a list of
    :class:`~sagemaker.amazon.record_pb2.Record` objects (assuming the default
    recordio-protobuf ``deserializer`` is used), one for each row in
    the input ``ndarray``. The nearest cluster is stored in the
    ``closest_cluster`` key of the ``Record.label`` field.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=RecordSerializer(),
        deserializer=RecordDeserializer(),
    ):
        """Initialization for KMeansPredictor class.

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
        super(KMeansPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class KMeansModel(Model):
    """Reference KMeans s3 model data.

    Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return a
    Predictor to performs k-means cluster assignment.
    """

    def __init__(
        self,
        model_data: Union[str, PipelineVariable],
        role: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        **kwargs
    ):
        """Initialization for KMeansModel class.

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
            KMeans.repo_name,
            sagemaker_session.boto_region_name,
            version=KMeans.repo_version,
        )
        pop_out_unused_kwarg("predictor_cls", kwargs, KMeansPredictor.__name__)
        pop_out_unused_kwarg("image_uri", kwargs, image_uri)
        super(KMeansModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=KMeansPredictor,
            sagemaker_session=sagemaker_session,
            **kwargs
        )
