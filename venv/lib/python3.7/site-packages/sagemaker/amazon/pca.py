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
from sagemaker.amazon.validation import gt, isin
from sagemaker.predictor import Predictor
from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.utils import pop_out_unused_kwarg
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable


class PCA(AmazonAlgorithmEstimatorBase):
    """An unsupervised machine learning algorithm to reduce feature dimensionality.

    As a result, number of features within a dataset is reduced but the dataset still
    retain as much information as possible.
    """

    repo_name: str = "pca"
    repo_version: str = "1"

    DEFAULT_MINI_BATCH_SIZE: int = 500

    num_components: hp = hp(
        "num_components", gt(0), "Value must be an integer greater than zero", int
    )
    algorithm_mode: hp = hp(
        "algorithm_mode",
        isin("regular", "randomized"),
        'Value must be one of "regular" and "randomized"',
        str,
    )
    subtract_mean: hp = hp(
        name="subtract_mean", validation_message="Value must be a boolean", data_type=bool
    )
    extra_components: hp = hp(
        name="extra_components",
        validation_message="Value must be an integer greater than or equal to 0, or -1.",
        data_type=int,
    )

    def __init__(
        self,
        role: Optional[Union[str, PipelineVariable]] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        num_components: Optional[int] = None,
        algorithm_mode: Optional[str] = None,
        subtract_mean: Optional[bool] = None,
        extra_components: Optional[int] = None,
        **kwargs
    ):
        """A Principal Components Analysis (PCA)

        :class:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase`.

        This Estimator may be fit via calls to
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit_ndarray`
        or
        :meth:`~sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase.fit`.
        The former allows a PCA model to be fit on a 2-dimensional numpy array.
        The latter requires Amazon :class:`~sagemaker.amazon.record_pb2.Record`
        protobuf serialized data to be stored in S3.

        To learn more about the Amazon protobuf Record class and how to
        prepare bulk data in this format, please consult AWS technical
        documentation:
        https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html

        After this Estimator is fit, model data is stored in S3. The model
        may be deployed to an Amazon SageMaker Endpoint by invoking
        :meth:`~sagemaker.amazon.estimator.EstimatorBase.deploy`. As well as
        deploying an Endpoint, deploy returns a
        :class:`~sagemaker.amazon.pca.PCAPredictor` object that can be used to
        project input vectors to the learned lower-dimensional representation,
        using the trained PCA model hosted in the SageMaker Endpoint.

        PCA Estimators can be configured by setting hyperparameters. The
        available hyperparameters for PCA are documented below. For further
        information on the AWS PCA algorithm, please consult AWS technical
        documentation: https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html

        This Estimator uses Amazon SageMaker PCA to perform training and host
        deployed models. To learn more about Amazon SageMaker PCA, please read:
        https://docs.aws.amazon.com/sagemaker/latest/dg/how-pca-works.html

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
            num_components (int): The number of principal components. Must be
                greater than zero.
            algorithm_mode (str): Mode for computing the principal components.
                One of 'regular' or 'randomized'.
            subtract_mean (bool): Whether the data should be unbiased both
                during train and at inference.
            extra_components (int): As the value grows larger, the solution
                becomes more accurate but the runtime and memory consumption
                increase linearly. If this value is unset or set to -1, then a
                default value equal to the maximum of 10 and num_components will
                be used. Valid for randomized mode only.
            **kwargs: base class keyword argument values.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.amazon_estimator.AmazonAlgorithmEstimatorBase` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        super(PCA, self).__init__(role, instance_count, instance_type, **kwargs)
        self.num_components = num_components
        self.algorithm_mode = algorithm_mode
        self.subtract_mean = subtract_mean
        self.extra_components = extra_components

    def create_model(self, vpc_config_override=VPC_CONFIG_DEFAULT, **kwargs):
        """Return a :class:`~sagemaker.amazon.pca.PCAModel`.

        It references the latest s3 model data produced by this Estimator.

        Args:
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional kwargs passed to the PCAModel constructor.
        """
        return PCAModel(
            self.model_data,
            self.role,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            **kwargs
        )

    def _prepare_for_training(self, records, mini_batch_size=None, job_name=None):
        """Set hyperparameters needed for training.

        Args:
            records (:class:`~RecordSet`): The records to train this ``Estimator`` on.
            mini_batch_size (int or None): The size of each mini-batch to use when
                training. If ``None``, a default value will be used.
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
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

        # mini_batch_size is a required parameter
        use_mini_batch_size = mini_batch_size or self._get_default_mini_batch_size(num_records)

        super(PCA, self)._prepare_for_training(
            records=records, mini_batch_size=use_mini_batch_size, job_name=job_name
        )


class PCAPredictor(Predictor):
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
        """Initialization for PCAPredictor.

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
        super(PCAPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class PCAModel(Model):
    """Reference PCA s3 model data.

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
        """Initialization for PCAModel.

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
            PCA.repo_name,
            sagemaker_session.boto_region_name,
            version=PCA.repo_version,
        )
        pop_out_unused_kwarg("predictor_cls", kwargs, PCAPredictor.__name__)
        pop_out_unused_kwarg("image_uri", kwargs, image_uri)
        super(PCAModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=PCAPredictor,
            sagemaker_session=sagemaker_session,
            **kwargs
        )
