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
from typing import Optional, Union, List, Dict

import sagemaker
from sagemaker import image_uris, ModelMetrics
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.fw_utils import (
    model_code_key_prefix,
    python_deprecation_warning,
    validate_version_or_image_args,
)
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.chainer import defaults
from sagemaker.deserializers import NumpyDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.utils import to_string
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger("sagemaker")


class ChainerPredictor(Predictor):
    """A Predictor for inference against Chainer Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to
    multidimensional tensors for Chainer inference.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    ):
        """Initialize an ``ChainerPredictor``.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to .npy format. Handles lists and numpy
                arrays.
            deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
                Default parses the response from .npy format to numpy array.
        """
        super(ChainerPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class ChainerModel(FrameworkModel):
    """An Chainer SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    _framework_name = "chainer"

    def __init__(
        self,
        model_data: Union[str, PipelineVariable],
        role: Optional[str] = None,
        entry_point: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        framework_version: Optional[str] = None,
        py_version: Optional[str] = None,
        predictor_cls: callable = ChainerPredictor,
        model_server_workers: Optional[Union[int, PipelineVariable]] = None,
        **kwargs
    ):
        """Initialize an ChainerModel.

        Args:
            model_data (str or PipelineVariable): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to model
                hosting. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            image_uri (str or PipelineVariable): A Docker image URI (default: None).
                If not specified, a default image for Chainer will be used.
                If ``framework_version`` or ``py_version``
                are ``None``, then ``image_uri`` is required. If ``image_uri`` is also ``None``,
                then a ``ValueError`` will be raised.
            framework_version (str): Chainer version you want to use for
                executing your model training code. Defaults to ``None``. Required
                unless ``image_uri`` is provided.
            py_version (str): Python version you want to use for executing your
                model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided.
            predictor_cls (callable[str, sagemaker.session.Session]): A function
                to call to create a predictor with an endpoint name and
                SageMaker ``Session``. If specified, ``deploy()`` returns the
                result of invoking this function on the created endpoint name.
            model_server_workers (int or PipelineVariable): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            **kwargs: Keyword arguments passed to the
                :class:`~sagemaker.model.FrameworkModel` initializer.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.FrameworkModel` and
            :class:`~sagemaker.model.Model`.
        """
        validate_version_or_image_args(framework_version, py_version, image_uri)
        if py_version == "py2":
            logger.warning(
                python_deprecation_warning(self._framework_name, defaults.LATEST_PY2_VERSION)
            )
        self.framework_version = framework_version
        self.py_version = py_version

        super(ChainerModel, self).__init__(
            model_data, image_uri, role, entry_point, predictor_cls=predictor_cls, **kwargs
        )

        self.model_server_workers = model_server_workers

    def register(
        self,
        content_types: List[Union[str, PipelineVariable]],
        response_types: List[Union[str, PipelineVariable]],
        inference_instances: Optional[List[Union[str, PipelineVariable]]] = None,
        transform_instances: Optional[List[Union[str, PipelineVariable]]] = None,
        model_package_name: Optional[Union[str, PipelineVariable]] = None,
        model_package_group_name: Optional[Union[str, PipelineVariable]] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        model_metrics: Optional[ModelMetrics] = None,
        metadata_properties: Optional[MetadataProperties] = None,
        marketplace_cert: bool = False,
        approval_status: Optional[Union[str, PipelineVariable]] = None,
        description: Optional[str] = None,
        drift_check_baselines: Optional[DriftCheckBaselines] = None,
        customer_metadata_properties: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        domain: Optional[Union[str, PipelineVariable]] = None,
        sample_payload_url: Optional[Union[str, PipelineVariable]] = None,
        task: Optional[Union[str, PipelineVariable]] = None,
        framework: Optional[Union[str, PipelineVariable]] = None,
        framework_version: Optional[Union[str, PipelineVariable]] = None,
        nearest_model_name: Optional[Union[str, PipelineVariable]] = None,
        data_input_configuration: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Creates a model package for creating SageMaker models or listing on Marketplace.

        Args:
            content_types (list[str] or list[PipelineVariable]): The supported MIME types
                for the input data.
            response_types (list[str] or list[PipelineVariable]): The supported MIME types
                for the output data.
            inference_instances (list[str] or list[PipelineVariable]): A list of the instance
                types that are used to generate inferences in real-time.
            transform_instances (list[str] or list[PipelineVariable]): A list of the instance
                types on which a transformation job can be run or on which an endpoint
                can be deployed.
            model_package_name (str or PipelineVariable): Model Package name, exclusive to
                `model_package_group_name`, using `model_package_name` makes the Model Package
                un-versioned (default: None).
            model_package_group_name (str or PipelineVariable): Model Package Group name,
                exclusive to `model_package_name`, using `model_package_group_name` makes the
                Model Package versioned (default: None).
            image_uri (str or PipelineVariable): Inference image uri for the container. Model class'
                self.image will be used if it is None (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties (default: None).
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace (default: False).
            approval_status (str or PipelineVariable): Model Approval Status, values can be
                "Approved", "Rejected", or "PendingManualApproval"
                (default: "PendingManualApproval").
            description (str): Model Package description (default: None).
            drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
            customer_metadata_properties (dict[str, str] or dict[str, PipelineVariable]):
                A dictionary of key-value paired metadata properties (default: None).
            domain (str or PipelineVariable): Domain values can be "COMPUTER_VISION",
                "NATURAL_LANGUAGE_PROCESSING", "MACHINE_LEARNING" (default: None).
            sample_payload_url (str or PipelineVariable): The S3 path where the sample payload
                is stored (default: None).
            task (str or PipelineVariable): Task values which are supported by Inference Recommender
                are "FILL_MASK", "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION",
                "IMAGE_SEGMENTATION", "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
            framework (str or PipelineVariable): Machine learning framework of the model package
                container image (default: None).
            framework_version (str or PipelineVariable): Framework version of the Model Package
                Container Image (default: None).
            nearest_model_name (str or PipelineVariable): Name of a pre-trained machine learning
                benchmarked by Amazon SageMaker Inference Recommender (default: None).
            data_input_configuration (str or PipelineVariable): Input object for the model
                (default: None).

        Returns:
            str: A string of SageMaker Model Package ARN.
        """
        instance_type = inference_instances[0]
        self._init_sagemaker_session_if_does_not_exist(instance_type)

        if image_uri:
            self.image_uri = image_uri
        if not self.image_uri:
            self.image_uri = self.serving_image_uri(
                region_name=self.sagemaker_session.boto_session.region_name,
                instance_type=instance_type,
            )
        if not is_pipeline_variable(framework):
            framework = (framework or self._framework_name).upper()
        return super(ChainerModel, self).register(
            content_types,
            response_types,
            inference_instances,
            transform_instances,
            model_package_name,
            model_package_group_name,
            image_uri,
            model_metrics,
            metadata_properties,
            marketplace_cert,
            approval_status,
            description,
            drift_check_baselines=drift_check_baselines,
            customer_metadata_properties=customer_metadata_properties,
            domain=domain,
            sample_payload_url=sample_payload_url,
            task=task,
            framework=framework,
            framework_version=framework_version or self.framework_version,
            nearest_model_name=nearest_model_name,
            data_input_configuration=data_input_configuration,
        )

    def prepare_container_def(
        self, instance_type=None, accelerator_type=None, serverless_inference_config=None
    ):
        """Return a container definition with framework configuration set in model environment.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. For example, 'ml.eia1.medium'.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.

        Returns:
            dict[str, str]: A container definition object usable with the
            CreateModel API.
        """
        deploy_image = self.image_uri
        if not deploy_image:
            if instance_type is None and serverless_inference_config is None:
                raise ValueError(
                    "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
                )

            region_name = self.sagemaker_session.boto_session.region_name
            deploy_image = self.serving_image_uri(
                region_name,
                instance_type,
                accelerator_type=accelerator_type,
                serverless_inference_config=serverless_inference_config,
            )

        deploy_key_prefix = model_code_key_prefix(self.key_prefix, self.name, deploy_image)
        self._upload_code(deploy_key_prefix)
        deploy_env = dict(self.env)
        deploy_env.update(self._script_mode_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = to_string(
                self.model_server_workers
            )
        return sagemaker.container_def(deploy_image, self.model_data, deploy_env)

    def serving_image_uri(
        self, region_name, instance_type, accelerator_type=None, serverless_inference_config=None
    ):
        """Create a URI for the serving image.

        Args:
            region_name (str): AWS region where the image is uploaded.
            instance_type (str): SageMaker instance type. Used to determine device type
                (cpu/gpu/family-specific optimized).
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to determine device type.

        Returns:
            str: The appropriate image URI based on the given parameters.

        """
        return image_uris.retrieve(
            self._framework_name,
            region_name,
            version=self.framework_version,
            py_version=self.py_version,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            image_scope="inference",
            serverless_inference_config=serverless_inference_config,
        )
