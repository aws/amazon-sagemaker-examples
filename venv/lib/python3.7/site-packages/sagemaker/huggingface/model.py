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
from sagemaker.deserializers import JSONDeserializer
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.fw_utils import (
    model_code_key_prefix,
    validate_version_or_image_args,
)
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.session import Session
from sagemaker.utils import to_string
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger("sagemaker")


class HuggingFacePredictor(Predictor):
    """A Predictor for inference against Hugging Face Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to
    multidimensional tensors for Hugging Face inference.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    ):
        """Initialize an ``HuggingFacePredictor``.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object that
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to .npy format. Handles lists and numpy
                arrays.
            deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
                Default parses the response from .npy format to numpy array.
        """
        super(HuggingFacePredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


def _validate_pt_tf_versions(pytorch_version, tensorflow_version, image_uri):
    """Placeholder docstring"""

    if image_uri is not None:
        return

    if tensorflow_version is not None and pytorch_version is not None:
        raise ValueError(
            "tensorflow_version and pytorch_version are both not None. "
            "Specify only tensorflow_version or pytorch_version."
        )
    if tensorflow_version is None and pytorch_version is None:
        raise ValueError(
            "tensorflow_version and pytorch_version are both None. "
            "Specify either tensorflow_version or pytorch_version."
        )


def fetch_framework_and_framework_version(tensorflow_version, pytorch_version):
    """Function to check the framework used in HuggingFace class"""

    if tensorflow_version is not None:  # pylint: disable=no-member
        return ("tensorflow", tensorflow_version)  # pylint: disable=no-member
    return ("pytorch", pytorch_version)  # pylint: disable=no-member


class HuggingFaceModel(FrameworkModel):
    """A Hugging Face SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    _framework_name = "huggingface"

    def __init__(
        self,
        role: Optional[str] = None,
        model_data: Optional[Union[str, PipelineVariable]] = None,
        entry_point: Optional[str] = None,
        transformers_version: Optional[str] = None,
        tensorflow_version: Optional[str] = None,
        pytorch_version: Optional[str] = None,
        py_version: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        predictor_cls: callable = HuggingFacePredictor,
        model_server_workers: Optional[Union[int, PipelineVariable]] = None,
        **kwargs,
    ):
        """Initialize a HuggingFaceModel.

        Args:
            model_data (str or PipelineVariable): The Amazon S3 location of a SageMaker
                model data ``.tar.gz`` file.
            role (str): An AWS IAM role specified with either the name or full ARN. The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            entry_point (str): The absolute or relative path to the Python source
                file that should be executed as the entry point to model
                hosting. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                Defaults to None.
            transformers_version (str): Transformers version you want to use for
                executing your model training code. Defaults to None. Required
                unless ``image_uri`` is provided.
            tensorflow_version (str): TensorFlow version you want to use for
                executing your inference code. Defaults to ``None``. Required unless
                ``pytorch_version`` is provided. List of supported versions:
                https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators.
            pytorch_version (str): PyTorch version you want to use for
                executing your inference code. Defaults to ``None``. Required unless
                ``tensorflow_version`` is provided. List of supported versions:
                https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators.
            py_version (str): Python version you want to use for executing your
                model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided.
            image_uri (str or PipelineVariable): A Docker image URI. Defaults to None.
                If not specified, a default image for PyTorch will be used. If ``framework_version``
                or ``py_version`` are ``None``, then ``image_uri`` is required. If
                also ``None``, then a ``ValueError`` will be raised.
            predictor_cls (callable[str, sagemaker.session.Session]): A function
                to call to create a predictor with an endpoint name and
                SageMaker ``Session``. If specified, ``deploy()`` returns the
                result of invoking this function on the created endpoint name.
            model_server_workers (int or PipelineVariable): Optional. The number of
                worker processes used by the inference server. If None, server will use one
                worker per vCPU.
            **kwargs: Keyword arguments passed to the superclass
                :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
                superclass :class:`~sagemaker.model.Model`.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.FrameworkModel` and
            :class:`~sagemaker.model.Model`.
        """
        validate_version_or_image_args(transformers_version, py_version, image_uri)
        _validate_pt_tf_versions(
            pytorch_version=pytorch_version,
            tensorflow_version=tensorflow_version,
            image_uri=image_uri,
        )
        if py_version == "py2":
            raise ValueError("py2 is not supported with HuggingFace images")
        self.framework_version = transformers_version
        self.pytorch_version = pytorch_version
        self.tensorflow_version = tensorflow_version
        self.py_version = py_version

        super(HuggingFaceModel, self).__init__(
            model_data, image_uri, role, entry_point, predictor_cls=predictor_cls, **kwargs
        )
        self.sagemaker_session = self.sagemaker_session or Session()

        self.model_server_workers = model_server_workers

    # TODO: Remove the following function
    # botocore needs to add hugginface to the list of valid neo compilable frameworks.
    # Ideally with inferentia framewrok, call to .compile( ... ) method will create the image_uri.
    # currently, call to compile( ... ) method is causing `ValidationException`
    def deploy(
        self,
        initial_instance_count=None,
        instance_type=None,
        serializer=None,
        deserializer=None,
        accelerator_type=None,
        endpoint_name=None,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config=None,
        async_inference_config=None,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        inference_recommendation_id=None,
        explainer_config=None,
        **kwargs,
    ):
        """Deploy this ``Model`` to an ``Endpoint`` and optionally return a ``Predictor``.

        Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an
        ``Endpoint`` from this ``Model``. If ``self.predictor_cls`` is not None,
        this method returns a the result of invoking ``self.predictor_cls`` on
        the created endpoint name.

        The name of the created model is accessible in the ``name`` field of
        this ``Model`` after deploy returns

        The name of the created endpoint is accessible in the
        ``endpoint_name`` field of this ``Model`` after deploy returns.

        Args:
            initial_instance_count (int): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. If not using
                serverless inference, then it need to be a number larger or equals
                to 1 (default: None)
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference, then it is required to deploy a model.
                (default: None)
            serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
                serializer object, used to encode data for an inference endpoint
                (default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``.
            deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
                deserializer object, used to decode data from an inference
                endpoint (default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``.
            accelerator_type (str): Type of Elastic Inference accelerator to
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
            tags (List[dict[str, str]]): The list of tags to attach to this
                specific endpoint.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint.
            wait (bool): Whether the call should wait until the deployment of
                this model completes (default: True).
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.
            async_inference_config (sagemaker.model_monitor.AsyncInferenceConfig): Specifies
                configuration related to async endpoint. Use this configuration when trying
                to create async endpoint and make async inference. If empty config object
                passed through, will use default config to deploy async endpoint. Deploy a
                real-time endpoint if it's None. (default: None)
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Use this configuration
                when trying to create serverless endpoint and make serverless inference. If
                empty object passed through, will use pre-defined values in
                ``ServerlessInferenceConfig`` class to deploy serverless endpoint. Deploy an
                instance based endpoint if it's None. (default: None)
            volume_size (int): The size, in GB, of the ML storage volume attached to individual
                inference instance associated with the production variant. Currenly only Amazon EBS
                gp2 storage volumes are supported.
            model_data_download_timeout (int): The timeout value, in seconds, to download and
                extract model data from Amazon S3 to the individual inference instance associated
                with this production variant.
            container_startup_health_check_timeout (int): The timeout value, in seconds, for your
                inference container to pass health check by SageMaker Hosting. For more information
                about health check see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-algo-ping-requests
            inference_recommendation_id (str): The recommendation id which specifies the
                recommendation you picked from inference recommendation job results and
                would like to deploy the model and endpoint with recommended parameters.
            explainer_config (sagemaker.explainer.ExplainerConfig): Specifies online explainability
                configuration for use with Amazon SageMaker Clarify. (default: None)
        Raises:
             ValueError: If arguments combination check failed in these circumstances:
                - If no role is specified or
                - If serverless inference config is not specified and instance type and instance
                    count are also not specified or
                - If a wrong type of object is provided as serverless inference config or async
                    inference config
        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of
                ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
                is not None. Otherwise, return None.
        """

        if not self.image_uri and instance_type is not None and instance_type.startswith("ml.inf"):
            inference_tool = "neuron" if instance_type.startswith("ml.inf1") else "neuronx"
            self.image_uri = self.serving_image_uri(
                region_name=self.sagemaker_session.boto_session.region_name,
                instance_type=instance_type,
                inference_tool=inference_tool,
            )

        return super(HuggingFaceModel, self).deploy(
            initial_instance_count,
            instance_type,
            serializer,
            deserializer,
            accelerator_type,
            endpoint_name,
            tags,
            kms_key,
            wait,
            data_capture_config,
            async_inference_config,
            serverless_inference_config,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
            inference_recommendation_id=inference_recommendation_id,
            explainer_config=explainer_config,
        )

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
                types that are used to generate inferences in real-time (default: None).
            transform_instances (list[str] or list[PipelineVariable]): A list of the instance types
                on which a transformation job can be run or on which an endpoint can be deployed
                (default: None).
            model_package_name (str or PipelineVariable): Model Package name, exclusive to
                `model_package_group_name`, using `model_package_name` makes the Model Package
                un-versioned. Defaults to ``None``.
            model_package_group_name (str or PipelineVariable): Model Package Group name,
                exclusive to `model_package_name`, using `model_package_group_name` makes the
                Model Package versioned. Defaults to ``None``.
            image_uri (str or PipelineVariable): Inference image URI for the container. Model class'
                self.image will be used if it is None. Defaults to ``None``.
            model_metrics (ModelMetrics): ModelMetrics object. Defaults to ``None``.
            metadata_properties (MetadataProperties): MetadataProperties object.
                Defaults to ``None``.
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace. Defaults to ``False``.
            approval_status (str or PipelineVariable): Model Approval Status, values can be
                "Approved", "Rejected", or "PendingManualApproval". Defaults to
                ``PendingManualApproval``.
            description (str): Model Package description. Defaults to ``None``.
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
            A `sagemaker.model.ModelPackage` instance.
        """
        instance_type = inference_instances[0] if inference_instances else None
        self._init_sagemaker_session_if_does_not_exist(instance_type)

        if image_uri:
            self.image_uri = image_uri
        if not self.image_uri:
            self.image_uri = self.serving_image_uri(
                region_name=self.sagemaker_session.boto_session.region_name,
                instance_type=instance_type,
            )
        if not is_pipeline_variable(framework):
            framework = (
                framework
                or fetch_framework_and_framework_version(
                    self.tensorflow_version, self.pytorch_version
                )[0]
            ).upper()
        return super(HuggingFaceModel, self).register(
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
            framework_version=framework_version
            or fetch_framework_and_framework_version(self.tensorflow_version, self.pytorch_version)[
                1
            ],
            nearest_model_name=nearest_model_name,
            data_input_configuration=data_input_configuration,
        )

    def prepare_container_def(
        self,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
        inference_tool=None,
    ):
        """A container definition with framework configuration set in model environment variables.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.
            inference_tool (str): the tool that will be used to aid in the inference.
                Valid values: "neuron, neuronx, None" (default: None).

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
                inference_tool=inference_tool,
            )

        deploy_key_prefix = model_code_key_prefix(self.key_prefix, self.name, deploy_image)
        self._upload_code(deploy_key_prefix, repack=True)
        deploy_env = dict(self.env)
        deploy_env.update(self._script_mode_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = to_string(
                self.model_server_workers
            )
        return sagemaker.container_def(
            deploy_image, self.repacked_model_data or self.model_data, deploy_env
        )

    def serving_image_uri(
        self,
        region_name,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
        inference_tool=None,
    ):
        """Create a URI for the serving image.

        Args:
            region_name (str): AWS region where the image is uploaded.
            instance_type (str): SageMaker instance type. Used to determine device type
                (cpu/gpu/family-specific optimized).
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used used to determine device type.
            inference_tool (str): the tool that will be used to aid in the inference.
                Valid values: "neuron, neuronx, None" (default: None).

        Returns:
            str: The appropriate image URI based on the given parameters.

        """
        if self.tensorflow_version is not None:  # pylint: disable=no-member
            base_framework_version = (
                f"tensorflow{self.tensorflow_version}"  # pylint: disable=no-member
            )
        else:
            base_framework_version = f"pytorch{self.pytorch_version}"  # pylint: disable=no-member
        return image_uris.retrieve(
            self._framework_name,
            region_name,
            version=self.framework_version,
            py_version=self.py_version,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            image_scope="inference",
            base_framework_version=base_framework_version,
            serverless_inference_config=serverless_inference_config,
            inference_tool=inference_tool,
        )
