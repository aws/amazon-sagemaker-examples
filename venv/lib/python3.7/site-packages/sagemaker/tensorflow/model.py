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
"""Classes for using TensorFlow on Amazon SageMaker for inference."""
from __future__ import absolute_import

import logging
from typing import Union, Optional, List, Dict

import sagemaker
from sagemaker import image_uris, s3, ModelMetrics
from sagemaker.deserializers import JSONDeserializer
from sagemaker.deprecations import removed_kwargs
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.pipeline_context import PipelineSession

logger = logging.getLogger(__name__)


class TensorFlowPredictor(Predictor):
    """A ``Predictor`` implementation for inference against TensorFlow Serving endpoints."""

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        model_name=None,
        model_version=None,
        **kwargs,
    ):
        """Initialize a ``TensorFlowPredictor``.

        See :class:`~sagemaker.predictor.Predictor` for more info about parameters.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (callable): Optional. Default serializes input data to
                json. Handles dicts, lists, and numpy arrays.
            deserializer (callable): Optional. Default parses the response using
                ``json.load(...)``.
            model_name (str): Optional. The name of the SavedModel model that
                should handle the request. If not specified, the endpoint's
                default model will handle the request.
            model_version (str): Optional. The version of the SavedModel model
                that should handle the request. If not specified, the latest
                version of the model will be used.
        """
        removed_kwargs("content_type", kwargs)
        removed_kwargs("accept", kwargs)
        super(TensorFlowPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer,
            deserializer,
        )

        attributes = []
        if model_name:
            attributes.append("tfs-model-name={}".format(model_name))
        if model_version:
            attributes.append("tfs-model-version={}".format(model_version))
        self._model_attributes = ",".join(attributes) if attributes else None

    def classify(self, data):
        """Placeholder docstring."""
        return self._classify_or_regress(data, "classify")

    def regress(self, data):
        """Placeholder docstring."""
        return self._classify_or_regress(data, "regress")

    def _classify_or_regress(self, data, method):
        """Placeholder docstring."""
        if method not in ["classify", "regress"]:
            raise ValueError("invalid TensorFlow Serving method: {}".format(method))

        if self.content_type != "application/json":
            raise ValueError("The {} api requires json requests.".format(method))

        args = {"CustomAttributes": "tfs-method={}".format(method)}

        return self.predict(data, args)

    def predict(self, data, initial_args=None):
        """Placeholder docstring."""
        args = dict(initial_args) if initial_args else {}
        if self._model_attributes:
            if "CustomAttributes" in args:
                args["CustomAttributes"] += "," + self._model_attributes
            else:
                args["CustomAttributes"] = self._model_attributes

        return super(TensorFlowPredictor, self).predict(data, args)


class TensorFlowModel(sagemaker.model.FrameworkModel):
    """A ``FrameworkModel`` implementation for inference with TensorFlow Serving."""

    _framework_name = "tensorflow"
    LOG_LEVEL_PARAM_NAME = "SAGEMAKER_TFS_NGINX_LOGLEVEL"
    LOG_LEVEL_MAP = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warn",
        logging.ERROR: "error",
        logging.CRITICAL: "crit",
    }
    LATEST_EIA_VERSION = [2, 3]

    def __init__(
        self,
        model_data: Union[str, PipelineVariable],
        role: str = None,
        entry_point: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        framework_version: Optional[str] = None,
        container_log_level: Optional[int] = None,
        predictor_cls: callable = TensorFlowPredictor,
        **kwargs,
    ):
        """Initialize a Model.

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
                If not specified, a default image for TensorFlow Serving will be used.
                If ``framework_version`` is ``None``, then ``image_uri`` is required.
                If ``image_uri`` is also ``None``, then a ``ValueError``
                will be raised.
            framework_version (str): Optional. TensorFlow Serving version you
                want to use. Defaults to ``None``. Required unless ``image_uri`` is
                provided.
            container_log_level (int): Log level to use within the container
                (default: logging.ERROR). Valid values are defined in the Python
                logging module.
            predictor_cls (callable[str, sagemaker.session.Session]): A function
                to call to create a predictor with an endpoint name and
                SageMaker ``Session``. If specified, ``deploy()`` returns the
                result of invoking this function on the created endpoint name.
            **kwargs: Keyword arguments passed to the superclass
                :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
                superclass :class:`~sagemaker.model.Model`.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.FrameworkModel` and
            :class:`~sagemaker.model.Model`.
        """
        if framework_version is None and image_uri is None:
            raise ValueError(
                "Both framework_version and image_uri were None. "
                "Either specify framework_version or specify image_uri."
            )
        self.framework_version = framework_version

        # Inference framework version is being introduced to accomodate the mismatch between
        # tensorflow and tensorflow serving releases, wherein the TF and TFS might have different
        # patch versions, but end up hosting the model of same TF version. For eg., the upstream
        # TFS-2.12.0 release was a bad release and hence a new TFS-2.12.1 release was made to host
        # models from TF-2.12.0.
        training_inference_version_mismatch_dict = {"2.12.0": "2.12.1"}
        self.inference_framework_version = training_inference_version_mismatch_dict.get(
            framework_version, framework_version
        )

        super(TensorFlowModel, self).__init__(
            model_data=model_data,
            role=role,
            image_uri=image_uri,
            predictor_cls=predictor_cls,
            entry_point=entry_point,
            **kwargs,
        )
        self._container_log_level = container_log_level

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
            transform_instances (list[str] or list[PipelineVariable]): A list of the instance
                types on which a transformation job can be run or on which an endpoint can
                be deployed (default: None).
            model_package_name (str or PipelineVariable): Model Package name, exclusive to
                `model_package_group_name`, using `model_package_name` makes the Model Package
                un-versioned (default: None).
            model_package_group_name (str or PipelineVariable): Model Package Group name,
                exclusive to `model_package_name`, using `model_package_group_name` makes the
                Model Package versioned (default: None).
            image_uri (str or PipelineVariable): Inference image uri for the container. Model class'
                self.image will be used if it is None (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties object (default: None).
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
            framework = (framework or self._framework_name).upper()
        return super(TensorFlowModel, self).register(
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
        """Deploy a Tensorflow ``Model`` to a SageMaker ``Endpoint``."""

        if accelerator_type and not self._eia_supported():
            msg = "The TensorFlow version %s doesn't support EIA." % self.framework_version
            raise AttributeError(msg)

        return super(TensorFlowModel, self).deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            serializer=serializer,
            deserializer=deserializer,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
            tags=tags,
            kms_key=kms_key,
            wait=wait,
            data_capture_config=data_capture_config,
            async_inference_config=async_inference_config,
            serverless_inference_config=serverless_inference_config,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
            inference_recommendation_id=inference_recommendation_id,
            explainer_config=explainer_config,
            **kwargs,
        )

    def _eia_supported(self):
        """Return true if TF version is EIA enabled"""
        framework_version = [int(s) for s in self.framework_version.split(".")][:2]
        return (
            framework_version != [2, 1]
            and framework_version != [2, 2]
            and framework_version <= self.LATEST_EIA_VERSION
        )

    def prepare_container_def(
        self, instance_type=None, accelerator_type=None, serverless_inference_config=None
    ):
        """Prepare the container definition.

        Args:
            instance_type: Instance type of the container.
            accelerator_type: Accelerator type, if applicable.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.

        Returns:
            A container definition for deploying a ``Model`` to an ``Endpoint``.
        """
        if not self.image_uri:
            if instance_type is None and serverless_inference_config is None:
                raise ValueError(
                    "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
                )

        image_uri = self._get_image_uri(
            instance_type, accelerator_type, serverless_inference_config=serverless_inference_config
        )
        env = self._get_container_env()

        bucket, key_prefix = s3.determine_bucket_and_prefix(
            bucket=self.bucket,
            key_prefix=sagemaker.fw_utils.model_code_key_prefix(
                self.key_prefix, self.name, image_uri
            ),
            sagemaker_session=self.sagemaker_session,
        )

        if self.entry_point and not is_pipeline_variable(self.model_data):
            model_data = s3.s3_path_join("s3://", bucket, key_prefix, "model.tar.gz")

            sagemaker.utils.repack_model(
                self.entry_point,
                self.source_dir,
                self.dependencies,
                self.model_data,
                model_data,
                self.sagemaker_session,
                kms_key=self.model_kms_key,
            )
        elif self.entry_point and is_pipeline_variable(self.model_data):
            # model is not yet there, defer repacking to later during pipeline execution
            if isinstance(self.sagemaker_session, PipelineSession):
                self.sagemaker_session.context.need_runtime_repack.add(id(self))
                self.sagemaker_session.context.runtime_repack_output_prefix = s3.s3_path_join(
                    "s3://", bucket, key_prefix
                )
            else:
                logging.warning(
                    "The model_data is a Pipeline variable of type %s, "
                    "which should be used under `PipelineSession` and "
                    "leverage `ModelStep` to create or register model. "
                    "Otherwise some functionalities e.g. "
                    "runtime repack may be missing. For more, see: "
                    "https://sagemaker.readthedocs.io/en/stable/"
                    "amazon_sagemaker_model_building_pipeline.html#model-step",
                    type(self.model_data),
                )
            model_data = self.model_data
        else:
            model_data = self.model_data

        return sagemaker.container_def(image_uri, model_data, env)

    def _get_container_env(self):
        """Placeholder docstring."""
        if not self._container_log_level:
            return self.env

        if self._container_log_level not in self.LOG_LEVEL_MAP:
            logging.warning("ignoring invalid container log level: %s", self._container_log_level)
            return self.env

        env = dict(self.env)
        env[self.LOG_LEVEL_PARAM_NAME] = self.LOG_LEVEL_MAP[self._container_log_level]
        return env

    def _get_image_uri(
        self,
        instance_type,
        accelerator_type=None,
        region_name=None,
        serverless_inference_config=None,
    ):
        """Placeholder docstring."""
        if self.image_uri:
            return self.image_uri

        logger.info(
            "image_uri is not presented, retrieving image_uri based on instance_type, "
            "framework etc."
        )
        return image_uris.retrieve(
            self._framework_name,
            region_name or self.sagemaker_session.boto_region_name,
            version=self.inference_framework_version,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            image_scope="inference",
            serverless_inference_config=serverless_inference_config,
        )

    def serving_image_uri(
        self, region_name, instance_type, accelerator_type=None, serverless_inference_config=None
    ):  # pylint: disable=unused-argument
        """Create a URI for the serving image.

        Args:
            region_name (str): AWS region where the image is uploaded.
            instance_type (str): SageMaker instance type. Used to determine device type
                (cpu/gpu/family-specific optimized).
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model (default: None). For example, 'ml.eia1.medium'.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to determine device type.

        Returns:
            str: The appropriate image URI based on the given parameters.

        """
        return self._get_image_uri(
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            region_name=region_name,
            serverless_inference_config=serverless_inference_config,
        )
