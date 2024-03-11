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

from typing import Optional, Dict, List, Union

import sagemaker
from sagemaker import ModelMetrics, Model
from sagemaker.config import (
    ENDPOINT_CONFIG_KMS_KEY_ID_PATH,
    MODEL_VPC_CONFIG_PATH,
    MODEL_ENABLE_NETWORK_ISOLATION_PATH,
    MODEL_EXECUTION_ROLE_ARN_PATH,
    load_sagemaker_config,
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.session import Session
from sagemaker.utils import (
    name_from_image,
    update_container_with_inference_params,
    resolve_value_from_config,
)
from sagemaker.transformer import Transformer
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.pipeline_context import runnable_by_pipeline
from sagemaker.utils import instance_supports_kms


class PipelineModel(object):
    """A pipeline of SageMaker `Model` instances.

    This pipeline can be deployed as an `Endpoint` on SageMaker.
    """

    def __init__(
        self,
        models: List[Model],
        role: str = None,
        predictor_cls: Optional[callable] = None,
        name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None,
        sagemaker_session: Optional[Session] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
    ):
        """Initialize a SageMaker `Model` instance.

        The `Model` can be used to build an Inference Pipeline comprising of
        multiple model containers.

        Args:
            models (list[sagemaker.Model]): For using multiple containers to
                build an inference pipeline, you can pass a list of
                ``sagemaker.Model`` objects in the order you want the inference
                to happen.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name.
            name (str): The model name. If None, a default model name will be
                selected on each ``deploy``.
            vpc_config (dict[str, list[str]] or dict[str, list[PipelineVariable]]):
                The VpcConfig set on the model (default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            enable_network_isolation (bool or PipelineVariable): Default False. if True,
                enables network isolation in the endpoint, isolating the model
                container. No inbound or outbound network calls can be made to
                or from the model container.Boolean
        """
        self.models = models
        self.predictor_cls = predictor_cls
        self.name = name
        self.endpoint_name = None
        self.sagemaker_session = sagemaker_session

        # In case, sagemaker_session is None, get sagemaker_config from load_sagemaker_config()
        # to resolve value from config for the respective parameter
        self._sagemaker_config = (
            load_sagemaker_config() if (self.sagemaker_session is None) else None
        )

        self.role = resolve_value_from_config(
            role,
            MODEL_EXECUTION_ROLE_ARN_PATH,
            sagemaker_session=self.sagemaker_session,
            sagemaker_config=self._sagemaker_config,
        )
        self.vpc_config = resolve_value_from_config(
            vpc_config,
            MODEL_VPC_CONFIG_PATH,
            sagemaker_session=self.sagemaker_session,
            sagemaker_config=self._sagemaker_config,
        )
        self.enable_network_isolation = resolve_value_from_config(
            direct_input=enable_network_isolation,
            config_path=MODEL_ENABLE_NETWORK_ISOLATION_PATH,
            default_value=False,
            sagemaker_session=self.sagemaker_session,
            sagemaker_config=self._sagemaker_config,
        )

        if not self.role:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create a Pipeline Model.")

    def pipeline_container_def(self, instance_type=None):
        """The pipeline definition for deploying this model.

        This is the dict created by ``sagemaker.pipeline_container_def()``.

        The instance type to be used may be specified.

        Subclasses can override this to provide custom container definitions
        for deployment to a specific instance type. Called by ``deploy()``.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.

        Returns:
            list[dict[str, str]]: A list of container definition objects usable
            with the CreateModel API in the scenario of multiple containers
            (Inference Pipeline).
        """

        return sagemaker.pipeline_container_def(self.models, instance_type)

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        serializer=None,
        deserializer=None,
        endpoint_name=None,
        tags=None,
        wait=True,
        update_endpoint=False,
        data_capture_config=None,
        kms_key=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    ):
        """Deploy the ``Model`` to an ``Endpoint``.

        It optionally return a ``Predictor``.

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
                in the ``Endpoint`` created from this ``Model``.
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
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
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
            tags (List[dict[str, str]]): The list of tags to attach to this
                specific endpoint.
            wait (bool): Whether the call should wait until the deployment of
                model completes (default: True).
            update_endpoint (bool): Flag to update the model in an existing
                Amazon SageMaker endpoint. If True, this will deploy a new
                EndpointConfig to an already existing endpoint and delete
                resources corresponding to the previous EndpointConfig. If
                False, a new endpoint will be created. Default: False
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.
            kms_key (str): The ARN, Key ID or Alias of the KMS key that is used to
                encrypt the data on the storage volume attached to the instance hosting
                the endpoint.
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

        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of
            ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
            is not None. Otherwise, return None.
        """
        if not self.sagemaker_session:
            self.sagemaker_session = Session(sagemaker_config=self._sagemaker_config)

        containers = self.pipeline_container_def(instance_type)

        self.name = self.name or name_from_image(containers[0]["Image"])
        self.sagemaker_session.create_model(
            self.name,
            self.role,
            containers,
            vpc_config=self.vpc_config,
            enable_network_isolation=self.enable_network_isolation,
        )

        production_variant = sagemaker.production_variant(
            self.name,
            instance_type,
            initial_instance_count,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
        )
        self.endpoint_name = endpoint_name or self.name
        kms_key = (
            resolve_value_from_config(
                kms_key, ENDPOINT_CONFIG_KMS_KEY_ID_PATH, sagemaker_session=self.sagemaker_session
            )
            if instance_supports_kms(instance_type)
            else kms_key
        )

        data_capture_config_dict = None
        if data_capture_config is not None:
            data_capture_config_dict = data_capture_config._to_request_dict()

        if update_endpoint:
            endpoint_config_name = self.sagemaker_session.create_endpoint_config(
                name=self.name,
                model_name=self.name,
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                tags=tags,
                kms_key=kms_key,
                data_capture_config_dict=data_capture_config_dict,
                volume_size=volume_size,
                model_data_download_timeout=model_data_download_timeout,
                container_startup_health_check_timeout=container_startup_health_check_timeout,
            )
            self.sagemaker_session.update_endpoint(
                self.endpoint_name, endpoint_config_name, wait=wait
            )
        else:
            self.sagemaker_session.endpoint_from_production_variants(
                name=self.endpoint_name,
                production_variants=[production_variant],
                tags=tags,
                kms_key=kms_key,
                wait=wait,
                data_capture_config_dict=data_capture_config_dict,
            )

        if self.predictor_cls:
            predictor = self.predictor_cls(self.endpoint_name, self.sagemaker_session)
            if serializer:
                predictor.serializer = serializer
            if deserializer:
                predictor.deserializer = deserializer
            return predictor
        return None

    @runnable_by_pipeline
    def create(self, instance_type: str):
        """Create a SageMaker Model Entity

        Args:
            instance_type (str): The EC2 instance type that this Model will be
                used for, this is only used to determine if the image needs GPU
                support or not.
        """
        self._create_sagemaker_pipeline_model(instance_type)

    def _create_sagemaker_pipeline_model(self, instance_type):
        """Create a SageMaker Model Entity

        Args:
            instance_type (str): The EC2 instance type that this Model will be
                used for, this is only used to determine if the image needs GPU
                support or not.
        """
        if not self.sagemaker_session:
            self.sagemaker_session = Session(sagemaker_config=self._sagemaker_config)

        containers = self.pipeline_container_def(instance_type)

        self.name = self.name or name_from_image(containers[0]["Image"])
        create_model_args = dict(
            name=self.name,
            role=self.role,
            container_defs=containers,
            vpc_config=self.vpc_config,
            enable_network_isolation=self.enable_network_isolation,
        )
        self.sagemaker_session.create_model(**create_model_args)

    @runnable_by_pipeline
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
                un-versioned (default: None).
            model_package_group_name (str or PipelineVariable): Model Package Group name,
                exclusive to `model_package_name`, using `model_package_group_name` makes
                the Model Package versioned (default: None).
            image_uri (str or PipelineVariable): Inference image uri for the container.
                Model class' self.image will be used if it is None (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties object (default: None).
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace (default: False).
            approval_status (str or PipelineVariable): Model Approval Status, values can
                be "Approved", "Rejected", or "PendingManualApproval"
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
        for model in self.models:
            if model.model_data is None:
                raise ValueError("SageMaker Model Package cannot be created without model data.")
        if model_package_group_name is not None:
            container_def = self.pipeline_container_def(
                inference_instances[0] if inference_instances else None
            )
            container_def = update_container_with_inference_params(
                framework=framework,
                framework_version=framework_version,
                nearest_model_name=nearest_model_name,
                data_input_configuration=data_input_configuration,
                container_list=container_def,
            )
        else:
            container_def = [
                {
                    "Image": image_uri or model.image_uri,
                    "ModelDataUrl": model.model_data,
                }
                for model in self.models
            ]

        model_pkg_args = sagemaker.get_model_package_args(
            content_types,
            response_types,
            inference_instances=inference_instances,
            transform_instances=transform_instances,
            model_package_name=model_package_name,
            model_package_group_name=model_package_group_name,
            model_metrics=model_metrics,
            metadata_properties=metadata_properties,
            marketplace_cert=marketplace_cert,
            approval_status=approval_status,
            description=description,
            container_def_list=container_def,
            drift_check_baselines=drift_check_baselines,
            customer_metadata_properties=customer_metadata_properties,
            domain=domain,
            sample_payload_url=sample_payload_url,
            task=task,
        )

        self.sagemaker_session.create_model_package_from_containers(**model_pkg_args)

    def transformer(
        self,
        instance_count,
        instance_type,
        strategy=None,
        assemble_with=None,
        output_path=None,
        output_kms_key=None,
        accept=None,
        env=None,
        max_concurrent_transforms=None,
        max_payload=None,
        tags=None,
        volume_kms_key=None,
    ):
        """Return a ``Transformer`` that uses this Model.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in
                a single request (default: None). Valid values: 'MultiRecord'
                and 'SingleRecord'.
            assemble_with (str): How the output is assembled (default: None).
                Valid values: 'Line' or 'None'.
            output_path (str): S3 location for saving the transform result. If
                not specified, results are stored to a default bucket.
            output_kms_key (str): Optional. KMS key ID for encrypting the
                transform output (default: None).
            accept (str): The accept header passed by the client to
                the inference endpoint. If it is supported by the endpoint,
                it will be the format of the batch transform output.
            env (dict): Environment variables to be set for use during the
                transform job (default: None).
            max_concurrent_transforms (int): The maximum number of HTTP requests
                to be made to each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP
                request to the container in MB.
            tags (list[dict]): List of tags for labeling a transform job. If
                none specified, then the tags used for the training job are used
                for the transform job.
            volume_kms_key (str): Optional. KMS key ID for encrypting the volume
                attached to the ML compute instance (default: None).
        """
        self._create_sagemaker_pipeline_model(instance_type)

        return Transformer(
            self.name,
            instance_count,
            instance_type,
            strategy=strategy,
            assemble_with=assemble_with,
            output_path=output_path,
            output_kms_key=output_kms_key,
            accept=accept,
            max_concurrent_transforms=max_concurrent_transforms,
            max_payload=max_payload,
            env=env,
            tags=tags,
            base_transform_job_name=self.name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=self.sagemaker_session,
        )

    def delete_model(self):
        """Delete the SageMaker model backing this pipeline model.

        This does not delete the list of SageMaker models used in multiple containers to build
        the inference pipeline.
        """

        if self.name is None:
            raise ValueError("The SageMaker model must be created before attempting to delete.")

        self.sagemaker_session.delete_model(self.name)
