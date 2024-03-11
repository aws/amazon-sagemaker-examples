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

import abc
import json
import logging
import os
import re
import copy
from typing import List, Dict, Optional, Union

import sagemaker
from sagemaker import (
    fw_utils,
    local,
    s3,
    session,
    utils,
    git_utils,
)
from sagemaker.config import (
    COMPILATION_JOB_ROLE_ARN_PATH,
    EDGE_PACKAGING_KMS_KEY_ID_PATH,
    EDGE_PACKAGING_ROLE_ARN_PATH,
    MODEL_CONTAINERS_PATH,
    EDGE_PACKAGING_RESOURCE_KEY_PATH,
    MODEL_VPC_CONFIG_PATH,
    MODEL_ENABLE_NETWORK_ISOLATION_PATH,
    MODEL_EXECUTION_ROLE_ARN_PATH,
    MODEL_PRIMARY_CONTAINER_ENVIRONMENT_PATH,
    ENDPOINT_CONFIG_ASYNC_KMS_KEY_ID_PATH,
    load_sagemaker_config,
)
from sagemaker.session import Session
from sagemaker.model_metrics import ModelMetrics
from sagemaker.deprecations import removed_kwargs
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.explainer import ExplainerConfig
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.predictor import PredictorBase
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.transformer import Transformer
from sagemaker.jumpstart.utils import (
    add_jumpstart_tags,
    get_jumpstart_base_name_if_jumpstart_model,
)
from sagemaker.utils import (
    unique_name_from_base,
    update_container_with_inference_params,
    to_string,
    resolve_value_from_config,
    resolve_nested_dict_value_from_config,
)
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.pipeline_context import runnable_by_pipeline, PipelineSession
from sagemaker.inference_recommender.inference_recommender_mixin import (
    InferenceRecommenderMixin,
)

LOGGER = logging.getLogger("sagemaker")

NEO_ALLOWED_FRAMEWORKS = set(
    ["mxnet", "tensorflow", "keras", "pytorch", "onnx", "xgboost", "tflite"]
)

NEO_IOC_TARGET_DEVICES = [
    "ml_c4",
    "ml_c5",
    "ml_m4",
    "ml_m5",
    "ml_p2",
    "ml_p3",
    "ml_g4dn",
]

NEO_MULTIVERSION_UNSUPPORTED = [
    "imx8mplus",
    "jacinto_tda4vm",
    "coreml",
    "sitara_am57x",
    "amba_cv2",
    "amba_cv22",
    "amba_cv25",
    "lambda",
]


class ModelBase(abc.ABC):
    """An object that encapsulates a trained model.

    Models can be deployed to compute services like a SageMaker ``Endpoint``
    or Lambda. Deployed models can be used to perform real-time inference.
    """

    @abc.abstractmethod
    def deploy(self, *args, **kwargs) -> PredictorBase:
        """Deploy this model to a compute service."""

    @abc.abstractmethod
    def delete_model(self, *args, **kwargs) -> None:
        """Destroy resources associated with this model."""


SCRIPT_PARAM_NAME = "sagemaker_program"
DIR_PARAM_NAME = "sagemaker_submit_directory"
CONTAINER_LOG_LEVEL_PARAM_NAME = "sagemaker_container_log_level"
JOB_NAME_PARAM_NAME = "sagemaker_job_name"
MODEL_SERVER_WORKERS_PARAM_NAME = "sagemaker_model_server_workers"
SAGEMAKER_REGION_PARAM_NAME = "sagemaker_region"
SAGEMAKER_OUTPUT_LOCATION = "sagemaker_s3_output"


class Model(ModelBase, InferenceRecommenderMixin):
    """A SageMaker ``Model`` that can be deployed to an ``Endpoint``."""

    def __init__(
        self,
        image_uri: Union[str, PipelineVariable],
        model_data: Optional[Union[str, PipelineVariable, dict]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None,
        sagemaker_session: Optional[Session] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Union[int, PipelineVariable] = logging.INFO,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
    ):
        """Initialize an SageMaker ``Model``.

        Args:
            image_uri (str or PipelineVariable): A Docker image URI.
            model_data (str or PipelineVariable or dict): Location
                of SageMaker model data (default: None).
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role if it needs to access some AWS resources.
                It can be null if this is being used to create a Model to pass
                to a ``PipelineModel`` which has its own Role field. (default:
                None)
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name.
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables
                to run with ``image_uri`` when hosted in SageMaker (default: None).
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
            enable_network_isolation (Boolean or PipelineVariable): Default False.
                if True, enables network isolation in the endpoint, isolating the model
                container. No inbound or outbound network calls can be made to
                or from the model container.
            model_kms_key (str): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked
            image_config (dict[str, str] or dict[str, PipelineVariable]): Specifies
                whether the image of model container is pulled from ECR, or private
                registry in your VPC. By default it is set to pull model container
                image from ECR. (default: None).
            source_dir (str): The absolute, relative, or S3 URI Path to a directory
                with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory is preserved
                when training on Amazon SageMaker. If 'git_config' is provided,
                'source_dir' should be a relative location to a directory in the Git repo.
                If the directory points to S3, no code is uploaded and the S3 location
                is used instead.

                .. admonition:: Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='inference.py', source_dir='src'.
            code_location (str): Name of the S3 bucket where custom code is
                uploaded (default: None). If not specified, the default bucket
                created by ``sagemaker.session.Session`` is used.
            entry_point (str): The absolute or relative path to the local Python
                source file that should be executed as the entry point to
                model hosting. (Default: None). If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo.

                Example:
                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='src/inference.py'.
            container_log_level (int or PipelineVariable): Log level to use within the
                container (default: logging.INFO). Valid values are defined in the Python
                logging module.
            dependencies (list[str]): A list of absolute or relative paths to directories
                with any additional libraries that should be exported
                to the container (default: []). The library folders are
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo. If the ```source_dir``` points
                to S3, code will be uploaded and the S3 location will be used
                instead.

                .. admonition:: Example

                    The following call

                    >>> Model(entry_point='inference.py',
                    ...       dependencies=['my/libs/common', 'virtual-env'])

                    results in the following structure inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ inference.py
                    >>>     |------ common
                    >>>     |------ virtual-env

                This is not supported with "local code" in Local Mode.
            git_config (dict[str, str]): Git configurations used for cloning
                files, including ``repo``, ``branch``, ``commit``,
                ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                ``repo`` field is required. All other fields are optional.
                ``repo`` specifies the Git repository where your training script
                is stored. If you don't provide ``branch``, the default value
                'master' is used. If you don't provide ``commit``, the latest
                commit in the specified branch is used.

                .. admonition:: Example

                    The following config:

                    >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                    >>>               'branch': 'test-branch-git-config',
                    >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

                    results in cloning the repo specified in 'repo', then
                    checking out the 'master' branch, and checking out the specified
                    commit.

                ``2FA_enabled``, ``username``, ``password`` and ``token`` are
                used for authentication. For GitHub (or other Git) accounts, set
                ``2FA_enabled`` to 'True' if two-factor authentication is
                enabled for the account, otherwise set it to 'False'. If you do
                not provide a value for ``2FA_enabled``, a default value of
                'False' is used. CodeCommit does not support two-factor
                authentication, so do not provide "2FA_enabled" with CodeCommit
                repositories.

                For GitHub and other Git repos, when SSH URLs are provided, it
                doesn't matter whether 2FA is enabled or disabled. You should
                either have no passphrase for the SSH key pairs or have the
                ssh-agent configured so that you will not be prompted for the SSH
                passphrase when you run the 'git clone' command with SSH URLs. When
                HTTPS URLs are provided, if 2FA is disabled, then either ``token``
                or ``username`` and ``password`` are be used for authentication if provided.
                ``Token`` is prioritized. If 2FA is enabled, only ``token`` is used
                for authentication if provided. If required authentication info
                is not provided, the SageMaker Python SDK attempts to use local credentials
                to authenticate. If that fails, an error message is thrown.

                For CodeCommit repos, 2FA is not supported, so ``2FA_enabled``
                should not be provided. There is no token in CodeCommit, so
                ``token`` should also not be provided. When ``repo`` is an SSH URL,
                the requirements are the same as GitHub  repos. When ``repo``
                is an HTTPS URL, ``username`` and ``password`` are used for
                authentication if they are provided. If they are not provided,
                the SageMaker Python SDK attempts to use either the CodeCommit
                credential helper or local credential storage for authentication.

        """
        self.model_data = model_data
        self.image_uri = image_uri
        self.predictor_cls = predictor_cls
        self.name = name
        self._base_name = None
        self.sagemaker_session = sagemaker_session

        # Workaround for config injection if sagemaker_session is None, since in
        # that case sagemaker_session will not be initialized until
        # `_init_sagemaker_session_if_does_not_exist` is called later
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
        self.endpoint_name = None
        self._is_compiled_model = False
        self._compilation_job_name = None
        self._is_edge_packaged_model = False
        self.inference_recommender_job_results = None
        self.inference_recommendations = None
        self._enable_network_isolation = resolve_value_from_config(
            enable_network_isolation,
            MODEL_ENABLE_NETWORK_ISOLATION_PATH,
            default_value=False,
            sagemaker_session=self.sagemaker_session,
            sagemaker_config=self._sagemaker_config,
        )
        self.env = resolve_value_from_config(
            env,
            MODEL_PRIMARY_CONTAINER_ENVIRONMENT_PATH,
            default_value={},
            sagemaker_session=self.sagemaker_session,
            sagemaker_config=self._sagemaker_config,
        )
        self.model_kms_key = model_kms_key
        self.image_config = image_config
        self.entry_point = entry_point
        self.source_dir = source_dir
        self.dependencies = dependencies or []
        self.git_config = git_config
        self.container_log_level = container_log_level
        if code_location:
            self.bucket, self.key_prefix = s3.parse_s3_url(code_location)
        else:
            self.bucket, self.key_prefix = None, None
        if self.git_config:
            updates = git_utils.git_clone_repo(
                self.git_config, self.entry_point, self.source_dir, self.dependencies
            )
            self.entry_point = updates["entry_point"]
            self.source_dir = updates["source_dir"]
            self.dependencies = updates["dependencies"]
        self.uploaded_code = None
        self.repacked_model_data = None

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
        validation_specification: Optional[Union[str, PipelineVariable]] = None,
        domain: Optional[Union[str, PipelineVariable]] = None,
        task: Optional[Union[str, PipelineVariable]] = None,
        sample_payload_url: Optional[Union[str, PipelineVariable]] = None,
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
                types on which a transformation job can be run or on which an endpoint can be
                deployed (default: None).
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
            approval_status (str or PipelineVariable): Model Approval Status, values can be
                "Approved", "Rejected", or "PendingManualApproval"
                (default: "PendingManualApproval").
            description (str): Model Package description (default: None).
            drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
            customer_metadata_properties (dict[str, str] or dict[str, PipelineVariable]):
                A dictionary of key-value paired metadata properties (default: None).
            domain (str or PipelineVariable): Domain values can be "COMPUTER_VISION",
                "NATURAL_LANGUAGE_PROCESSING", "MACHINE_LEARNING" (default: None).
            task (str or PipelineVariable): Task values which are supported by Inference Recommender
                are "FILL_MASK", "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION",
                "IMAGE_SEGMENTATION", "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
            sample_payload_url (str or PipelineVariable): The S3 path where the sample
                payload is stored (default: None).
            framework (str or PipelineVariable): Machine learning framework of the model package
                container image (default: None).
            framework_version (str or PipelineVariable): Framework version of the Model Package
                Container Image (default: None).
            nearest_model_name (str or PipelineVariable): Name of a pre-trained machine learning
                benchmarked by Amazon SageMaker Inference Recommender (default: None).
            data_input_configuration (str or PipelineVariable): Input object for the model
                (default: None).

        Returns:
            A `sagemaker.model.ModelPackage` instance or pipeline step arguments
            in case the Model instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        """
        if self.model_data is None:
            raise ValueError("SageMaker Model Package cannot be created without model data.")
        if isinstance(self.model_data, dict):
            raise ValueError(
                "SageMaker Model Package currently cannot be created with ModelDataSource."
            )

        if image_uri is not None:
            self.image_uri = image_uri

        if model_package_group_name is not None:
            container_def = self.prepare_container_def()
            container_def = update_container_with_inference_params(
                framework=framework,
                framework_version=framework_version,
                nearest_model_name=nearest_model_name,
                data_input_configuration=data_input_configuration,
                container_def=container_def,
            )
        else:
            container_def = {
                "Image": self.image_uri,
                "ModelDataUrl": self.model_data,
            }

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
            container_def_list=[container_def],
            drift_check_baselines=drift_check_baselines,
            customer_metadata_properties=customer_metadata_properties,
            validation_specification=validation_specification,
            domain=domain,
            sample_payload_url=sample_payload_url,
            task=task,
        )
        model_package = self.sagemaker_session.create_model_package_from_containers(
            **model_pkg_args
        )
        if isinstance(self.sagemaker_session, PipelineSession):
            return None
        return ModelPackage(
            role=self.role,
            model_data=self.model_data,
            model_package_arn=model_package.get("ModelPackageArn"),
        )

    @runnable_by_pipeline
    def create(
        self,
        instance_type: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        serverless_inference_config: Optional[ServerlessInferenceConfig] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
    ):
        """Create a SageMaker Model Entity

        Args:
            instance_type (str): The EC2 instance type that this Model will be
                used for, this is only used to determine if the image needs GPU
                support or not (default: None).
            accelerator_type (str): Type of Elastic Inference accelerator to
                attach to an endpoint for model loading and inference, for
                example, 'ml.eia1.medium'. If not specified, no Elastic
                Inference accelerator will be attached to the endpoint (default: None).
            serverless_inference_config (ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs
                (default: None).
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): The list of
                tags to add to the model (default: None). Example::

                    tags = [{'Key': 'tagname', 'Value':'tagvalue'}]

                For more information about tags, see
                `boto3 documentation <https://boto3.amazonaws.com/v1/documentation/\
api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags>`_

        Returns:
            None or pipeline step arguments in case the Model instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        """
        # TODO: we should replace _create_sagemaker_model() with create()
        self._create_sagemaker_model(
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            tags=tags,
            serverless_inference_config=serverless_inference_config,
        )

    def _init_sagemaker_session_if_does_not_exist(self, instance_type=None):
        """Set ``self.sagemaker_session`` to ``LocalSession`` or ``Session`` if it's not already.

        The type of session object is determined by the instance type.
        """
        if self.sagemaker_session:
            return

        if instance_type in ("local", "local_gpu"):
            self.sagemaker_session = local.LocalSession(sagemaker_config=self._sagemaker_config)
        else:
            self.sagemaker_session = session.Session(sagemaker_config=self._sagemaker_config)

    def prepare_container_def(
        self,
        instance_type=None,
        accelerator_type=None,
        serverless_inference_config=None,
    ):  # pylint: disable=unused-argument
        """Return a dict created by ``sagemaker.container_def()``.

        It is used for deploying this model to a specified instance type.

        Subclasses can override this to provide custom container definitions
        for deployment to a specific instance type. Called by ``deploy()``.

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
            dict: A container definition object usable with the CreateModel API.
        """
        deploy_key_prefix = fw_utils.model_code_key_prefix(
            self.key_prefix, self.name, self.image_uri
        )
        deploy_env = copy.deepcopy(self.env)
        if self.source_dir or self.dependencies or self.entry_point or self.git_config:
            is_repack = (
                self.source_dir
                and self.entry_point
                and not (
                    (self.key_prefix and issubclass(type(self), FrameworkModel)) or self.git_config
                )
            )
            self._upload_code(deploy_key_prefix, repack=is_repack)
            deploy_env.update(self._script_mode_env_vars())

        return sagemaker.container_def(
            self.image_uri,
            self.repacked_model_data or self.model_data,
            deploy_env,
            image_config=self.image_config,
        )

    def _upload_code(self, key_prefix: str, repack: bool = False) -> None:
        """Uploads code to S3 to be used with script mode with SageMaker inference.

        Args:
            key_prefix (str): The S3 key associated with the ``code_location`` parameter of the
                ``Model`` class.
            repack (bool): Optional. Set to ``True`` to indicate that the source code and model
                artifact should be repackaged into a new S3 object. (default: False).
        """
        local_code = utils.get_config_value("local.local_code", self.sagemaker_session.config)

        bucket, key_prefix = s3.determine_bucket_and_prefix(
            bucket=self.bucket,
            key_prefix=key_prefix,
            sagemaker_session=self.sagemaker_session,
        )

        if (self.sagemaker_session.local_mode and local_code) or self.entry_point is None:
            self.uploaded_code = None
        elif not repack:
            self.uploaded_code = fw_utils.tar_and_upload_dir(
                session=self.sagemaker_session.boto_session,
                bucket=bucket,
                s3_key_prefix=key_prefix,
                script=self.entry_point,
                directory=self.source_dir,
                dependencies=self.dependencies,
                kms_key=self.model_kms_key,
                settings=self.sagemaker_session.settings,
            )

        if repack and self.model_data is not None and self.entry_point is not None:
            if isinstance(self.model_data, dict):
                logging.warning("ModelDataSource currently doesn't support model repacking")
                return
            if is_pipeline_variable(self.model_data):
                # model is not yet there, defer repacking to later during pipeline execution
                if not isinstance(self.sagemaker_session, PipelineSession):
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
                    return
                self.sagemaker_session.context.need_runtime_repack.add(id(self))
                self.sagemaker_session.context.runtime_repack_output_prefix = s3.s3_path_join(
                    "s3://", bucket, key_prefix
                )
                # Add the uploaded_code and repacked_model_data to update the container env
                self.repacked_model_data = self.model_data
                self.uploaded_code = fw_utils.UploadedCode(
                    s3_prefix=self.repacked_model_data,
                    script_name=os.path.basename(self.entry_point),
                )
                return
            if local_code and self.model_data.startswith("file://"):
                repacked_model_data = self.model_data
            else:
                repacked_model_data = "s3://" + "/".join([bucket, key_prefix, "model.tar.gz"])
                self.uploaded_code = fw_utils.UploadedCode(
                    s3_prefix=repacked_model_data,
                    script_name=os.path.basename(self.entry_point),
                )

            LOGGER.info(
                "Repacking model artifact (%s), script artifact "
                "(%s), and dependencies (%s) "
                "into single tar.gz file located at %s. "
                "This may take some time depending on model size...",
                self.model_data,
                self.source_dir,
                self.dependencies,
                repacked_model_data,
            )

            utils.repack_model(
                inference_script=self.entry_point,
                source_directory=self.source_dir,
                dependencies=self.dependencies,
                model_uri=self.model_data,
                repacked_model_uri=repacked_model_data,
                sagemaker_session=self.sagemaker_session,
                kms_key=self.model_kms_key,
            )

            self.repacked_model_data = repacked_model_data

    def _script_mode_env_vars(self):
        """Returns a mapping of environment variables for script mode execution"""
        script_name = None
        dir_name = None
        if self.uploaded_code:
            script_name = self.uploaded_code.script_name
            if self.repacked_model_data or self.enable_network_isolation():
                dir_name = "/opt/ml/model/code"
            else:
                dir_name = self.uploaded_code.s3_prefix
        elif self.entry_point is not None:
            script_name = self.entry_point
            if self.source_dir is not None:
                dir_name = (
                    self.source_dir
                    if self.source_dir.startswith("s3://")
                    else "file://" + self.source_dir
                )
        return {
            SCRIPT_PARAM_NAME.upper(): script_name or str(),
            DIR_PARAM_NAME.upper(): dir_name or str(),
            CONTAINER_LOG_LEVEL_PARAM_NAME.upper(): to_string(self.container_log_level),
            SAGEMAKER_REGION_PARAM_NAME.upper(): self.sagemaker_session.boto_region_name,
        }

    def enable_network_isolation(self):
        """Whether to enable network isolation when creating this Model

        Returns:
            bool: If network isolation should be enabled or not.
        """
        return False if not self._enable_network_isolation else self._enable_network_isolation

    def _create_sagemaker_model(
        self,
        instance_type=None,
        accelerator_type=None,
        tags=None,
        serverless_inference_config=None,
    ):
        """Create a SageMaker Model Entity

        Args:
            instance_type (str): The EC2 instance type that this Model will be
                used for, this is only used to determine if the image needs GPU
                support or not.
            accelerator_type (str): Type of Elastic Inference accelerator to
                attach to an endpoint for model loading and inference, for
                example, 'ml.eia1.medium'. If not specified, no Elastic
                Inference accelerator will be attached to the endpoint.
            tags (List[dict[str, str]]): Optional. The list of tags to add to
                the model. Example: >>> tags = [{'Key': 'tagname', 'Value':
                'tagvalue'}] For more information about tags, see
                https://boto3.amazonaws.com/v1/documentation
                /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.
        """
        container_def = self.prepare_container_def(
            instance_type,
            accelerator_type=accelerator_type,
            serverless_inference_config=serverless_inference_config,
        )

        if not isinstance(self.sagemaker_session, PipelineSession):
            # _base_name, model_name are not needed under PipelineSession.
            # the model_data may be Pipeline variable
            # which may break the _base_name generation
            model_uri = None
            if isinstance(self.model_data, (str, PipelineVariable)):
                model_uri = self.model_data
            elif isinstance(self.model_data, dict):
                model_uri = self.model_data.get("S3DataSource", {}).get("S3Uri", None)

            self._ensure_base_name_if_needed(
                image_uri=container_def["Image"],
                script_uri=self.source_dir,
                model_uri=model_uri,
            )
            self._set_model_name_if_needed()

        self._init_sagemaker_session_if_does_not_exist(instance_type)
        # Depending on the instance type, a local session (or) a session is initialized.
        self.role = resolve_value_from_config(
            self.role,
            MODEL_EXECUTION_ROLE_ARN_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.vpc_config = resolve_value_from_config(
            self.vpc_config,
            MODEL_VPC_CONFIG_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self._enable_network_isolation = resolve_value_from_config(
            self._enable_network_isolation,
            MODEL_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.env = resolve_nested_dict_value_from_config(
            self.env,
            ["Environment"],
            MODEL_CONTAINERS_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        create_model_args = dict(
            name=self.name,
            role=self.role,
            container_defs=container_def,
            vpc_config=self.vpc_config,
            enable_network_isolation=self._enable_network_isolation,
            tags=tags,
        )
        self.sagemaker_session.create_model(**create_model_args)

    def _ensure_base_name_if_needed(self, image_uri, script_uri, model_uri):
        """Create a base name from the image URI if there is no model name provided.

        If a JumpStart script or model uri is used, select the JumpStart base name.
        """
        if self.name is None:
            self._base_name = (
                self._base_name
                or get_jumpstart_base_name_if_jumpstart_model(script_uri, model_uri)
                or utils.base_name_from_image(image_uri, default_base_name=Model.__name__)
            )

    def _set_model_name_if_needed(self):
        """Generate a new model name if ``self._base_name`` is present."""
        if self._base_name:
            self.name = utils.name_from_base(self._base_name)

    def _framework(self):
        """Placeholder docstring"""
        return getattr(self, "_framework_name", None)

    def _get_framework_version(self):
        """Placeholder docstring"""
        return getattr(self, "framework_version", None)

    def _edge_packaging_job_config(
        self,
        output_path,
        role,
        model_name,
        model_version,
        packaging_job_name,
        compilation_job_name,
        resource_key,
        s3_kms_key,
        tags,
    ):
        """Creates a request object for a packaging job.

        Args:
            output_path (str): where in S3 to store the output of the job
            role (str): what role to use when executing the job
            packaging_job_name (str): what to name the packaging job
            compilation_job_name (str): what compilation job to source the model from
            resource_key (str): the kms key to encrypt the disk with
            s3_kms_key (str): the kms key to encrypt the output with
            tags (list[dict]): List of tags for labeling an edge packaging job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
        Returns:
            dict: the request object to use when creating a packaging job
        """
        output_model_config = {
            "S3OutputLocation": output_path,
        }
        if s3_kms_key is not None:
            output_model_config["KmsKeyId"] = s3_kms_key

        return {
            "output_model_config": output_model_config,
            "role": role,
            "tags": tags,
            "model_name": model_name,
            "model_version": model_version,
            "job_name": packaging_job_name,
            "compilation_job_name": compilation_job_name,
            "resource_key": resource_key,
        }

    def _compilation_job_config(
        self,
        target_instance_type,
        input_shape,
        output_path,
        role,
        compile_max_run,
        job_name,
        framework,
        tags,
        target_platform_os=None,
        target_platform_arch=None,
        target_platform_accelerator=None,
        compiler_options=None,
        framework_version=None,
    ):
        """Placeholder Docstring"""
        input_model_config = {
            "S3Uri": self.model_data,
            "DataInputConfig": json.dumps(input_shape)
            if isinstance(input_shape, dict)
            else input_shape,
            "Framework": framework.upper(),
        }

        def multi_version_compilation_supported(
            target_instance_type: str, framework: str, framework_version: str
        ):
            if target_instance_type and framework and framework_version:
                framework = framework.lower()

                multi_version_frameworks_support_mapping = {
                    "ml_inf1": ["pytorch", "tensorflow", "mxnet"],
                    "ml_inf2": ["pytorch", "tensorflow"],
                    "ml_trn1": ["pytorch", "tensorflow"],
                    "neo_ioc_targets": ["pytorch", "tensorflow"],
                    "neo_edge_targets": ["pytorch", "tensorflow"],
                }
                if target_instance_type in NEO_IOC_TARGET_DEVICES:
                    return framework in multi_version_frameworks_support_mapping["neo_ioc_targets"]
                if target_instance_type in ["ml_inf1", "ml_inf2", "ml_trn1"]:
                    return (
                        framework in multi_version_frameworks_support_mapping[target_instance_type]
                    )
                if target_instance_type not in NEO_MULTIVERSION_UNSUPPORTED:
                    return framework in multi_version_frameworks_support_mapping["neo_edge_targets"]
            return False

        if multi_version_compilation_supported(target_instance_type, framework, framework_version):
            input_model_config["FrameworkVersion"] = utils.get_short_version(framework_version)

        role = self.sagemaker_session.expand_role(role)
        output_model_config = {
            "S3OutputLocation": output_path,
        }

        if target_instance_type is not None:
            output_model_config["TargetDevice"] = target_instance_type
        else:
            if target_platform_os is None and target_platform_arch is None:
                raise ValueError(
                    "target_instance_type or (target_platform_os and target_platform_arch) "
                    "should be provided"
                )
            target_platform = {
                "Os": target_platform_os,
                "Arch": target_platform_arch,
            }
            if target_platform_accelerator is not None:
                target_platform["Accelerator"] = target_platform_accelerator
            output_model_config["TargetPlatform"] = target_platform

        if compiler_options is not None:
            output_model_config["CompilerOptions"] = (
                json.dumps(compiler_options)
                if isinstance(compiler_options, dict)
                else compiler_options
            )

        return {
            "input_model_config": input_model_config,
            "output_model_config": output_model_config,
            "role": role,
            "stop_condition": {"MaxRuntimeInSeconds": compile_max_run},
            "tags": tags,
            "job_name": job_name,
        }

    def package_for_edge(
        self,
        output_path,
        model_name,
        model_version,
        role=None,
        job_name=None,
        resource_key=None,
        s3_kms_key=None,
        tags=None,
    ):
        """Package this ``Model`` with SageMaker Edge.

        Creates a new EdgePackagingJob and wait for it to finish.
        model_data will now point to the packaged artifacts.

        Args:
            output_path (str): Specifies where to store the packaged model
            role (str): Execution role
            model_name (str): the name to attach to the model metadata
            model_version (str): the version to attach to the model metadata
            job_name (str): The name of the edge packaging job
            resource_key (str): the kms key to encrypt the disk with
            s3_kms_key (str): the kms key to encrypt the output with
            tags (list[dict]): List of tags for labeling an edge packaging job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """
        if self._compilation_job_name is None:
            raise ValueError("You must first compile this model")
        if job_name is None:
            job_name = f"packaging{self._compilation_job_name[11:]}"
        self._init_sagemaker_session_if_does_not_exist(None)
        s3_kms_key = resolve_value_from_config(
            s3_kms_key,
            EDGE_PACKAGING_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        role = resolve_value_from_config(
            role, EDGE_PACKAGING_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        resource_key = resolve_value_from_config(
            resource_key, EDGE_PACKAGING_RESOURCE_KEY_PATH, sagemaker_session=self.sagemaker_session
        )
        if role is not None:
            role = self.sagemaker_session.expand_role(role)
        config = self._edge_packaging_job_config(
            output_path,
            role,
            model_name,
            model_version,
            job_name,
            self._compilation_job_name,
            resource_key,
            s3_kms_key,
            tags,
        )
        self.sagemaker_session.package_model_for_edge(**config)
        job_status = self.sagemaker_session.wait_for_edge_packaging_job(job_name)
        self.model_data = job_status["ModelArtifact"]
        self._is_edge_packaged_model = True

        return self

    def compile(
        self,
        target_instance_family,
        input_shape,
        output_path,
        role=None,
        tags=None,
        job_name=None,
        compile_max_run=15 * 60,
        framework=None,
        framework_version=None,
        target_platform_os=None,
        target_platform_arch=None,
        target_platform_accelerator=None,
        compiler_options=None,
    ):
        """Compile this ``Model`` with SageMaker Neo.

        Args:
            target_instance_family (str): Identifies the device that you want to
                run your model after compilation, for example: ml_c5. For allowed
                strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                Alternatively, you can select an OS, Architecture and Accelerator using
                ``target_platform_os``, ``target_platform_arch``,
                and ``target_platform_accelerator``.
            input_shape (dict): Specifies the name and shape of the expected
                inputs for your trained model in json dictionary form, for
                example: {'data': [1,3,1024,1024]}, or {'var1': [1,1,28,28],
                'var2': [1,1,28,28]}
            output_path (str): Specifies where to store the compiled model
            role (str): Execution role
            tags (list[dict]): List of tags for labeling a compilation job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            job_name (str): The name of the compilation job
            compile_max_run (int): Timeout in seconds for compilation (default:
                15 * 60). After this amount of time Amazon SageMaker Neo
                terminates the compilation job regardless of its current status.
            framework (str): The framework that is used to train the original
                model. Allowed values: 'mxnet', 'tensorflow', 'keras', 'pytorch',
                'onnx', 'xgboost'
            framework_version (str): The version of framework, for example:
                '1.5' for PyTorch
            target_platform_os (str): Target Platform OS, for example: 'LINUX'.
                For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family by setting target_instance
                family to None.
            target_platform_arch (str): Target Platform Architecture, for example: 'X86_64'.
                For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family by setting target_instance
                family to None.
            target_platform_accelerator (str, optional): Target Platform Accelerator,
                for example: 'NVIDIA'. For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family by setting target_instance
                family to None.
            compiler_options (dict, optional): Additional parameters for compiler.
                Compiler Options are TargetPlatform / target_instance_family specific. See
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html for details.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """
        framework = framework or self._framework()
        if framework is None:
            raise ValueError(
                "You must specify framework, allowed values {}".format(NEO_ALLOWED_FRAMEWORKS)
            )
        if framework not in NEO_ALLOWED_FRAMEWORKS:
            raise ValueError(
                "You must provide valid framework, allowed values {}".format(NEO_ALLOWED_FRAMEWORKS)
            )
        if job_name is None:
            raise ValueError("You must provide a compilation job name")
        if self.model_data is None:
            raise ValueError("You must provide an S3 path to the compressed model artifacts.")
        if isinstance(self.model_data, dict):
            raise ValueError("Compiling model data from ModelDataSource is currently not supported")

        framework_version = framework_version or self._get_framework_version()

        self._init_sagemaker_session_if_does_not_exist(target_instance_family)
        role = resolve_value_from_config(
            role,
            COMPILATION_JOB_ROLE_ARN_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        if not role:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create a compilation job.")
        config = self._compilation_job_config(
            target_instance_family,
            input_shape,
            output_path,
            role,
            compile_max_run,
            job_name,
            framework,
            tags,
            target_platform_os,
            target_platform_arch,
            target_platform_accelerator,
            compiler_options,
            framework_version,
        )
        self.sagemaker_session.compile_model(**config)
        job_status = self.sagemaker_session.wait_for_compilation_job(job_name)
        self.model_data = job_status["ModelArtifacts"]["S3ModelArtifacts"]
        if target_instance_family is not None:
            if target_instance_family == "ml_eia2":
                pass
            elif target_instance_family.startswith("ml_"):
                self.image_uri = job_status.get("InferenceImage", None)
                self._is_compiled_model = True
            else:
                LOGGER.warning(
                    "The instance type %s is not supported for deployment via SageMaker."
                    "Please deploy the model manually.",
                    target_instance_family,
                )
        else:
            LOGGER.warning(
                "Devices described by Target Platform OS, Architecture and Accelerator are not"
                "supported for deployment via SageMaker. Please deploy the model manually."
            )

        self._compilation_job_name = job_name

        return self

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
                serverless inference or the model has not called ``right_size()``,
                then it need to be a number larger or equals
                to 1 (default: None)
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference or the model has not called ``right_size()``,
                then it is required to deploy a model.
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
                This can also be a recommendation id returned from ``DescribeModel`` contained in
                a list of ``RealtimeInferenceRecommendations`` within ``DeploymentRecommendation``
            explainer_config (sagemaker.explainer.ExplainerConfig): Specifies online explainability
                configuration for use with Amazon SageMaker Clarify. Default: None.
        Raises:
             ValueError: If arguments combination check failed in these circumstances:
                - If no role is specified or
                - If serverless inference config is not specified and instance type and instance
                    count are also not specified or
                - If a wrong type of object is provided as serverless inference config or async
                    inference config or
                - If inference recommendation id is specified along with incompatible parameters
        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of
                ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
                is not None. Otherwise, return None.
        """
        removed_kwargs("update_endpoint", kwargs)

        self._init_sagemaker_session_if_does_not_exist(instance_type)
        # Depending on the instance type, a local session (or) a session is initialized.
        self.role = resolve_value_from_config(
            self.role,
            MODEL_EXECUTION_ROLE_ARN_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.vpc_config = resolve_value_from_config(
            self.vpc_config,
            MODEL_VPC_CONFIG_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self._enable_network_isolation = resolve_value_from_config(
            self._enable_network_isolation,
            MODEL_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self.sagemaker_session,
        )

        tags = add_jumpstart_tags(
            tags=tags,
            inference_model_uri=self.model_data if isinstance(self.model_data, str) else None,
            inference_script_uri=self.source_dir,
        )

        if self.role is None:
            raise ValueError("Role can not be null for deploying a model")

        if (
            inference_recommendation_id is not None
            or self.inference_recommender_job_results is not None
        ):
            instance_type, initial_instance_count = self._update_params(
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                accelerator_type=accelerator_type,
                async_inference_config=async_inference_config,
                serverless_inference_config=serverless_inference_config,
                explainer_config=explainer_config,
                inference_recommendation_id=inference_recommendation_id,
                inference_recommender_job_results=self.inference_recommender_job_results,
            )

        is_async = async_inference_config is not None
        if is_async and not isinstance(async_inference_config, AsyncInferenceConfig):
            raise ValueError("async_inference_config needs to be a AsyncInferenceConfig object")

        is_explainer_enabled = explainer_config is not None
        if is_explainer_enabled and not isinstance(explainer_config, ExplainerConfig):
            raise ValueError("explainer_config needs to be a ExplainerConfig object")

        is_serverless = serverless_inference_config is not None
        if not is_serverless and not (instance_type and initial_instance_count):
            raise ValueError(
                "Must specify instance type and instance count unless using serverless inference"
            )

        if is_serverless and not isinstance(serverless_inference_config, ServerlessInferenceConfig):
            raise ValueError(
                "serverless_inference_config needs to be a ServerlessInferenceConfig object"
            )

        if instance_type and instance_type.startswith("ml.inf") and not self._is_compiled_model:
            LOGGER.warning(
                "Your model is not compiled. Please compile your model before using Inferentia."
            )

        compiled_model_suffix = None if is_serverless else "-".join(instance_type.split(".")[:-1])
        if self._is_compiled_model and not is_serverless:
            self._ensure_base_name_if_needed(
                image_uri=self.image_uri,
                script_uri=self.source_dir,
                model_uri=self.model_data,
            )
            if self._base_name is not None:
                self._base_name = "-".join((self._base_name, compiled_model_suffix))

        self._create_sagemaker_model(
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            tags=tags,
            serverless_inference_config=serverless_inference_config,
        )

        serverless_inference_config_dict = (
            serverless_inference_config._to_request_dict() if is_serverless else None
        )
        production_variant = sagemaker.production_variant(
            self.name,
            instance_type,
            initial_instance_count,
            accelerator_type=accelerator_type,
            serverless_inference_config=serverless_inference_config_dict,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
        )
        if endpoint_name:
            self.endpoint_name = endpoint_name
        else:
            base_endpoint_name = self._base_name or utils.base_from_name(self.name)
            if self._is_compiled_model and not is_serverless:
                if not base_endpoint_name.endswith(compiled_model_suffix):
                    base_endpoint_name = "-".join((base_endpoint_name, compiled_model_suffix))
            self.endpoint_name = utils.name_from_base(base_endpoint_name)

        data_capture_config_dict = None
        if data_capture_config is not None:
            data_capture_config_dict = data_capture_config._to_request_dict()

        async_inference_config_dict = None
        if is_async:
            if (
                async_inference_config.output_path is None
                or async_inference_config.failure_path is None
            ):
                async_inference_config = self._build_default_async_inference_config(
                    async_inference_config
                )
            async_inference_config.kms_key_id = resolve_value_from_config(
                async_inference_config.kms_key_id,
                ENDPOINT_CONFIG_ASYNC_KMS_KEY_ID_PATH,
                sagemaker_session=self.sagemaker_session,
            )
            async_inference_config_dict = async_inference_config._to_request_dict()

        explainer_config_dict = None
        if is_explainer_enabled:
            explainer_config_dict = explainer_config._to_request_dict()

        self.sagemaker_session.endpoint_from_production_variants(
            name=self.endpoint_name,
            production_variants=[production_variant],
            tags=tags,
            kms_key=kms_key,
            wait=wait,
            data_capture_config_dict=data_capture_config_dict,
            explainer_config_dict=explainer_config_dict,
            async_inference_config_dict=async_inference_config_dict,
        )

        if self.predictor_cls:
            predictor = self.predictor_cls(self.endpoint_name, self.sagemaker_session)
            if serializer:
                predictor.serializer = serializer
            if deserializer:
                predictor.deserializer = deserializer
            if is_async:
                return AsyncPredictor(predictor, self.name)
            return predictor
        return None

    def _build_default_async_inference_config(self, async_inference_config):
        """Build default async inference config and return ``AsyncInferenceConfig``"""
        unique_folder = unique_name_from_base(self.name)
        if async_inference_config.output_path is None:
            async_output_s3uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                "async-endpoint-outputs",
                unique_folder,
            )
            async_inference_config.output_path = async_output_s3uri

        if async_inference_config.failure_path is None:
            async_failure_s3uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                "async-endpoint-failures",
                unique_folder,
            )
            async_inference_config.failure_path = async_failure_s3uri

        return async_inference_config

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
        self._init_sagemaker_session_if_does_not_exist(instance_type)

        self._create_sagemaker_model(instance_type, tags=tags)
        if self.enable_network_isolation():
            env = None

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
            base_transform_job_name=self._base_name or self.name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=self.sagemaker_session,
        )

    def delete_model(self):
        """Delete an Amazon SageMaker Model.

        Raises:
            ValueError: if the model is not created yet.
        """
        if self.name is None:
            raise ValueError(
                "The SageMaker model must be created first before attempting to delete."
            )
        self.sagemaker_session.delete_model(self.name)


class FrameworkModel(Model):
    """A Model for working with an SageMaker ``Framework``.

    This class hosts user-defined code in S3 and sets code location and
    configuration in model environment variables.
    """

    def __init__(
        self,
        model_data: Union[str, PipelineVariable, dict],
        image_uri: Union[str, PipelineVariable],
        role: Optional[str] = None,
        entry_point: Optional[str] = None,
        source_dir: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        name: Optional[str] = None,
        container_log_level: Union[int, PipelineVariable] = logging.INFO,
        code_location: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Initialize a ``FrameworkModel``.

        Args:
            model_data (str or PipelineVariable or dict): The S3 location of
                SageMaker model data.
            image_uri (str or PipelineVariable): A Docker image URI.
            role (str): An IAM role name or ARN for SageMaker to access AWS
                resources on your behalf.
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to model
                hosting. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo.

                Example:
                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='src/inference.py'.
            source_dir (str): Path (absolute, relative or an S3 URI) to a directory
                with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory are preserved
                when training on Amazon SageMaker. If 'git_config' is provided,
                'source_dir' should be a relative location to a directory in the Git repo.
                If the directory points to S3, no code will be uploaded and the S3 location
                will be used instead.

                .. admonition:: Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='inference.py', source_dir='src'.
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name.
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables to
                run with ``image_uri`` when hosted in SageMaker (default: None).
            name (str): The model name. If None, a default model name will be
                selected on each ``deploy``.
            container_log_level (int or PipelineVariable): Log level to use within
                the container (default: logging.INFO). Valid values are defined
                in the Python logging module.
            code_location (str): Name of the S3 bucket where custom code is
                uploaded (default: None). If not specified, default bucket
                created by ``sagemaker.session.Session`` is used.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            dependencies (list[str]): A list of paths to directories (absolute
                or relative) with any additional libraries that will be exported
                to the container (default: []). The library folders will be
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo. If the ```source_dir``` points
                to S3, code will be uploaded and the S3 location will be used
                instead.

                .. admonition:: Example

                    The following call

                    >>> Model(entry_point='inference.py',
                    ...       dependencies=['my/libs/common', 'virtual-env'])

                    results in the following inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ inference.py
                    >>>     |------ common
                    >>>     |------ virtual-env

                This is not supported with "local code" in Local Mode.
            git_config (dict[str, str]): Git configurations used for cloning
                files, including ``repo``, ``branch``, ``commit``,
                ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                ``repo`` field is required. All other fields are optional.
                ``repo`` specifies the Git repository where your training script
                is stored. If you don't provide ``branch``, the default value
                'master' is used. If you don't provide ``commit``, the latest
                commit in the specified branch is used.

                .. admonition:: Example

                    The following config:

                    >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                    >>>               'branch': 'test-branch-git-config',
                    >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

                    results in cloning the repo specified in 'repo', then
                    checkout the 'master' branch, and checkout the specified
                    commit.

                ``2FA_enabled``, ``username``, ``password`` and ``token`` are
                used for authentication. For GitHub (or other Git) accounts, set
                ``2FA_enabled`` to 'True' if two-factor authentication is
                enabled for the account, otherwise set it to 'False'. If you do
                not provide a value for ``2FA_enabled``, a default value of
                'False' is used. CodeCommit does not support two-factor
                authentication, so do not provide "2FA_enabled" with CodeCommit
                repositories.

                For GitHub and other Git repos, when SSH URLs are provided, it
                doesn't matter whether 2FA is enabled or disabled; you should
                either have no passphrase for the SSH key pairs, or have the
                ssh-agent configured so that you will not be prompted for SSH
                passphrase when you do 'git clone' command with SSH URLs. When
                HTTPS URLs are provided: if 2FA is disabled, then either token
                or username+password will be used for authentication if provided
                (token prioritized); if 2FA is enabled, only token will be used
                for authentication if provided. If required authentication info
                is not provided, python SDK will try to use local credentials
                storage to authenticate. If that fails either, an error message
                will be thrown.

                For CodeCommit repos, 2FA is not supported, so '2FA_enabled'
                should not be provided. There is no token in CodeCommit, so
                'token' should not be provided too. When 'repo' is an SSH URL,
                the requirements are the same as GitHub-like repos. When 'repo'
                is an HTTPS URL, username+password will be used for
                authentication if they are provided; otherwise, python SDK will
                try to use either CodeCommit credential helper or local
                credential storage for authentication.
            **kwargs: Keyword arguments passed to the superclass
                :class:`~sagemaker.model.Model`.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.Model`.
        """
        super(FrameworkModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=predictor_cls,
            env=env,
            name=name,
            sagemaker_session=sagemaker_session,
            source_dir=source_dir,
            code_location=code_location,
            entry_point=entry_point,
            container_log_level=container_log_level,
            dependencies=dependencies,
            git_config=git_config,
            **kwargs,
        )


# works for MODEL_PACKAGE_ARN with or without version info.
MODEL_PACKAGE_ARN_PATTERN = r"arn:aws:sagemaker:(.*?):(.*?):model-package/(.*?)(?:/(\d+))?$"


class ModelPackage(Model):
    """A SageMaker ``Model`` that can be deployed to an ``Endpoint``."""

    def __init__(
        self,
        role=None,
        model_data=None,
        algorithm_arn=None,
        model_package_arn=None,
        **kwargs,
    ):
        """Initialize a SageMaker ModelPackage.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file. Must be provided if algorithm_arn is provided.
            algorithm_arn (str): algorithm arn used to train the model, can be
                just the name if your account owns the algorithm. Must also
                provide ``model_data``.
            model_package_arn (str): An existing SageMaker Model Package arn,
                can be just the name if your account owns the Model Package.
                ``model_data`` is not required.
            **kwargs: Additional kwargs passed to the Model constructor.
        """
        if isinstance(model_data, dict):
            raise ValueError(
                "Creating ModelPackage with ModelDataSource is currently not supported"
            )

        super(ModelPackage, self).__init__(
            role=role, model_data=model_data, image_uri=None, **kwargs
        )

        if model_package_arn and algorithm_arn:
            raise ValueError(
                "model_package_arn and algorithm_arn are mutually exclusive."
                "Both were provided: model_package_arn: %s algorithm_arn: %s"
                % (model_package_arn, algorithm_arn)
            )

        if model_package_arn is None and algorithm_arn is None:
            raise ValueError(
                "either model_package_arn or algorithm_arn is required." " None was provided."
            )

        self.algorithm_arn = algorithm_arn
        if self.algorithm_arn is not None:
            if model_data is None:
                raise ValueError("model_data must be provided with algorithm_arn")
            self.model_data = model_data

        self.model_package_arn = model_package_arn
        self._created_model_package_name = None

    def _create_sagemaker_model_package(self):
        """Placeholder docstring"""
        if self.algorithm_arn is None:
            raise ValueError("No algorithm_arn was provided to create a SageMaker Model Pacakge")

        name = self.name or utils.name_from_base(self.algorithm_arn.split("/")[-1])
        description = "Model Package created from training with %s" % self.algorithm_arn
        self.sagemaker_session.create_model_package_from_algorithm(
            name, description, self.algorithm_arn, self.model_data
        )
        return name

    def enable_network_isolation(self):
        """Whether to enable network isolation when creating a model out of this ModelPackage

        Returns:
            bool: If network isolation should be enabled or not.
        """
        return self._is_marketplace()

    def _is_marketplace(self):
        """Placeholder docstring"""
        model_package_name = self.model_package_arn or self._created_model_package_name
        if model_package_name is None:
            return True

        # Models can lazy-init sagemaker_session until deploy() is called to support
        # LocalMode so we must make sure we have an actual session to describe the model package.
        sagemaker_session = self.sagemaker_session or sagemaker.Session()

        model_package_desc = sagemaker_session.sagemaker_client.describe_model_package(
            ModelPackageName=model_package_name
        )
        for container in model_package_desc["InferenceSpecification"]["Containers"]:
            if "ProductId" in container:
                return True
        return False

    def _create_sagemaker_model(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Create a SageMaker Model Entity

        Args:
            args: Positional arguments coming from the caller. This class does not require
                any so they are ignored.

            kwargs: Keyword arguments coming from the caller. This class does not require
                any so they are ignored.
        """
        if self.algorithm_arn:
            # When ModelPackage is created using an algorithm_arn we need to first
            # create a ModelPackage. If we had already created one then its fine to re-use it.
            if self._created_model_package_name is None:
                model_package_name = self._create_sagemaker_model_package()
                self.sagemaker_session.wait_for_model_package(model_package_name)
                self._created_model_package_name = model_package_name
            model_package_name = self._created_model_package_name
            container_def = {"ModelPackageName": model_package_name}
        else:
            # When a ModelPackageArn is provided we just create the Model
            match = re.match(MODEL_PACKAGE_ARN_PATTERN, self.model_package_arn)
            if match:
                model_package_name = match.group(3)
            else:
                # model_package_arn can be just the name if your account owns the Model Package
                model_package_name = self.model_package_arn
            container_def = {"ModelPackageName": self.model_package_arn}

        if self.env != {}:
            container_def["Environment"] = self.env

        self._ensure_base_name_if_needed(model_package_name)
        self._set_model_name_if_needed()

        self.sagemaker_session.create_model(
            self.name,
            self.role,
            container_def,
            vpc_config=self.vpc_config,
            enable_network_isolation=self.enable_network_isolation(),
            tags=kwargs.get("tags"),
        )

    def _ensure_base_name_if_needed(self, base_name):
        """Set the base name if there is no model name provided."""
        if self.name is None:
            self._base_name = base_name
