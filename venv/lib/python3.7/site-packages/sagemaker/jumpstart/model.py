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
"""This module stores JumpStart implementation of Model class."""

from __future__ import absolute_import
import re

from typing import Dict, List, Optional, Union
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.jumpstart.accessors import JumpStartModelsAccessor
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.exceptions import INVALID_MODEL_ID_ERROR_MSG
from sagemaker.jumpstart.factory.model import (
    get_default_predictor,
    get_deploy_kwargs,
    get_init_kwargs,
)
from sagemaker.jumpstart.utils import is_valid_model_id
from sagemaker.utils import stringify_object
from sagemaker.model import MODEL_PACKAGE_ARN_PATTERN, Model
from sagemaker.model_monitor.data_capture_config import DataCaptureConfig
from sagemaker.predictor import PredictorBase
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.session import Session
from sagemaker.workflow.entities import PipelineVariable


class JumpStartModel(Model):
    """JumpStartModel class.

    This class sets defaults based on the model ID and version.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        region: Optional[str] = None,
        instance_type: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        model_data: Optional[Union[str, PipelineVariable]] = None,
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
        container_log_level: Optional[Union[int, PipelineVariable]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        model_package_arn: Optional[str] = None,
    ):
        """Initializes a ``JumpStartModel``.

        This method sets model-specific defaults for the ``Model.__init__`` method.

        Only model ID is required to instantiate this class, however any field can be overriden.

        Any field set to ``None`` does not get passed to the parent class method.

        Args:
            model_id (Optional[str]): JumpStart model ID to use. See
                https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html
                for list of model IDs.
            model_version (Optional[str]): Version for JumpStart model to use (Default: None).
            tolerate_vulnerable_model (Optional[bool]): True if vulnerable versions of model
                specifications should be tolerated (exception not raised). If False, raises an
                exception if the script used by this version of the model has dependencies with
                known security vulnerabilities. (Default: None).
            tolerate_deprecated_model (Optional[bool]): True if deprecated models should be
                tolerated (exception not raised). False if these models should raise an exception.
                (Default: None).
            region (Optional[str]): The AWS region in which to launch the model. (Default: None).
            instance_type (Optional[str]): The EC2 instance type to use when provisioning a hosting
                endpoint. (Default: None).
            image_uri (Optional[Union[str, PipelineVariable]]): A Docker image URI. (Default: None).
            model_data (Optional[Union[str, PipelineVariable]]): The S3 location of a SageMaker
                model data ``.tar.gz`` file. (Default: None).
            role (Optional[str]): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role if it needs to access some AWS resources.
                It can be null if this is being used to create a Model to pass
                to a ``PipelineModel`` which has its own Role field. (Default:
                None).
            predictor_cls (Optional[callable[string, sagemaker.session.Session]]): A
                function to call to create a predictor (Default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name. (Default: None).
            env (Optional[dict[str, str] or dict[str, PipelineVariable]]): Environment variables
                to run with ``image_uri`` when hosted in SageMaker. (Default: None).
            name (Optional[str]): The model name. If None, a default model name will be
                selected on each ``deploy``. (Default: None).
            vpc_config (Optional[Union[dict[str, list[str]],dict[str, list[PipelineVariable]]]]):
                The VpcConfig set on the model (Default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids. (Default: None).
            sagemaker_session (Optional[sagemaker.session.Session]): A SageMaker Session
                object, used for SageMaker interactions (Default: None). If not
                specified, one is created using the default AWS configuration
                chain. (Default: None).
            enable_network_isolation (Optional[Union[bool, PipelineVariable]]): If True,
                enables network isolation in the endpoint, isolating the model
                container. No inbound or outbound network calls can be made to
                or from the model container. (Default: None).
            model_kms_key (Optional[str]): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked. (Default: None).
            image_config (Optional[Union[dict[str, str], dict[str, PipelineVariable]]]): Specifies
                whether the image of model container is pulled from ECR, or private
                registry in your VPC. By default it is set to pull model container
                image from ECR. (Default: None).
            source_dir (Optional[str]): The absolute, relative, or S3 URI Path to a directory
                with any other training source code dependencies aside from the entry
                point file (Default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory is preserved
                when training on Amazon SageMaker. If 'git_config' is provided,
                'source_dir' should be a relative location to a directory in the Git repo.
                If the directory points to S3, no code is uploaded and the S3 location
                is used instead. (Default: None).

                .. admonition:: Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='inference.py', source_dir='src'.
            code_location (Optional[str]): Name of the S3 bucket where custom code is
                uploaded (Default: None). If not specified, the default bucket
                created by ``sagemaker.session.Session`` is used. (Default: None).
            entry_point (Optional[str]): The absolute or relative path to the local Python
                source file that should be executed as the entry point to
                model hosting. (Default: None). If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo. (Default: None).

                .. admonition:: Example
                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='src/inference.py'.
            container_log_level (Optional[Union[int, PipelineVariable]]): Log level to use
                within the container. Valid values are defined in the Python
                logging module. (Default: None).
            dependencies (Optional[list[str]]): A list of absolute or relative paths to directories
                with any additional libraries that should be exported
                to the container (default: []). The library folders are
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo. If the ```source_dir``` points
                to S3, code will be uploaded and the S3 location will be used
                instead. This is not supported with "local code" in Local Mode.
                (Default: None).

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
            git_config (Optional[dict[str, str]]): Git configurations used for cloning
                files, including ``repo``, ``branch``, ``commit``,
                ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                ``repo`` field is required. All other fields are optional.
                ``repo`` specifies the Git repository where your training script
                is stored. If you don't provide ``branch``, the default value
                'master' is used. If you don't provide ``commit``, the latest
                commit in the specified branch is used.

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
                (Default: None).

                .. admonition:: Example

                    The following config results in cloning the repo specified in 'repo', then
                    checking out the 'master' branch, and checking out the specified
                    commit.

                    >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                    >>>               'branch': 'test-branch-git-config',
                    >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

            model_package_arn (Optional[str]): An existing SageMaker Model Package arn,
                can be just the name if your account owns the Model Package.
                ``model_data`` is not required. (Default: None).
        Raises:
            ValueError: If the model ID is not recognized by JumpStart.
        """

        def _is_valid_model_id_hook():
            return is_valid_model_id(
                model_id=model_id,
                model_version=model_version,
                region=region,
                script=JumpStartScriptScope.INFERENCE,
                sagemaker_session=sagemaker_session,
            )

        if not _is_valid_model_id_hook():
            JumpStartModelsAccessor.reset_cache()
            if not _is_valid_model_id_hook():
                raise ValueError(INVALID_MODEL_ID_ERROR_MSG.format(model_id=model_id))

        self._model_data_is_set = model_data is not None

        model_init_kwargs = get_init_kwargs(
            model_id=model_id,
            model_from_estimator=False,
            model_version=model_version,
            instance_type=instance_type,
            tolerate_vulnerable_model=tolerate_vulnerable_model,
            tolerate_deprecated_model=tolerate_deprecated_model,
            region=region,
            image_uri=image_uri,
            model_data=model_data,
            source_dir=source_dir,
            entry_point=entry_point,
            env=env,
            predictor_cls=predictor_cls,
            role=role,
            name=name,
            vpc_config=vpc_config,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=enable_network_isolation,
            model_kms_key=model_kms_key,
            image_config=image_config,
            code_location=code_location,
            container_log_level=container_log_level,
            dependencies=dependencies,
            git_config=git_config,
            model_package_arn=model_package_arn,
        )

        self.orig_predictor_cls = predictor_cls

        self.model_id = model_init_kwargs.model_id
        self.model_version = model_init_kwargs.model_version
        self.instance_type = model_init_kwargs.instance_type
        self.tolerate_vulnerable_model = model_init_kwargs.tolerate_vulnerable_model
        self.tolerate_deprecated_model = model_init_kwargs.tolerate_deprecated_model
        self.region = model_init_kwargs.region
        self.model_package_arn = model_init_kwargs.model_package_arn
        self.sagemaker_session = model_init_kwargs.sagemaker_session

        super(JumpStartModel, self).__init__(**model_init_kwargs.to_kwargs_dict())

    def _create_sagemaker_model(
        self,
        instance_type=None,
        accelerator_type=None,
        tags=None,
        serverless_inference_config=None,
        **kwargs,
    ):
        """Create a SageMaker Model Entity

        Args:
            instance_type (str): Optional. The EC2 instance type that this Model will be
                used for, this is only used to determine if the image needs GPU
                support or not. (Default: None).
            accelerator_type (str): Optional. Type of Elastic Inference accelerator to
                attach to an endpoint for model loading and inference, for
                example, 'ml.eia1.medium'. If not specified, no Elastic
                Inference accelerator will be attached to the endpoint. (Default: None).
            tags (List[dict[str, str]]): Optional. The list of tags to add to
                the model. Example: >>> tags = [{'Key': 'tagname', 'Value':
                'tagvalue'}] For more information about tags, see
                https://boto3.amazonaws.com/v1/documentation
                /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
                (Default: None).
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Optional. Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.
                (Default: None).
            kwargs: Keyword arguments coming from the caller. This class does not require
                any so they are ignored.
        """

        # if the user inputs a model artifact uri, do not use model package arn to create
        # inference endpoint.
        if self.model_package_arn and not self._model_data_is_set:
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

            if self.name is None:
                self._base_name = model_package_name

            self._set_model_name_if_needed()

            self.sagemaker_session.create_model(
                self.name,
                self.role,
                container_def,
                vpc_config=self.vpc_config,
                enable_network_isolation=self.enable_network_isolation(),
                tags=tags,
            )
        else:
            super(JumpStartModel, self)._create_sagemaker_model(
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                tags=tags,
                serverless_inference_config=serverless_inference_config,
                **kwargs,
            )

    def deploy(
        self,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        serializer: Optional[BaseSerializer] = None,
        deserializer: Optional[BaseDeserializer] = None,
        accelerator_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        tags: List[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
        wait: Optional[bool] = True,
        data_capture_config: Optional[DataCaptureConfig] = None,
        async_inference_config: Optional[AsyncInferenceConfig] = None,
        serverless_inference_config: Optional[ServerlessInferenceConfig] = None,
        volume_size: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        inference_recommendation_id: Optional[str] = None,
        explainer_config: Optional[ExplainerConfig] = None,
    ) -> PredictorBase:
        """Creates endpoint by calling base ``Model`` class `deploy` method.

        Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an
        ``Endpoint`` from this ``Model``.

        Any field set to ``None`` does not get passed to the parent class method.


        Args:
            initial_instance_count (Optional[int]): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. If not using
                serverless inference or the model has not called ``right_size()``,
                then it need to be a number larger or equals
                to 1. (Default: None)
            instance_type (Optional[str]): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference or the model has not called ``right_size()``,
                then it is required to deploy a model.
                (Default: None)
            serializer (Optional[:class:`~sagemaker.serializers.BaseSerializer`]): A
                serializer object, used to encode data for an inference endpoint
                (Default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``. (Default: None).
            deserializer (Optional[:class:`~sagemaker.deserializers.BaseDeserializer`]): A
                deserializer object, used to decode data from an inference
                endpoint (Default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``. (Default: None).
            accelerator_type (Optional[str]): Type of Elastic Inference accelerator to
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
                (Default: None).
            endpoint_name (Optional[str]): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
                (Default: None).
            tags (Optional[List[dict[str, str]]]): The list of tags to attach to this
                specific endpoint. (Default: None).
            kms_key (Optional[str]): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint. (Default: None).
            wait (Optional[bool]): Whether the call should wait until the deployment of
                this model completes. (Default: True).
            data_capture_config (Optional[sagemaker.model_monitor.DataCaptureConfig]): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. (Default: None).
            async_inference_config (Optional[sagemaker.model_monitor.AsyncInferenceConfig]):
                Specifies configuration related to async endpoint. Use this configuration when
                trying to create async endpoint and make async inference. If empty config object
                passed through, will use default config to deploy async endpoint. Deploy a
                real-time endpoint if it's None. (Default: None)
            serverless_inference_config (Optional[sagemaker.serverless.ServerlessInferenceConfig]):
                Specifies configuration related to serverless endpoint. Use this configuration
                when trying to create serverless endpoint and make serverless inference. If
                empty object passed through, will use pre-defined values in
                ``ServerlessInferenceConfig`` class to deploy serverless endpoint. Deploy an
                instance based endpoint if it's None. (Default: None)
            volume_size (Optional[int]): The size, in GB, of the ML storage volume attached to
                individual inference instance associated with the production variant. Currenly only
                Amazon EBS gp2 storage volumes are supported. (Default: None).
            model_data_download_timeout (Optional[int]): The timeout value, in seconds, to download
                and extract model data from Amazon S3 to the individual inference instance
                associated with this production variant. (Default: None).
            container_startup_health_check_timeout (Optional[int]): The timeout value, in seconds,
                for your inference container to pass health check by SageMaker Hosting. For more
                information about health check see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-algo-ping-requests
                (Default: None).
            inference_recommendation_id (Optional[str]): The recommendation id which specifies the
                recommendation you picked from inference recommendation job results and
                would like to deploy the model and endpoint with recommended parameters.
                (Default: None).
            explainer_config (Optional[sagemaker.explainer.ExplainerConfig]): Specifies online
                explainability configuration for use with Amazon SageMaker Clarify. (Default: None).

        """

        deploy_kwargs = get_deploy_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            initial_instance_count=initial_instance_count,
            instance_type=instance_type or self.instance_type,
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
        )

        predictor = super(JumpStartModel, self).deploy(**deploy_kwargs.to_kwargs_dict())

        # If no predictor class was passed, add defaults to predictor
        if self.orig_predictor_cls is None and async_inference_config is None:
            return get_default_predictor(
                predictor=predictor,
                model_id=self.model_id,
                model_version=self.model_version,
                region=self.region,
                tolerate_deprecated_model=self.tolerate_deprecated_model,
                tolerate_vulnerable_model=self.tolerate_vulnerable_model,
                sagemaker_session=self.sagemaker_session,
            )

        # If a predictor class was passed, do not mutate predictor
        return predictor

    def __str__(self) -> str:
        """Overriding str(*) method to make more human-readable."""
        return stringify_object(self)
