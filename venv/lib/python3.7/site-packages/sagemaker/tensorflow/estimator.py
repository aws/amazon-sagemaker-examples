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
"""An estimator class for training with TensorFlow on Amazon SageMaker."""
from __future__ import absolute_import

import logging
from typing import Optional, Union, Dict

from packaging import version

from sagemaker import image_uris, s3, utils
from sagemaker.deprecations import renamed_kwargs
from sagemaker.estimator import Framework, EstimatorBase
import sagemaker.fw_utils as fw
from sagemaker.tensorflow import defaults
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.transformer import Transformer
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow import is_pipeline_variable
from sagemaker.tensorflow.training_compiler.config import TrainingCompilerConfig
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger("sagemaker")


class TensorFlow(Framework):
    """Handle end-to-end training and deployment of user-provided TensorFlow code."""

    _framework_name = "tensorflow"

    _HIGHEST_LEGACY_MODE_ONLY_VERSION = version.Version("1.10.0")
    _HIGHEST_PYTHON_2_VERSION = version.Version("2.1.1")

    def __init__(
        self,
        py_version: Optional[str] = None,
        framework_version: Optional[str] = None,
        model_dir: Optional[Union[str, PipelineVariable]] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        distribution: Optional[Dict[str, str]] = None,
        compiler_config: Optional[TrainingCompilerConfig] = None,
        **kwargs,
    ):
        """Initialize a ``TensorFlow`` estimator.

        Args:
            py_version (str): Python version you want to use for executing your model training
                code. Defaults to ``None``. Required unless ``image_uri`` is provided.
            framework_version (str): TensorFlow version you want to use for executing your model
                training code. Defaults to ``None``. Required unless ``image_uri`` is provided.
                List of supported versions:
                https://github.com/aws/sagemaker-python-sdk#tensorflow-sagemaker-estimators.
            model_dir (str or PipelineVariable): S3 location where the checkpoint data and models
                can be exported to during training (default: None). It will be passed in the
                training script as one of the command line arguments. If not specified,
                one is provided based on your training configuration:

                * *distributed training with SMDistributed or MPI with Horovod* - ``/opt/ml/model``
                * *single-machine training or distributed training without MPI* - \
                    ``s3://{output_path}/model``
                * *Local Mode with local sources (file:// instead of s3://)* - \
                    ``/opt/ml/shared/model``

                To disable having ``model_dir`` passed to your training script,
                set ``model_dir=False``.
            image_uri (str or PipelineVariable): If specified, the estimator will use this image
                for training and hosting, instead of selecting the appropriate SageMaker official
                image based on framework_version and py_version.
                It can be an ECR url or dockerhub image and tag.

                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.

                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If also ``None``, then a ``ValueError``
                will be raised.
            distribution (dict): A dictionary with information on how to run distributed training
                (default: None). Currently, the following are supported:
                distributed training with parameter servers, SageMaker Distributed (SMD) Data
                and Model Parallelism, and MPI. SMD Model Parallelism can only be used with MPI.

                **To enable the SageMaker distributed data parallelism:**

                    .. code:: python

                        { "smdistributed": { "dataparallel": { "enabled": True } } }

                    .. seealso::

                        To learn more, see :ref:`sdp_api_docs_toc`.

                **To enable the SageMaker distributed model parallelism:**

                    .. code:: python

                        {
                            "smdistributed": {
                                "modelparallel": {
                                    "enabled":True,
                                    "parameters": {
                                        "partitions": 2,
                                        "microbatches": 4,
                                        "placement_strategy": "spread",
                                        "pipeline": "interleaved",
                                        "optimize": "speed",
                                        "ddp": True,
                                    }
                            },
                            "mpi": {
                                "enabled" : True,
                                "processes_per_host" : 8,
                            }
                        }

                    .. note::

                        The SageMaker distributed model parallel library internally uses MPI.
                        In order to use model parallelism, MPI also must be enabled.

                    .. seealso::

                        To learn more, see :ref:`smp_api_docs_toc`.

                    .. seealso::

                        To find a complete list of parameters for SageMaker model parallelism,
                        see :ref:`sm-sdk-modelparallel-general`.

                **To enable Multi Worker Mirrored Strategy:**

                    .. code:: python

                        {
                            "multi_worker_mirrored_strategy": {
                                "enabled": True
                            }
                        }

                    This distribution strategy option is available for TensorFlow 2.9 and later in
                    the SageMaker Python SDK v2.xx.yy and later.
                    To learn more about the mirrored strategy for TensorFlow,
                    see `TensorFlow Distributed Training
                    <https://www.tensorflow.org/guide/distributed_training>`_
                    in the *TensorFlow documentation*.

                **To enable MPI:**

                    .. code:: python

                        {
                            "mpi": {
                                "enabled": True
                            }
                        }

                    To learn more, see `Training with Horovod
                    <https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-horovod>`_.

                **To enable parameter server:**

                    .. code:: python

                        {
                            "parameter_server": {
                                "enabled": True
                            }
                        }

                    To learn more, see `Training with parameter servers
                    <https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-parameter-servers>`_.
            compiler_config (:class:`~sagemaker.tensorflow.TrainingCompilerConfig`):
                Configures SageMaker Training Compiler to accelerate training.

            **kwargs: Additional kwargs passed to the Framework constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        distribution = renamed_kwargs("distributions", "distribution", distribution, kwargs)
        instance_type = renamed_kwargs(
            "train_instance_type", "instance_type", kwargs.get("instance_type"), kwargs
        )
        fw.validate_version_or_image_args(framework_version, py_version, image_uri)
        if py_version == "py2":
            logger.warning(
                fw.python_deprecation_warning(self._framework_name, defaults.LATEST_PY2_VERSION)
            )
        self.framework_version = framework_version
        self.py_version = py_version
        self.instance_type = instance_type

        if "enable_sagemaker_metrics" not in kwargs:
            # enable sagemaker metrics for TF v1.15 or greater:
            if framework_version and version.Version(framework_version) >= version.Version("1.15"):
                kwargs["enable_sagemaker_metrics"] = True

        super(TensorFlow, self).__init__(image_uri=image_uri, **kwargs)
        if distribution is not None:
            distribution = fw.validate_distribution(
                distribution,
                self.instance_groups,
                self._framework_name,
                framework_version,
                py_version,
                image_uri,
                kwargs,
            )
        self.model_dir = model_dir
        self.distribution = distribution or {}

        self._validate_args(py_version=py_version)
        if compiler_config is not None:
            if not isinstance(compiler_config, TrainingCompilerConfig):
                error_string = (
                    f"Expected instance of type {TrainingCompilerConfig}"
                    f"for argument compiler_config. "
                    f"Instead got {type(compiler_config)}"
                )
                raise ValueError(error_string)
            if compiler_config:
                compiler_config.validate(self)
        self.compiler_config = compiler_config

        if "profiler_config" in kwargs:
            fw.profiler_config_deprecation_warning(
                kwargs["profiler_config"], image_uri, self._framework_name, framework_version
            )

    def _validate_args(self, py_version):
        """Placeholder docstring"""

        if py_version == "py2" and self._only_python_3_supported():
            msg = (
                "Python 2 containers are only available with {} and lower versions. "
                "Please use a Python 3 container.".format(defaults.LATEST_PY2_VERSION)
            )
            raise AttributeError(msg)

        if self.image_uri is None and self._only_legacy_mode_supported():
            legacy_image_uri = image_uris.retrieve(
                "tensorflow",
                self.sagemaker_session.boto_region_name,
                instance_type=self.instance_type,
                version=self.framework_version,
                py_version=self.py_version,
                image_scope="training",
            )

            msg = (
                "TF {} supports only legacy mode. Please supply the image URI directly with "
                "'image_uri={}' and set 'model_dir=False'. If you are using any legacy parameters "
                "(training_steps, evaluation_steps, checkpoint_path, requirements_file), "
                "make sure to pass them directly as hyperparameters instead. For more, see "
                "https://sagemaker.readthedocs.io/en/v2.0.0.rc0/frameworks/tensorflow/upgrade_from_legacy.html."
            ).format(self.framework_version, legacy_image_uri)

            raise ValueError(msg)

    def _only_legacy_mode_supported(self):
        """Placeholder docstring"""
        return version.Version(self.framework_version) <= self._HIGHEST_LEGACY_MODE_ONLY_VERSION

    def _only_python_3_supported(self):
        """Placeholder docstring"""
        if not self.framework_version:
            return False
        return version.Version(self.framework_version) > self._HIGHEST_PYTHON_2_VERSION

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(TensorFlow, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        image_uri = init_params.pop("image_uri")
        framework, py_version, tag, script_mode = fw.framework_name_from_image(image_uri)

        if not framework:
            # If we were unable to parse the framework name from the image, it is not one of our
            # officially supported images, so just add the image to the init params.
            init_params["image_uri"] = image_uri
            return init_params

        model_dir = init_params["hyperparameters"].pop("model_dir", None)
        if model_dir:
            init_params["model_dir"] = model_dir
        elif script_mode is None:
            init_params["model_dir"] = False

        init_params["py_version"] = py_version

        # We switched image tagging scheme from regular image version (e.g. '1.0') to more
        # expressive containing framework version, device type and python version
        # (e.g. '1.5-gpu-py2'). For backward compatibility map deprecated image tag '1.0' to a
        # '1.4' framework version otherwise extract framework version from the tag itself.
        init_params["framework_version"] = (
            "1.4" if tag == "1.0" else fw.framework_version_from_tag(tag)
        )

        # Legacy images are required to be passed in explicitly.
        if not script_mode:
            init_params["image_uri"] = image_uri

        if framework != cls._framework_name:
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    job_details["TrainingJobName"]
                )
            )

        return init_params

    def create_model(
        self,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        **kwargs,
    ):
        """Creates ``TensorFlowModel`` object to be used for creating SageMaker model entities.

        This can be done by deploying it to a SageMaker endpoint,
        or starting SageMaker Batch Transform jobs.

        Args:
            role (str): The ``TensorFlowModel``, which is also used during transform jobs.
                If not specified, the role from the Estimator is used.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the
                model. Default: use subnets and security groups from this Estimator.

                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

            entry_point (str): Path (absolute or relative) to the local Python source file which
                should be executed as the entry point to training. If ``source_dir`` is specified,
                then ``entry_point`` must point to a file located at the root of ``source_dir``.
                If not specified and ``endpoint_type`` is 'tensorflow-serving',
                no entry point is used. If ``endpoint_type`` is also ``None``,
                then the training entry point is used.
            source_dir (str): Path (absolute or relative or an S3 URI) to a directory with any other
                serving source code dependencies aside from the entry point file (default: None).
            dependencies (list[str]): A list of paths to directories (absolute or relative) with
                any additional libraries that will be exported to the container (default: None).
            **kwargs: Additional kwargs passed to
                :class:`~sagemaker.tensorflow.model.TensorFlowModel`.

        Returns:
            sagemaker.tensorflow.model.TensorFlowModel: A ``TensorFlowModel`` object.
                See :class:`~sagemaker.tensorflow.model.TensorFlowModel` for full details.
        """
        kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        if "image_uri" not in kwargs:
            kwargs["image_uri"] = self.image_uri

        if "enable_network_isolation" not in kwargs:
            kwargs["enable_network_isolation"] = self.enable_network_isolation()

        return TensorFlowModel(
            model_data=self.model_data,
            role=role or self.role,
            container_log_level=self.container_log_level,
            framework_version=self.framework_version,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            entry_point=entry_point,
            source_dir=source_dir,
            dependencies=dependencies,
            **kwargs,
        )

    def hyperparameters(self):
        """Return hyperparameters used by your custom TensorFlow code during model training."""
        hyperparameters = super(TensorFlow, self).hyperparameters()
        additional_hyperparameters = self._distribution_configuration(self.distribution)

        if self.model_dir is not False:
            self.model_dir = self.model_dir or self._default_s3_path(
                "model", mpi=additional_hyperparameters.get(self.LAUNCH_MPI_ENV_NAME, False)
            )
            additional_hyperparameters["model_dir"] = self.model_dir

        hyperparameters.update(
            EstimatorBase._json_encode_hyperparameters(additional_hyperparameters)
        )

        if self.compiler_config:
            training_compiler_hyperparameters = self.compiler_config._to_hyperparameter_dict()
            hyperparameters.update(
                EstimatorBase._json_encode_hyperparameters(training_compiler_hyperparameters)
            )

        return hyperparameters

    def _default_s3_path(self, directory, mpi=False):
        """Placeholder docstring"""
        local_code = utils.get_config_value("local.local_code", self.sagemaker_session.config)
        if self.sagemaker_session.local_mode and local_code:
            return "/opt/ml/shared/{}".format(directory)
        if mpi:
            return "/opt/ml/model"
        if self._current_job_name:
            if is_pipeline_variable(self.output_path):
                return s3.s3_path_join(
                    "s3://",
                    self.sagemaker_session.default_bucket(),
                    self.sagemaker_session.default_bucket_prefix,
                    self._current_job_name,
                    directory,
                )
            return s3.s3_path_join(self.output_path, self._current_job_name, directory)
        return None

    def _validate_and_set_debugger_configs(self):
        """Disable Debugger Hook Config for ParameterServer (PS) as it is not supported in smdebug.

        Else, set default HookConfig
        """
        super(TensorFlow, self)._validate_and_set_debugger_configs()
        ps_enabled = "parameter_server" in self.distribution and self.distribution[
            "parameter_server"
        ].get("enabled", False)
        if ps_enabled:
            if self.debugger_hook_config is not None or self.debugger_rule_configs is not None:
                logger.info(
                    "Amazon SageMaker Debugger does not currently support "
                    "Parameter Server distribution"
                )
            self.debugger_hook_config = None
            self.debugger_rule_configs = None

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
        role=None,
        volume_kms_key=None,
        entry_point=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
        enable_network_isolation=None,
        model_name=None,
    ):
        """Return a ``Transformer`` that uses a SageMaker Model based on the training job.

        It reuses the SageMaker Session and base job name used by the Estimator.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example, 'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in a single request
                (default: None). Valid values: 'MultiRecord' and 'SingleRecord'.
            assemble_with (str): How the output is assembled (default: None). Valid values: 'Line'
                or 'None'.
            output_path (str): S3 location for saving the transform result. If not specified,
                results are stored to a default bucket.
            output_kms_key (str): Optional. KMS key ID for encrypting the transform output
                (default: None).
            accept (str): The accept header passed by the client to
                the inference endpoint. If it is supported by the endpoint,
                it will be the format of the batch transform output.
            env (dict): Environment variables to be set for use during the transform job
                (default: None).
            max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
                each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP request to the
                container in MB.
            tags (list[dict]): List of tags for labeling a transform job. If none specified, then
                the tags used for the training job are used for the transform job.
            role (str): The IAM Role ARN for the ``TensorFlowModel``, which is also used
                during transform jobs. If not specified, the role from the Estimator is used.
            volume_kms_key (str): Optional. KMS key ID for encrypting the volume attached to the ML
                compute instance (default: None).
            entry_point (str): Path (absolute or relative) to the local Python source file which
                should be executed as the entry point to training. If ``source_dir`` is specified,
                then ``entry_point`` must point to a file located at the root of ``source_dir``.
                If not specified and ``endpoint_type`` is 'tensorflow-serving',
                no entry point is used. If ``endpoint_type`` is also ``None``,
                then the training entry point is used.
            vpc_config_override (dict[str, list[str]]): Optional override for
                the VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.

                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode. Network isolation mode restricts
                the container access to outside networks (such as the internet).
                The container does not make any inbound or outbound network
                calls. If True, a channel named "code" will be created for any
                user entry script for inference. Also known as Internet-free mode.
                If not specified, this setting is taken from the estimator's
                current configuration.
            model_name (str): Name to use for creating an Amazon SageMaker
                model. If not specified, the estimator generates a default job name
                based on the training image name and current timestamp.
        """
        role = role or self.role
        model_name = self._get_or_create_name(model_name)

        if self.latest_training_job is None:
            logger.warning(
                "No finished training job found associated with this estimator. Please make sure "
                "this estimator is only used for building workflow config"
            )
            return Transformer(
                model_name,
                instance_count,
                instance_type,
                strategy=strategy,
                assemble_with=assemble_with,
                output_path=output_path,
                output_kms_key=output_kms_key,
                accept=accept,
                max_concurrent_transforms=max_concurrent_transforms,
                max_payload=max_payload,
                env=env or {},
                tags=tags,
                base_transform_job_name=self.base_job_name,
                volume_kms_key=volume_kms_key,
                sagemaker_session=self.sagemaker_session,
            )

        if enable_network_isolation is None:
            enable_network_isolation = self.enable_network_isolation()

        model = self.create_model(
            role=role,
            vpc_config_override=vpc_config_override,
            entry_point=entry_point,
            enable_network_isolation=enable_network_isolation,
            name=model_name,
        )

        return model.transformer(
            instance_count,
            instance_type,
            strategy=strategy,
            assemble_with=assemble_with,
            output_path=output_path,
            output_kms_key=output_kms_key,
            accept=accept,
            env=env,
            max_concurrent_transforms=max_concurrent_transforms,
            max_payload=max_payload,
            tags=tags,
            volume_kms_key=volume_kms_key,
        )
