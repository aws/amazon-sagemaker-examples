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
import re
from typing import Optional, Union, Dict

from sagemaker.estimator import Framework, EstimatorBase
from sagemaker.fw_utils import (
    framework_name_from_image,
    validate_distribution,
)
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

from sagemaker.huggingface.training_compiler.config import TrainingCompilerConfig
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger("sagemaker")


class HuggingFace(Framework):
    """Handle training of custom HuggingFace code."""

    _framework_name = "huggingface"
    LAUNCH_PYTORCH_DDP_ENV_NAME = "sagemaker_pytorch_ddp_enabled"
    LAUNCH_TORCH_DISTRIBUTED_ENV_NAME = "sagemaker_torch_distributed_enabled"
    INSTANCE_TYPE_ENV_NAME = "sagemaker_instance_type"

    def __init__(
        self,
        py_version: str,
        entry_point: Union[str, PipelineVariable],
        transformers_version: Optional[str] = None,
        tensorflow_version: Optional[str] = None,
        pytorch_version: Optional[str] = None,
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        distribution: Optional[Dict] = None,
        compiler_config: Optional[TrainingCompilerConfig] = None,
        **kwargs,
    ):
        """This estimator runs a Hugging Face training script in a SageMaker training environment.

        The estimator initiates the SageMaker-managed Hugging Face environment
        by using the pre-built Hugging Face Docker container and runs
        the Hugging Face training script that user provides through
        the ``entry_point`` argument.

        After configuring the estimator class, use the class method
        :meth:`~sagemaker.amazon.estimator.Framework.fit()` to start a training job.

        Args:
            py_version (str): Python version you want to use for executing your model training
                code. Defaults to ``None``. Required unless ``image_uri`` is provided.  If
                using PyTorch, the current supported version is ``py36``. If using TensorFlow,
                the current supported version is ``py37``.
            entry_point (str or PipelineVariable): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            transformers_version (str): Transformers version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided. The current supported version is ``4.6.1``.
            tensorflow_version (str): TensorFlow version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``pytorch_version`` is provided. The current supported version is ``2.4.1``.
            pytorch_version (str): PyTorch version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``tensorflow_version`` is provided. The current supported versions are ``1.7.1`` and ``1.6.0``.
            source_dir (str or PipelineVariable): Path (absolute, relative or an S3 URI) to a
                directory with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory are preserved
                when training on Amazon SageMaker.
            hyperparameters (dict[str, str] or dict[str, PipelineVariable]): Hyperparameters
                that will be used for training (default: None). The hyperparameters are made
                accessible as a dict[str, str] to the training code on
                SageMaker. For convenience, this accepts other types for keys
                and values, but ``str()`` will be called to convert them before
                training.
            image_uri (str or PipelineVariable): If specified, the estimator will use this image
                for training and hosting, instead of selecting the appropriate
                SageMaker official image based on framework_version and
                py_version. It can be an ECR url or dockerhub image and tag.
                Examples:
                    * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
                    * ``custom-image:latest``

                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If also ``None``, then a ``ValueError``
                will be raised.
            distribution (dict): A dictionary with information on how to run distributed training
                (default: None).  Currently, the following are supported:
                distributed training with parameter servers, SageMaker Distributed (SMD) Data
                and Model Parallelism, and MPI. SMD Model Parallelism can only be used with MPI.
                To enable parameter server use the following setup:

                .. code:: python

                    {
                        "parameter_server": {
                            "enabled": True
                        }
                    }

                To enable MPI:

                .. code:: python

                    {
                        "mpi": {
                            "enabled": True
                        }
                    }

                To enable SMDistributed Data Parallel or Model Parallel:

                .. code:: python

                    {
                        "smdistributed": {
                            "dataparallel": {
                                "enabled": True
                            },
                            "modelparallel": {
                                "enabled": True,
                                "parameters": {}
                            }
                        }
                    }

                **To enable PyTorch DDP:**

                    .. code:: python

                        {
                            "pytorchddp": {
                                "enabled": True
                            }
                        }

                    To learn more, see `Distributed PyTorch Training
                    <https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#distributed-pytorch-training>`_.

                **To enable Torch Distributed:**

                    This is available for general distributed training on
                    GPU instances from PyTorch v1.13.1 and later.

                    .. code:: python

                        {
                            "torch_distributed": {
                                "enabled": True
                            }
                        }

                    This option also supports distributed training on Trn1.
                    To learn more, see `Distributed PyTorch Training on Trainium
                    <https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#distributed-pytorch-training-on-trainium>`_.

                To enable distributed training with
                `SageMaker Training Compiler <https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html>`_
                for Hugging Face Transformers with PyTorch:

                .. code:: python

                    {
                        "pytorchxla": {
                            "enabled": True
                        }
                    }

                To learn more, see `SageMaker Training Compiler
                <https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html>`_
                in the *Amazon SageMaker Developer Guide*.

                .. note::

                    When you use this PyTorch XLA option for distributed training strategy,
                    you must add the ``compiler_config`` parameter and activate SageMaker
                    Training Compiler.
            compiler_config (:class:`~sagemaker.huggingface.TrainingCompilerConfig`):
                Configures SageMaker Training Compiler to accelerate training.

            **kwargs: Additional kwargs passed to the :class:`~sagemaker.estimator.Framework`
                constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        self.framework_version = transformers_version
        self.py_version = py_version
        self.tensorflow_version = tensorflow_version
        self.pytorch_version = pytorch_version

        self._validate_args(image_uri=image_uri)

        if "enable_sagemaker_metrics" not in kwargs:
            kwargs["enable_sagemaker_metrics"] = True

        kwargs["py_version"] = self.py_version

        super(HuggingFace, self).__init__(
            entry_point, source_dir, hyperparameters, image_uri=image_uri, **kwargs
        )

        if "entry_point" not in kwargs:
            kwargs["entry_point"] = entry_point

        self.base_framework_name = "tensorflow" if tensorflow_version is not None else "pytorch"
        self.base_framework_version = (
            tensorflow_version if tensorflow_version is not None else pytorch_version
        )

        if distribution is not None:
            distribution = validate_distribution(
                distribution,
                self.instance_groups,
                self.base_framework_name,
                self.base_framework_version,
                py_version,
                image_uri,
                kwargs,
            )

        self.distribution = distribution or {}

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
        elif distribution is not None and "pytorchxla" in distribution:
            raise ValueError(
                "Distributed training through PyTorch XLA is currently only supported "
                "when SageMaker Training Compiler is enabled. To learn more, "
                "see Enable SageMaker Training Compiler at "
                "https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler-enable.html."
            )
        self.compiler_config = compiler_config

    def _validate_args(self, image_uri):
        """Placeholder docstring"""
        if image_uri is not None:
            return

        if self.framework_version is None and image_uri is None:
            raise ValueError(
                "transformers_version, and image_uri are both None. "
                "Specify either transformers_version or image_uri"
            )
        if self.tensorflow_version is not None and self.pytorch_version is not None:
            raise ValueError(
                "tensorflow_version and pytorch_version are both not None. "
                "Specify only tensorflow_version or pytorch_version."
            )
        if self.tensorflow_version is None and self.pytorch_version is None:
            raise ValueError(
                "tensorflow_version and pytorch_version are both None. "
                "Specify either tensorflow_version or pytorch_version."
            )
        base_framework_version_len = (
            len(self.tensorflow_version.split("."))
            if self.tensorflow_version is not None
            else len(self.pytorch_version.split("."))
        )
        transformers_version_len = len(self.framework_version.split("."))
        if transformers_version_len != base_framework_version_len:
            raise ValueError(
                "Please use either full version or shortened version for both "
                "transformers_version, tensorflow_version and pytorch_version."
            )

    def _huggingface_distribution_configuration(self, distribution):
        """Returns a dict of distribution config for Hugging Face training

        Args:
            distribution (dict): A dictionary with information on how to run distributed training.
        Returns:
            dict containing Pytorch DDP config
        """
        distribution_config = {}
        pytorch_ddp_enabled = False
        torch_distributed_enabled = False

        if "pytorchddp" in distribution:
            pytorch_ddp_enabled = distribution.get("pytorchddp").get("enabled", False)
        elif "torch_distributed" in distribution:
            torch_distributed_enabled = distribution.get("torch_distributed").get("enabled", False)

        if pytorch_ddp_enabled:
            distribution_config[self.LAUNCH_PYTORCH_DDP_ENV_NAME] = pytorch_ddp_enabled
            if self.instance_type is not None:
                distribution_config[self.INSTANCE_TYPE_ENV_NAME] = self.instance_type
        elif torch_distributed_enabled:
            distribution_config[self.LAUNCH_TORCH_DISTRIBUTED_ENV_NAME] = torch_distributed_enabled
            if self.instance_type is not None:
                distribution_config[self.INSTANCE_TYPE_ENV_NAME] = self.instance_type
        else:
            distribution_config = self._distribution_configuration(distribution=distribution)

        return distribution_config

    def hyperparameters(self):
        """Return hyperparameters used by your custom PyTorch code during model training."""
        hyperparameters = super(HuggingFace, self).hyperparameters()
        additional_hyperparameters = self._huggingface_distribution_configuration(
            distribution=self.distribution
        )
        hyperparameters.update(
            EstimatorBase._json_encode_hyperparameters(additional_hyperparameters)
        )

        if self.compiler_config:
            training_compiler_hyperparameters = self.compiler_config._to_hyperparameter_dict()
            hyperparameters.update(
                EstimatorBase._json_encode_hyperparameters(training_compiler_hyperparameters)
            )

        return hyperparameters

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        **kwargs,
    ):
        """Create a SageMaker ``HuggingFaceModel`` object that can be deployed to an ``Endpoint``.

        Args:
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            entry_point (str): Path (absolute or relative) to the local Python source file which
                should be executed as the entry point to training. If ``source_dir`` is specified,
                then ``entry_point`` must point to a file located at the root of ``source_dir``.
                Defaults to `None`.
            source_dir (str): Path (absolute or relative) to a directory with any other serving
                source code dependencies aside from the entry point file.
                If not specified, the model source directory from training is used.
            dependencies (list[str]): A list of paths to directories (absolute or relative) with
                any additional libraries that will be exported to the container.
                If not specified, the dependencies from training are used.
                This is not supported with "local code" in Local Mode.
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.huggingface.model.HuggingFaceModel`
                constructor.
        Returns:
            sagemaker.huggingface.model.HuggingFaceModel: A SageMaker ``HuggingFaceModel``
            object. See :func:`~sagemaker.huggingface.model.HuggingFaceModel` for full details.
        """
        if "image_uri" not in kwargs:
            kwargs["image_uri"] = self.image_uri

        kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        return HuggingFaceModel(
            role or self.role,
            model_data=self.model_data,
            entry_point=entry_point,
            transformers_version=self.framework_version,
            tensorflow_version=self.tensorflow_version,
            pytorch_version=self.pytorch_version,
            py_version=self.py_version,
            source_dir=(source_dir or self._model_source_dir()),
            container_log_level=self.container_log_level,
            code_location=self.code_location,
            model_server_workers=model_server_workers,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            dependencies=(dependencies or self.dependencies),
            **kwargs,
        )

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details: The returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(HuggingFace, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
        image_uri = init_params.pop("image_uri")
        framework, py_version, tag, _ = framework_name_from_image(image_uri)

        if tag is None:
            framework_version = None
        else:
            framework, pt_or_tf = framework.split("-")[:2]
            tag_pattern = re.compile(r"^(.*)-transformers(.*)-(cpu|gpu)-(py2|py3\d*)$")
            tag_match = tag_pattern.match(tag)
            pt_or_tf_version = tag_match.group(1)
            framework_version = tag_match.group(2)
            if pt_or_tf == "pytorch":
                init_params["pytorch_version"] = pt_or_tf_version
            else:
                init_params["tensorflow_version"] = pt_or_tf_version

        init_params["transformers_version"] = framework_version
        init_params["py_version"] = py_version

        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params["image_uri"] = image_uri
            return init_params

        if framework != cls._framework_name:
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    job_details["TrainingJobName"]
                )
            )

        return init_params
