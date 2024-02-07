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
from typing import Union, Optional, Dict

from packaging.version import Version

from sagemaker.deprecations import renamed_kwargs
from sagemaker.estimator import Framework
from sagemaker.fw_utils import (
    framework_name_from_image,
    framework_version_from_tag,
    python_deprecation_warning,
    validate_version_or_image_args,
    warn_if_parameter_server_with_multi_gpu,
)
from sagemaker.mxnet import defaults
from sagemaker.mxnet.model import MXNetModel
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger("sagemaker")


class MXNet(Framework):
    """Handle end-to-end training and deployment of custom MXNet code."""

    _framework_name = "mxnet"
    _LOWEST_SCRIPT_MODE_VERSION = ["1", "3"]

    def __init__(
        self,
        entry_point: Union[str, PipelineVariable],
        framework_version: Optional[str] = None,
        py_version: Optional[str] = None,
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        distribution: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """This ``Estimator`` executes an MXNet script in a managed MXNet execution environment.

        The managed MXNet environment is an Amazon-built Docker container that executes
        functions defined in the supplied ``entry_point`` Python script.

        Training is started by calling
        :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling
        :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
        SageMaker endpoint and returns an
        :class:`~sagemaker.amazon.mxnet.model.MXNetPredictor` instance that can
        be used to perform inference against the hosted model.

        Technical documentation on preparing MXNet scripts for SageMaker
        training and using the MXNet Estimator is available on the project
        home-page: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str or PipelineVariable): Path (absolute or relative) to the
                Python source file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            framework_version (str): MXNet version you want to use for executing
                your model training code. Defaults to `None`. Required unless
                ``image_uri`` is provided. List of supported versions.
                https://github.com/aws/sagemaker-python-sdk#mxnet-sagemaker-estimators.
            py_version (str): Python version you want to use for executing your
                model training code. One of 'py2' or 'py3'. Defaults to ``None``. Required
                unless ``image_uri`` is provided.
            source_dir (str or PipelineVariable): Path (absolute, relative or an S3 URI) to
                a directory with any other training source code dependencies aside from the entry
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
                for training and hosting, instead of selecting the appropriate SageMaker official
                image based on framework_version and py_version. It can be an ECR url or dockerhub
                image and tag.

                Examples:
                    * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
                    * ``custom-image:latest``

                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If also ``None``, then a ``ValueError``
                will be raised.
            distribution (dict): A dictionary with information on how to run distributed
                training (default: None). Currently we support distributed training with
                parameter server and MPI [Horovod].
                To enable parameter server use the following setup:

                .. code:: python

                    {
                        'parameter_server':
                        {
                            'enabled': True
                        }
                    }

                To enable MPI:

                .. code:: python

                    {
                        'mpi':
                        {
                            'enabled': True
                        }
                    }

                Option parameters within ``mpi`` are ``processes_per_host``
                and ``custom_mpi_options``.

                .. code:: python

                    {
                        'mpi':
                        {
                            'enabled': True,
                            'processes_per_host': 2,
                            'custom_mpi_options': '-verbose --NCCL_DEBUG=INFO'
                        }
                    }

            **kwargs: Additional kwargs passed to the
                :class:`~sagemaker.estimator.Framework` constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        distribution = renamed_kwargs("distributions", "distribution", distribution, kwargs)
        instance_type = renamed_kwargs(
            "train_instance_type", "instance_type", kwargs.get("instance_type"), kwargs
        )
        validate_version_or_image_args(framework_version, py_version, image_uri)
        if py_version == "py2":
            logger.warning(
                python_deprecation_warning(self._framework_name, defaults.LATEST_PY2_VERSION)
            )
        self.framework_version = framework_version
        self.py_version = py_version

        if "enable_sagemaker_metrics" not in kwargs:
            # enable sagemaker metrics for MXNet v1.6 or greater:
            if self.framework_version and Version(self.framework_version) >= Version("1.6"):
                kwargs["enable_sagemaker_metrics"] = True

        super(MXNet, self).__init__(
            entry_point, source_dir, hyperparameters, image_uri=image_uri, **kwargs
        )

        if distribution is not None:
            warn_if_parameter_server_with_multi_gpu(
                training_instance_type=instance_type, distribution=distribution
            )

        self._configure_distribution(distribution)

    def _configure_distribution(self, distribution):
        """Placeholder docstring"""
        if distribution is None:
            return

        if (
            self.framework_version
            and self.framework_version.split(".") < self._LOWEST_SCRIPT_MODE_VERSION
        ):
            raise ValueError(
                "The distribution option is valid for only versions {} and higher".format(
                    ".".join(self._LOWEST_SCRIPT_MODE_VERSION)
                )
            )

        if "parameter_server" in distribution:
            enabled = distribution["parameter_server"].get("enabled", False)
            self._hyperparameters[self.LAUNCH_PS_ENV_NAME] = enabled

        if "mpi" in distribution:
            mpi_dict = distribution["mpi"]
            mpi_enabled = mpi_dict.get("enabled", False)
            self._hyperparameters[self.LAUNCH_MPI_ENV_NAME] = mpi_enabled

            if mpi_dict.get("processes_per_host"):
                self._hyperparameters[self.MPI_NUM_PROCESSES_PER_HOST] = mpi_dict.get(
                    "processes_per_host"
                )

                self._hyperparameters[self.MPI_CUSTOM_MPI_OPTIONS] = mpi_dict.get(
                    "custom_mpi_options", ""
                )

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        image_uri=None,
        **kwargs
    ):
        """Create a SageMaker ``MXNetModel`` object that can be deployed to an ``Endpoint``.

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
                If not specified, the training entry point is used.
            source_dir (str): Path (absolute or relative) to a directory with any other serving
                source code dependencies aside from the entry point file.
                If not specified, the model source directory from training is used.
            dependencies (list[str]): A list of paths to directories (absolute or relative) with
                any additional libraries that will be exported to the container.
                If not specified, the dependencies from training are used.
                This is not supported with "local code" in Local Mode.
            image_uri (str): If specified, the estimator will use this image for hosting, instead
                of selecting the appropriate SageMaker official image based on framework_version
                and py_version. It can be an ECR url or dockerhub image and tag.

                Examples:
                    * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
                    * ``custom-image:latest``

            **kwargs: Additional kwargs passed to the :class:`~sagemaker.mxnet.model.MXNetModel`
                constructor.

        Returns:
            sagemaker.mxnet.model.MXNetModel: A SageMaker ``MXNetModel`` object.
            See :func:`~sagemaker.mxnet.model.MXNetModel` for full details.
        """
        if "image_uri" not in kwargs:
            kwargs["image_uri"] = image_uri or self.image_uri

        kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        model = MXNetModel(
            self.model_data,
            role or self.role,
            entry_point,
            framework_version=self.framework_version,
            py_version=self.py_version,
            source_dir=(source_dir or self._model_source_dir()),
            container_log_level=self.container_log_level,
            code_location=self.code_location,
            model_server_workers=model_server_workers,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            dependencies=(dependencies or self.dependencies),
            **kwargs
        )

        if entry_point is None:
            model.entry_point = (
                self.entry_point if model._is_mms_version() else self._model_entry_point()
            )

        return model

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(MXNet, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
        image_uri = init_params.pop("image_uri")
        framework, py_version, tag, _ = framework_name_from_image(image_uri)

        # We switched image tagging scheme from regular image version (e.g. '1.0') to more
        # expressive containing framework version, device type and python version
        # (e.g. '0.12-gpu-py2'). For backward compatibility map deprecated image tag '1.0' to a
        # '0.12' framework version otherwise extract framework version from the tag itself.
        if tag is None:
            framework_version = None
        elif tag == "1.0":
            framework_version = "0.12"
        else:
            framework_version = framework_version_from_tag(tag)
        init_params["framework_version"] = framework_version
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
