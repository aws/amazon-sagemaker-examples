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

from sagemaker.estimator import Framework, EstimatorBase
from sagemaker.fw_utils import (
    framework_name_from_image,
    framework_version_from_tag,
    python_deprecation_warning,
    validate_version_or_image_args,
)
from sagemaker.chainer import defaults
from sagemaker.chainer.model import ChainerModel
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger("sagemaker")


class Chainer(Framework):
    """Handle end-to-end training and deployment of custom Chainer code."""

    _framework_name: str = "chainer"

    # Hyperparameters
    _use_mpi: str = "sagemaker_use_mpi"
    _num_processes: str = "sagemaker_num_processes"
    _process_slots_per_host: str = "sagemaker_process_slots_per_host"
    _additional_mpi_options: str = "sagemaker_additional_mpi_options"

    def __init__(
        self,
        entry_point: Union[str, PipelineVariable],
        use_mpi: Optional[Union[bool, PipelineVariable]] = None,
        num_processes: Optional[Union[int, PipelineVariable]] = None,
        process_slots_per_host: Optional[Union[int, PipelineVariable]] = None,
        additional_mpi_options: Optional[Union[str, PipelineVariable]] = None,
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        framework_version: Optional[str] = None,
        py_version: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        **kwargs
    ):
        """This ``Estimator`` executes an Chainer script in a managed execution environment.

        The managed Chainer environment is an Amazon-built Docker container that executes functions
        defined in the supplied ``entry_point`` Python script within a SageMaker Training Job.

        Training is started by calling
        :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling
        :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
        SageMaker endpoint and returns an
        :class:`~sagemaker.amazon.chainer.model.ChainerPredictor` instance that
        can be used to perform inference against the hosted model.

        Technical documentation on preparing Chainer scripts for SageMaker
        training and using the Chainer Estimator is available on the project
        home-page: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str or PipelineVariable): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            use_mpi (bool or PipelineVariable): If true, entry point is run as an MPI script. By
                default, the Chainer Framework runs the entry point with
                'mpirun' if more than one instance is used.
            num_processes (int or PipelineVariable): Total number of processes to run the entry
                point with. By default, the Chainer Framework runs one process
                per GPU (on GPU instances), or one process per host (on CPU
                instances).
            process_slots_per_host (int or PipelineVariable): The number of processes that can run
                on each instance. By default, this is set to the number of GPUs
                on the instance (on GPU instances), or one (on CPU instances).
            additional_mpi_options (str or PipelineVariable): String of options to the 'mpirun'
                command used to run the entry point. For example, '-X
                NCCL_DEBUG=WARN' will pass that option string to the mpirun
                command.
            source_dir (str or PipelineVariable): Path (absolute or relative) to a directory with
                any other training source code dependencies aside from the entry
                point file (default: None). Structure within this directory are
                preserved when training on Amazon SageMaker.
            hyperparameters (dict[str, str] or dict[str, PipelineVariable]): Hyperparameters
                that will be used for training (default: None). The hyperparameters are made
                accessible as a dict[str, str] to the training code on
                SageMaker. For convenience, this accepts other types for keys
                and values, but ``str()`` will be called to convert them before
                training.
            py_version (str): Python version you want to use for executing your
                model training code. Defaults to ``None``. Required unless ``image_uri``
                is provided.
            framework_version (str): Chainer version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided. List of supported versions:
                https://github.com/aws/sagemaker-python-sdk#chainer-sagemaker-estimators.
            image_uri (str): If specified, the estimator will use this image
                for training and hosting, instead of selecting the appropriate
                SageMaker official image based on framework_version and
                py_version. It can be an ECR url or dockerhub image and tag.

                Examples
                    * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
                    * ``custom-image:latest``

                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If also ``None``, then a ``ValueError``
                will be raised.
            **kwargs: Additional kwargs passed to the
                :class:`~sagemaker.estimator.Framework` constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        validate_version_or_image_args(framework_version, py_version, image_uri)
        if py_version == "py2":
            logger.warning(
                python_deprecation_warning(self._framework_name, defaults.LATEST_PY2_VERSION)
            )
        self.framework_version = framework_version
        self.py_version = py_version

        super(Chainer, self).__init__(
            entry_point, source_dir, hyperparameters, image_uri=image_uri, **kwargs
        )

        self.use_mpi = use_mpi
        self.num_processes = num_processes
        self.process_slots_per_host = process_slots_per_host
        self.additional_mpi_options = additional_mpi_options

    def hyperparameters(self):
        """Return hyperparameters used by your custom Chainer code during training."""
        hyperparameters = super(Chainer, self).hyperparameters()

        additional_hyperparameters = {
            Chainer._use_mpi: self.use_mpi,
            Chainer._num_processes: self.num_processes,
            Chainer._process_slots_per_host: self.process_slots_per_host,
            Chainer._additional_mpi_options: self.additional_mpi_options,
        }

        # remove unset keys.
        additional_hyperparameters = {k: v for k, v in additional_hyperparameters.items() if v}
        hyperparameters.update(
            EstimatorBase._json_encode_hyperparameters(additional_hyperparameters)
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
        **kwargs
    ):
        """Create a SageMaker ``ChainerModel`` object that can be deployed to an ``Endpoint``.

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
            **kwargs: Additional kwargs passed to the ChainerModel constructor.

        Returns:
            sagemaker.chainer.model.ChainerModel: A SageMaker ``ChainerModel``
            object. See :func:`~sagemaker.chainer.model.ChainerModel` for full details.
        """
        kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        if "image_uri" not in kwargs:
            kwargs["image_uri"] = self.image_uri

        return ChainerModel(
            self.model_data,
            role or self.role,
            entry_point or self._model_entry_point(),
            source_dir=(source_dir or self._model_source_dir()),
            container_log_level=self.container_log_level,
            code_location=self.code_location,
            py_version=self.py_version,
            framework_version=self.framework_version,
            model_server_workers=model_server_workers,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            dependencies=(dependencies or self.dependencies),
            **kwargs
        )

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
        init_params = super(Chainer, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        for argument in [
            Chainer._use_mpi,
            Chainer._num_processes,
            Chainer._process_slots_per_host,
            Chainer._additional_mpi_options,
        ]:

            value = init_params["hyperparameters"].pop(argument, None)
            if value:
                init_params[argument[len("sagemaker_") :]] = value

        image_uri = init_params.pop("image_uri")
        framework, py_version, tag, _ = framework_name_from_image(image_uri)

        if tag is None:
            framework_version = None
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
