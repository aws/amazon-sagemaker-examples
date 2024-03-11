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

from sagemaker import image_uris
from sagemaker.deprecations import renamed_kwargs
from sagemaker.estimator import Framework
from sagemaker.fw_utils import (
    framework_name_from_image,
    framework_version_from_tag,
    validate_version_or_image_args,
)
from sagemaker.sklearn import defaults
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow import is_pipeline_variable

logger = logging.getLogger("sagemaker")


class SKLearn(Framework):
    """Handle end-to-end training and deployment of custom Scikit-learn code."""

    _framework_name = defaults.SKLEARN_NAME

    def __init__(
        self,
        entry_point: Union[str, PipelineVariable],
        framework_version: Optional[str] = None,
        py_version: str = "py3",
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        image_uri_region: Optional[str] = None,
        **kwargs
    ):
        """Creates a SKLearn Estimator for Scikit-learn environment.

        It will execute an Scikit-learn script within a SageMaker Training Job. The managed
        Scikit-learn environment is an Amazon-built Docker container that executes functions
        defined in the supplied ``entry_point`` Python script.

        Training is started by calling
        :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling
        :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
        SageMaker endpoint and returns an
        :class:`~sagemaker.amazon.sklearn.model.SKLearnPredictor` instance that
        can be used to perform inference against the hosted model.

        Technical documentation on preparing Scikit-learn scripts for
        SageMaker training and using the Scikit-learn Estimator is available on
        the project home-page: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str or PipelineVariable): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            framework_version (str): Scikit-learn version you want to use for
                executing your model training code. Defaults to ``None``. Required
                unless ``image_uri`` is provided. List of supported versions:
                https://github.com/aws/sagemaker-python-sdk#sklearn-sagemaker-estimators
            py_version (str): Python version you want to use for executing your
                model training code (default: 'py3'). Currently, 'py3' is the only
                supported version. If ``None`` is passed in, ``image_uri`` must be
                provided.
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
            image_uri (str or PipelineVariable)): If specified, the estimator will use this image
                for training and hosting, instead of selecting the appropriate
                SageMaker official image based on framework_version and
                py_version. It can be an ECR url or dockerhub image and tag.

                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.

                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If also ``None``, then a ``ValueError``
                will be raised.
            image_uri_region (str): If ``image_uri`` argument is None, the image uri
                associated with this object will be in this region.
                Default: region associated with SageMaker session.
            **kwargs: Additional kwargs passed to the
                :class:`~sagemaker.estimator.Framework` constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        instance_type = renamed_kwargs(
            "train_instance_type", "instance_type", kwargs.get("instance_type"), kwargs
        )
        instance_count = renamed_kwargs(
            "train_instance_count", "instance_count", kwargs.get("instance_count"), kwargs
        )
        validate_version_or_image_args(framework_version, py_version, image_uri)
        if py_version and py_version != "py3":
            raise AttributeError(
                "Scikit-learn image only supports Python 3. Please use 'py3' for py_version."
            )
        self.framework_version = framework_version
        self.py_version = py_version

        # SciKit-Learn does not support distributed training or training on GPU instance types.
        # Fail fast.
        _validate_not_gpu_instance_type(instance_type)

        if instance_count:
            instance_cnt_err_msg = (
                "Scikit-Learn does not support distributed training. Please remove the "
                "'instance_count' argument or set 'instance_count=1' when initializing SKLearn."
            )
            if is_pipeline_variable(instance_count):
                raise TypeError(
                    "Invalid type of instance_count (PipelineVariable - {}). ".format(
                        type(instance_count)
                    )
                    + instance_cnt_err_msg
                )

            if instance_count != 1:
                raise AttributeError(instance_cnt_err_msg)

        super(SKLearn, self).__init__(
            entry_point,
            source_dir,
            hyperparameters,
            image_uri=image_uri,
            **dict(kwargs, instance_count=1)
        )

        if image_uri is None:
            self.image_uri = image_uris.retrieve(
                SKLearn._framework_name,
                image_uri_region or self.sagemaker_session.boto_region_name,
                version=self.framework_version,
                py_version=self.py_version,
                instance_type=instance_type,
            )

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
        """Create a SageMaker ``SKLearnModel`` object that can be deployed to an ``Endpoint``.

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
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.sklearn.model.SKLearnModel`
                constructor.

        Returns:
            sagemaker.sklearn.model.SKLearnModel: A SageMaker ``SKLearnModel``
            object. See :func:`~sagemaker.sklearn.model.SKLearnModel` for full details.
        """
        role = role or self.role
        kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        if "image_uri" not in kwargs:
            kwargs["image_uri"] = self.image_uri

        if "enable_network_isolation" not in kwargs:
            kwargs["enable_network_isolation"] = self.enable_network_isolation()

        return SKLearnModel(
            self.model_data,
            role,
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
                model data will be downloaded (default: None).

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(SKLearn, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
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

        if framework and framework != "scikit-learn":
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    job_details["TrainingJobName"]
                )
            )

        return init_params


def _validate_not_gpu_instance_type(training_instance_type):
    """Placeholder docstring."""
    gpu_instance_types = [
        "ml.p2.xlarge",
        "ml.p2.8xlarge",
        "ml.p2.16xlarge",
        "ml.p3.xlarge",
        "ml.p3.8xlarge",
        "ml.p3.16xlarge",
    ]

    if is_pipeline_variable(training_instance_type):
        warn_msg = (
            "instance_type is a PipelineVariable (%s). "
            "Its interpreted value in execution time should not be of GPU types "
            "since GPU training is not supported for Scikit-Learn."
        )
        logger.warning(warn_msg, type(training_instance_type))
        return

    if training_instance_type in gpu_instance_types:
        raise ValueError(
            "GPU training is not supported for Scikit-Learn. "
            "Please pick a different instance type from here: "
            "https://aws.amazon.com/ec2/instance-types/"
        )
