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
from sagemaker.estimator import Framework, _TrainingJob
from sagemaker.fw_utils import (
    framework_name_from_image,
    framework_version_from_tag,
    UploadedCode,
)
from sagemaker.session import Session
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.xgboost import defaults
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.xgboost.utils import validate_py_version, validate_framework_version

logger = logging.getLogger("sagemaker")


class XGBoost(Framework):
    """Handle end-to-end training and deployment of XGBoost booster training.

    It can also handle training using customer provided XGBoost entry point script.
    """

    _framework_name = defaults.XGBOOST_NAME

    def __init__(
        self,
        entry_point: Union[str, PipelineVariable],
        framework_version: str,
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        py_version: str = "py3",
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        image_uri_region: Optional[str] = None,
        **kwargs
    ):
        """An estimator that executes an XGBoost-based SageMaker Training Job.

        The managed XGBoost environment is an Amazon-built Docker container thatexecutes functions
        defined in the supplied ``entry_point`` Python script.

        Training is started by calling :meth:`~sagemaker.amazon.estimator.Framework.fit` on this
        Estimator. After training is complete, calling
        :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted SageMaker endpoint
        and returns an :class:`~sagemaker.amazon.xgboost.model.XGBoostPredictor` instance that
        can be used to perform inference against the hosted model.

        Technical documentation on preparing XGBoost scripts for SageMaker training and using the
        XGBoost Estimator is available on the project home-page:
        https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str or PipelineVariable): Path (absolute or relative) to
                the Python source file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point`` must point to
                a file located at the root of ``source_dir``.
            framework_version (str): XGBoost version you want to use for executing your model
                training code.
            source_dir (str or PipelineVariable): Path (absolute, relative or an S3 URI) to
                a directory with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory are preserved
                when training on Amazon SageMaker.
            hyperparameters (dict[str, str] or dict[str, PipelineVariable]): Hyperparameters
                that will be used for training (default: None).
                The hyperparameters are made accessible as a dict[str, str] to the training code
                on SageMaker. For convenience, this accepts other types for keys and values, but
                ``str()`` will be called to convert them before training.
            py_version (str): Python version you want to use for executing your model
                training code (default: 'py3').
            image_uri (str or PipelineVariable): If specified, the estimator will use this image
                for training and hosting, instead of selecting the appropriate SageMaker official
                image based on framework_version and py_version. It can be an ECR url or
                dockerhub image and tag.
                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.
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
        super(XGBoost, self).__init__(
            entry_point, source_dir, hyperparameters, image_uri=image_uri, **kwargs
        )

        self.py_version = py_version
        self.framework_version = framework_version

        validate_py_version(py_version)
        validate_framework_version(framework_version)

        if image_uri is None:
            self.image_uri = image_uris.retrieve(
                self._framework_name,
                image_uri_region or self.sagemaker_session.boto_region_name,
                version=framework_version,
                py_version=self.py_version,
                instance_type=instance_type,
                image_scope="training",
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
        """Create a SageMaker ``XGBoostModel`` object that can be deployed to an ``Endpoint``.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used
                during transform jobs. If not specified, the role from the Estimator will be used.
            model_server_workers (int): Optional. The number of worker processes used by the
                inference server. If None, server will use one worker per vCPU.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the
                model.
                Default: use subnets and security groups from this Estimator.
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
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.xgboost.model.XGBoostModel`
                constructor.

        Returns:
            sagemaker.xgboost.model.XGBoostModel: A SageMaker ``XGBoostModel`` object.
                See :func:`~sagemaker.xgboost.model.XGBoostModel` for full details.
        """
        role = role or self.role
        kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        if "image_uri" not in kwargs:
            kwargs["image_uri"] = self.image_uri

        return XGBoostModel(
            self.model_data,
            role,
            entry_point or self._model_entry_point(),
            framework_version=self.framework_version,
            source_dir=(source_dir or self._model_source_dir()),
            container_log_level=self.container_log_level,
            code_location=self.code_location,
            py_version=self.py_version,
            model_server_workers=model_server_workers,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            dependencies=(dependencies or self.dependencies),
            **kwargs
        )

    @classmethod
    def attach(cls, training_job_name, sagemaker_session=None, model_channel_name="model"):
        """Attach to an existing training job.

        Create an Estimator bound to an existing training job, each subclass
        is responsible to implement
        ``_prepare_init_params_from_job_description()`` as this method delegates
        the actual conversion of a training job description to the arguments
        that the class constructor expects. After attaching, if the training job
        has a Complete status, it can be ``deploy()`` ed to create a SageMaker
        Endpoint and return a ``Predictor``.

        If the training job is in progress, attach will block and display log
        messages from the training job, until the training job completes.

        Examples:
            >>> my_estimator.fit(wait=False)
            >>> training_job_name = my_estimator.latest_training_job.name
            Later on:
            >>> attached_estimator = Estimator.attach(training_job_name)
            >>> attached_estimator.deploy()

        Args:
            training_job_name (str): The name of the training job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded (default: 'model'). If no channel
                with the same name exists in the training job, this option will
                be ignored.

        Returns:
            Instance of the calling ``Estimator`` Class with the attached
            training job.
        """
        sagemaker_session = sagemaker_session or Session()

        job_details = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        init_params = cls._prepare_init_params_from_job_description(job_details, model_channel_name)
        tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=job_details["TrainingJobArn"]
        )["Tags"]
        init_params.update(tags=tags)

        estimator = cls(sagemaker_session=sagemaker_session, **init_params)
        estimator.latest_training_job = _TrainingJob(
            sagemaker_session=sagemaker_session, job_name=training_job_name
        )
        estimator._current_job_name = estimator.latest_training_job.name
        estimator.latest_training_job.wait()

        # pylint gets confused thinking that estimator is an EstimatorBase instance, but it actually
        # is a Framework or any of its derived classes. We can safely ignore the no-member errors.
        estimator.uploaded_code = UploadedCode(
            estimator.source_dir, estimator.entry_point  # pylint: disable=no-member
        )
        return estimator

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(XGBoost, cls)._prepare_init_params_from_job_description(job_details)

        image_uri = init_params.pop("image_uri")
        framework, py_version, tag, _ = framework_name_from_image(image_uri)
        init_params["py_version"] = py_version

        if framework and framework != cls._framework_name:
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    job_details["TrainingJobName"]
                )
            )
        init_params["framework_version"] = framework_version_from_tag(tag)

        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params["image_uri"] = image_uri
        return init_params
