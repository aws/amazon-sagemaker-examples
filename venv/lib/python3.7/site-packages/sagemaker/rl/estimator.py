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

import enum
import logging
import re
from typing import Union, Optional, List, Dict

from sagemaker import image_uris, fw_utils
from sagemaker.estimator import Framework, EstimatorBase
from sagemaker.model import FrameworkModel, SAGEMAKER_OUTPUT_LOCATION
from sagemaker.mxnet.model import MXNetModel
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger("sagemaker")

SAGEMAKER_ESTIMATOR = "sagemaker_estimator"
SAGEMAKER_ESTIMATOR_VALUE = "RLEstimator"
PYTHON_VERSION = "py3"
TOOLKIT_FRAMEWORK_VERSION_MAP = {
    "coach": {
        "0.10.1": {"tensorflow": "1.11"},
        "0.10": {"tensorflow": "1.11"},
        "0.11.0": {"tensorflow": "1.11", "mxnet": "1.3"},
        "0.11.1": {"tensorflow": "1.12"},
        "0.11": {"tensorflow": "1.12", "mxnet": "1.3"},
        "1.0.0": {"tensorflow": "1.12"},
    },
    "ray": {
        "0.5.3": {"tensorflow": "1.11"},
        "0.5": {"tensorflow": "1.11"},
        "0.6.5": {"tensorflow": "1.12"},
        "0.6": {"tensorflow": "1.12"},
        "0.8.2": {"tensorflow": "2.1"},
        "0.8.5": {"tensorflow": "2.1", "pytorch": "1.5"},
        "1.6.0": {"tensorflow": "2.5.0", "pytorch": "1.8.1"},
    },
}


class RLToolkit(enum.Enum):
    """Placeholder docstring"""

    COACH = "coach"
    RAY = "ray"


class RLFramework(enum.Enum):
    """Placeholder docstring"""

    TENSORFLOW = "tensorflow"
    MXNET = "mxnet"
    PYTORCH = "pytorch"


class RLEstimator(Framework):
    """Handle end-to-end training and deployment of custom RLEstimator code."""

    COACH_LATEST_VERSION_TF = "0.11.1"
    COACH_LATEST_VERSION_MXNET = "0.11.0"
    RAY_LATEST_VERSION = "1.6.0"

    def __init__(
        self,
        entry_point: Union[str, PipelineVariable],
        toolkit: Optional[RLToolkit] = None,
        toolkit_version: Optional[str] = None,
        framework: Optional[Framework] = None,
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        metric_definitions: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        **kwargs
    ):
        """Creates an RLEstimator for managed Reinforcement Learning (RL).

        It will execute an RLEstimator script within a SageMaker Training Job. The managed RL
        environment is an Amazon-built Docker container that executes functions defined in the
        supplied ``entry_point`` Python script.

        Training is started by calling
        :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling
        :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
        SageMaker endpoint and based on the specified framework returns an
        :class:`~sagemaker.amazon.mxnet.model.MXNetPredictor` or
        :class:`~sagemaker.amazon.tensorflow.model.TensorFlowPredictor` instance that
        can be used to perform inference against the hosted model.

        Technical documentation on preparing RLEstimator scripts for
        SageMaker training and using the RLEstimator is available on the project
        homepage: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str or PipelineVariable): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            toolkit (sagemaker.rl.RLToolkit): RL toolkit you want to use for
                executing your model training code.
            toolkit_version (str): RL toolkit version you want to be use for
                executing your model training code.
            framework (sagemaker.rl.RLFramework): Framework (MXNet or
                TensorFlow) you want to be used as a toolkit backed for
                reinforcement learning training.
            source_dir (str or PipelineVariable): Path (absolute, relative or an S3 URI)
                to a directory with any other training source code dependencies aside from
                the entry point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory are preserved
                when training on Amazon SageMaker.
            hyperparameters (dict[str, str] or dict[str, PipelineVariable]): Hyperparameters
                that will be used for training (default: None). The hyperparameters are made
                accessible as a dict[str, str] to the training code on
                SageMaker. For convenience, this accepts other types for keys
                and values.
            image_uri (str or PipelineVariable): An ECR url. If specified, the estimator will use
                this image for training and hosting, instead of selecting the
                appropriate SageMaker official image based on framework_version
                and py_version. Example:
                123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
            metric_definitions (list[dict[str, str] or list[dict[str, PipelineVariable]]):
                A list of dictionaries that defines the metric(s) used to evaluate the
                training jobs. Each dictionary contains two keys: 'Name' for the name of the metric,
                and 'Regex' for the regular expression used to extract the
                metric from the logs. This should be defined only for jobs that
                don't use an Amazon algorithm.
            **kwargs: Additional kwargs passed to the
                :class:`~sagemaker.estimator.Framework` constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        self._validate_images_args(toolkit, toolkit_version, framework, image_uri)

        if not image_uri:
            self._validate_toolkit_support(toolkit.value, toolkit_version, framework.value)
            self.toolkit = toolkit.value
            self.toolkit_version = toolkit_version
            self.framework = framework.value
            self.framework_version = TOOLKIT_FRAMEWORK_VERSION_MAP[self.toolkit][
                self.toolkit_version
            ][self.framework]

            # set default metric_definitions based on the toolkit
            if not metric_definitions:
                metric_definitions = self.default_metric_definitions(toolkit)

        super(RLEstimator, self).__init__(
            entry_point,
            source_dir,
            hyperparameters,
            image_uri=image_uri,
            metric_definitions=metric_definitions,
            **kwargs
        )

    def create_model(
        self,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        **kwargs
    ):
        """Create a SageMaker ``RLEstimatorModel`` object that can be deployed to an Endpoint.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.

                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point for MXNet
                hosting (default: self.entry_point). If ``source_dir`` is specified,
                then ``entry_point`` must point to a file located at the root of ``source_dir``.
            source_dir (str): Path (absolute or relative) to a directory with
                any other training source code dependencies aside from the entry
                point file (default: self.source_dir). Structure within this
                directory are preserved when hosting on Amazon SageMaker.
            dependencies (list[str]): A list of paths to directories (absolute
                or relative) with any additional libraries that will be exported
                to the container (default: self.dependencies). The library
                folders will be copied to SageMaker in the same folder where the
                entry_point is copied. If the ```source_dir``` points to S3,
                code will be uploaded and the S3 location will be used instead.
                This is not supported with "local code" in Local Mode.
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.model.FrameworkModel`
                constructor.

        Returns:
            sagemaker.model.FrameworkModel: Depending on input parameters returns
                one of the following:

                * :class:`~sagemaker.model.FrameworkModel` - if ``image_uri`` is specified
                    on the estimator;
                * :class:`~sagemaker.mxnet.MXNetModel` - if ``image_uri`` isn't specified and
                    MXNet is used as the RL backend;
                * :class:`~sagemaker.tensorflow.model.TensorFlowModel` - if ``image_uri`` isn't
                    specified and TensorFlow is used as the RL backend.

        Raises:
            ValueError: If image_uri is not specified and framework enum is not valid.
        """
        base_args = dict(
            model_data=self.model_data,
            role=role or self.role,
            image_uri=kwargs.get("image_uri", self.image_uri),
            container_log_level=self.container_log_level,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
        )

        base_args["name"] = self._get_or_create_name(kwargs.get("name"))

        if not entry_point and (source_dir or dependencies):
            raise AttributeError("Please provide an `entry_point`.")

        entry_point = entry_point or self._model_entry_point()
        source_dir = source_dir or self._model_source_dir()
        dependencies = dependencies or self.dependencies

        extended_args = dict(
            entry_point=entry_point,
            source_dir=source_dir,
            code_location=self.code_location,
            dependencies=dependencies,
        )
        extended_args.update(base_args)

        if self.image_uri:
            return FrameworkModel(**extended_args)

        if self.toolkit == RLToolkit.RAY.value:
            raise NotImplementedError(
                "Automatic deployment of Ray models is not currently available."
                " Train policy parameters are available in model checkpoints"
                " in the TrainingJob output."
            )

        if self.framework == RLFramework.TENSORFLOW.value:
            return TensorFlowModel(framework_version=self.framework_version, **base_args)
        if self.framework == RLFramework.MXNET.value:
            return MXNetModel(
                framework_version=self.framework_version, py_version=PYTHON_VERSION, **extended_args
            )
        raise ValueError(
            "An unknown RLFramework enum was passed in. framework: {}".format(self.framework)
        )

    def training_image_uri(self):
        """Return the Docker image to use for training.

        The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does
        the model training, calls this method to find the image to use for model
        training.

        Returns:
            str: The URI of the Docker image.
        """
        if self.image_uri:
            return self.image_uri

        logger.info(
            "image_uri is not presented, retrieving image_uri based on instance_type, "
            "framework etc."
        )
        return image_uris.retrieve(
            self._image_framework(),
            self.sagemaker_session.boto_region_name,
            version=self.toolkit_version,
            instance_type=self.instance_type,
        )

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params.

        This is done so that the init params can be handled by the class constructor.

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(RLEstimator, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        image_uri = init_params.pop("image_uri")
        framework, _, tag, _ = fw_utils.framework_name_from_image(image_uri)

        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params["image_uri"] = image_uri
            return init_params

        toolkit, toolkit_version = cls._toolkit_and_version_from_tag(tag)

        if not cls._is_combination_supported(toolkit, toolkit_version, framework):
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    job_details["TrainingJobName"]
                )
            )

        init_params["toolkit"] = RLToolkit(toolkit)
        init_params["toolkit_version"] = toolkit_version
        init_params["framework"] = RLFramework(framework)

        return init_params

    def hyperparameters(self):
        """Return hyperparameters used by your custom TensorFlow code during model training."""
        hyperparameters = super(RLEstimator, self).hyperparameters()

        additional_hyperparameters = {
            SAGEMAKER_OUTPUT_LOCATION: self.output_path,
            # TODO: can be applied to all other estimators
            SAGEMAKER_ESTIMATOR: SAGEMAKER_ESTIMATOR_VALUE,
        }

        hyperparameters.update(
            EstimatorBase._json_encode_hyperparameters(additional_hyperparameters)
        )
        return hyperparameters

    @classmethod
    def _toolkit_and_version_from_tag(cls, image_tag):
        """Placeholder docstring."""
        tag_pattern = re.compile(
            "^([A-Z]*|[a-z]*)(\d.*)-(cpu|gpu)-(py2|py3)$"  # noqa: W605,E501 pylint: disable=anomalous-backslash-in-string
        )
        tag_match = tag_pattern.match(image_tag)
        if tag_match is not None:
            return tag_match.group(1), tag_match.group(2)
        return None, None

    @classmethod
    def _validate_framework_format(cls, framework):
        """Placeholder docstring."""
        if framework and framework not in list(RLFramework):
            raise ValueError(
                "Invalid type: {}, valid RL frameworks types are: {}".format(
                    framework, list(RLFramework)
                )
            )

    @classmethod
    def _validate_toolkit_format(cls, toolkit):
        """Placeholder docstring."""
        if toolkit and toolkit not in list(RLToolkit):
            raise ValueError(
                "Invalid type: {}, valid RL toolkits types are: {}".format(toolkit, list(RLToolkit))
            )

    @classmethod
    def _validate_images_args(cls, toolkit, toolkit_version, framework, image_uri):
        """Placeholder docstring."""
        cls._validate_toolkit_format(toolkit)
        cls._validate_framework_format(framework)

        if not image_uri:
            not_found_args = []
            if not toolkit:
                not_found_args.append("toolkit")
            if not toolkit_version:
                not_found_args.append("toolkit_version")
            if not framework:
                not_found_args.append("framework")
            if not_found_args:
                raise AttributeError(
                    "Please provide `{}` or `image_uri` parameter.".format(
                        "`, `".join(not_found_args)
                    )
                )
        else:
            found_args = []
            if toolkit:
                found_args.append("toolkit")
            if toolkit_version:
                found_args.append("toolkit_version")
            if framework:
                found_args.append("framework")
            if found_args:
                logger.warning(
                    "Parameter `image_uri` is specified, "
                    "`%s` are going to be ignored when choosing the image.",
                    "`, `".join(found_args),
                )

    @classmethod
    def _is_combination_supported(cls, toolkit, toolkit_version, framework):
        """Placeholder docstring."""
        supported_versions = TOOLKIT_FRAMEWORK_VERSION_MAP.get(toolkit, None)
        if supported_versions:
            supported_frameworks = supported_versions.get(toolkit_version, None)
            if supported_frameworks and supported_frameworks.get(framework, None):
                return True
        return False

    @classmethod
    def _validate_toolkit_support(cls, toolkit, toolkit_version, framework):
        """Placeholder docstring."""
        if not cls._is_combination_supported(toolkit, toolkit_version, framework):
            raise AttributeError(
                "Provided `{}-{}` and `{}` combination is not supported.".format(
                    toolkit, toolkit_version, framework
                )
            )

    def _image_framework(self):
        """Toolkit name and framework name for retrieving Docker image URI config."""
        return "-".join((self.toolkit, self.framework))

    @classmethod
    def default_metric_definitions(cls, toolkit):
        """Provides default metric definitions based on provided toolkit.

        Args:
            toolkit (sagemaker.rl.RLToolkit): RL Toolkit to be used for
                training.

        Returns:
            list: metric definitions

        Raises:
            ValueError: If toolkit enum is not valid.
        """
        if toolkit is RLToolkit.COACH:
            return [
                {"Name": "reward-training", "Regex": "^Training>.*Total reward=(.*?),"},
                {"Name": "reward-testing", "Regex": "^Testing>.*Total reward=(.*?),"},
            ]
        if toolkit is RLToolkit.RAY:
            float_regex = "[-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?"  # noqa: W605, E501

            return [
                {"Name": "episode_reward_mean", "Regex": "episode_reward_mean: (%s)" % float_regex},
                {"Name": "episode_reward_max", "Regex": "episode_reward_max: (%s)" % float_regex},
            ]
        raise ValueError("An unknown RLToolkit enum was passed in. toolkit: {}".format(toolkit))
