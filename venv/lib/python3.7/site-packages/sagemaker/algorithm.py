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
"""Test docstring"""
from __future__ import absolute_import

from typing import Optional, Union, Dict, List

import sagemaker
import sagemaker.parameter
from sagemaker import vpc_utils
from sagemaker.deserializers import BytesDeserializer
from sagemaker.deprecations import removed_kwargs
from sagemaker.estimator import EstimatorBase
from sagemaker.inputs import TrainingInput, FileSystemInput
from sagemaker.serializers import IdentitySerializer
from sagemaker.transformer import Transformer
from sagemaker.predictor import Predictor
from sagemaker.session import Session
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.pipeline_context import runnable_by_pipeline

from sagemaker.workflow import is_pipeline_variable


class AlgorithmEstimator(EstimatorBase):
    """A generic Estimator to train using any algorithm object (with an ``algorithm_arn``).

    The Algorithm can be your own, or any Algorithm from AWS
    Marketplace that you have a valid subscription for. This class will perform
    client-side validation on all the inputs.
    """

    # These Hyperparameter Types have a range definition.
    _hyperpameters_with_range = ("Integer", "Continuous", "Categorical")

    def __init__(
        self,
        algorithm_arn: str,
        role: str = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        volume_size: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        max_run: Union[int, PipelineVariable] = 24 * 60 * 60,
        input_mode: Union[str, PipelineVariable] = "File",
        output_path: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        subnets: Optional[List[Union[str, PipelineVariable]]] = None,
        security_group_ids: Optional[List[Union[str, PipelineVariable]]] = None,
        model_uri: Optional[str] = None,
        model_channel_name: Union[str, PipelineVariable] = "model",
        metric_definitions: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        encrypt_inter_container_traffic: Union[bool, PipelineVariable] = False,
        use_spot_instances: Union[bool, PipelineVariable] = False,
        max_wait: Optional[Union[int, PipelineVariable]] = None,
        **kwargs  # pylint: disable=W0613
    ):
        """Initialize an ``AlgorithmEstimator`` instance.

        Args:
            algorithm_arn (str): algorithm arn used for training. Can be just the name if your
                account owns the algorithm.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker
                training jobs and APIsthat create Amazon SageMaker endpoints use this role to
                access training data and model artifacts. After the endpoint
                is created, the inference code might use the IAM role, if it
                needs to access an AWS resource.
            instance_count (int or PipelineVariable): Number of Amazon EC2 instances to use
                for training.
            instance_type (str or PipelineVariable): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            volume_size (int or PipelineVariable): Size in GB of the EBS volume to use for
                storing input data during training (default: 30). Must be large enough to store
                training data if File Mode is used (which is the default).
            volume_kms_key (str or PipelineVariable): Optional. KMS key ID for encrypting
                EBS volume attached to the training instance (default: None).
            max_run (int or PipelineVariable): Timeout in seconds for training
                (default: 24 * 60 * 60).
                After this amount of time Amazon SageMaker terminates the
                job regardless of its current status.
            input_mode (str or PipelineVariable): The input mode that the algorithm supports
                (default: 'File'). Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from
                  the S3 location to a local directory.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to
                  the container via a Unix-named pipe.

                This argument can be overriden on a per-channel basis using
                ``sagemaker.inputs.TrainingInput.input_mode``.

            output_path (str or PipelineVariable): S3 location for saving the training result
                (model artifacts and output files). If not specified,
                results are stored to a default bucket. If
                the bucket with the specific name does not exist, the
                estimator creates the bucket during the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method
                execution.
            output_kms_key (str or PipelineVariable): Optional. KMS key ID for encrypting the
                training output (default: None). base_job_name (str): Prefix for
                training job name when the
                :meth:`~sagemaker.estimator.EstimatorBase.fit`
                method launches. If not specified, the estimator generates a
                default job name, based on the training image name and
                current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which manages
                interactions with Amazon SageMaker APIs and any other AWS services needed. If
                not specified, the estimator creates one using the default
                AWS configuration chain.
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): List of tags for
                labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            subnets (list[str] or list[PipelineVariable]): List of subnet ids. If not specified
                training job will be created without VPC config.
                security_group_ids (list[str]): List of security group ids. If
                not specified training job will be created without VPC config.
            model_uri (str): URI where a pre-trained model is stored, either locally or in S3
                (default: None). If specified, the estimator will create a channel pointing to
                the model so the training job can download it. This model
                can be a 'model.tar.gz' from a previous training job, or
                other artifacts coming from a different source.
                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
            model_channel_name (str or PipelineVariable): Name of the channel where 'model_uri'
                will be downloaded (default: 'model'). metric_definitions
                (list[dict]): A list of dictionaries that defines the metric(s)
                used to evaluate the training jobs. Each dictionary contains two keys: 'Name' for
                the name of the metric, and 'Regex' for the regular
                expression used to extract the metric from the logs.
            encrypt_inter_container_traffic (bool or PipelineVariable): Specifies whether traffic
                between training containers is encrypted for the training job (default: ``False``).
            use_spot_instances (bool or PipelineVariable): Specifies whether to use SageMaker
                Managed Spot instances for training. If enabled then the
                `max_wait` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (default: ``False``).
            max_wait (int or PipelineVariable): Timeout in seconds waiting for spot training
                instances (default: None). After this amount of time Amazon
                SageMaker will stop waiting for Spot instances to become
                available (default: ``None``).
            **kwargs: Additional kwargs. This is unused. It's only added for AlgorithmEstimator
                to ignore the irrelevant arguments.
        """
        self.algorithm_arn = algorithm_arn
        super(AlgorithmEstimator, self).__init__(
            role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size=volume_size,
            volume_kms_key=volume_kms_key,
            max_run=max_run,
            input_mode=input_mode,
            output_path=output_path,
            output_kms_key=output_kms_key,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            tags=tags,
            subnets=subnets,
            security_group_ids=security_group_ids,
            model_uri=model_uri,
            model_channel_name=model_channel_name,
            metric_definitions=metric_definitions,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait,
        )

        self.algorithm_spec = self.sagemaker_session.sagemaker_client.describe_algorithm(
            AlgorithmName=algorithm_arn
        )
        self.validate_train_spec()
        self.hyperparameter_definitions = self._parse_hyperparameters()

        self._hyperparameters = {}
        if hyperparameters:
            self.set_hyperparameters(**hyperparameters)

    def validate_train_spec(self):
        """Placeholder docstring"""
        train_spec = self.algorithm_spec["TrainingSpecification"]
        algorithm_name = self.algorithm_spec["AlgorithmName"]

        # Check that the input mode provided is compatible with the training input modes for the
        # algorithm.
        input_modes = self._algorithm_training_input_modes(train_spec["TrainingChannels"])
        if not is_pipeline_variable(self.input_mode) and self.input_mode not in input_modes:
            raise ValueError(
                "Invalid input mode: %s. %s only supports: %s"
                % (self.input_mode, algorithm_name, input_modes)
            )

        # Check that the training instance type is compatible with the algorithm.
        supported_instances = train_spec["SupportedTrainingInstanceTypes"]
        if (
            not is_pipeline_variable(self.instance_type)
            and self.instance_type not in supported_instances
        ):
            raise ValueError(
                "Invalid instance_type: %s. %s supports the following instance types: %s"
                % (self.instance_type, algorithm_name, supported_instances)
            )

        # Verify if distributed training is supported by the algorithm
        if not is_pipeline_variable(self.instance_count) and (
            self.instance_count > 1
            and "SupportsDistributedTraining" in train_spec
            and not train_spec["SupportsDistributedTraining"]
        ):
            raise ValueError(
                "Distributed training is not supported by %s. "
                "Please set instance_count=1" % algorithm_name
            )

    def set_hyperparameters(self, **kwargs):
        """Placeholder docstring"""
        for k, v in kwargs.items():
            value = self._validate_and_cast_hyperparameter(k, v)
            self._hyperparameters[k] = value

        self._validate_and_set_default_hyperparameters()

    def hyperparameters(self):
        """Returns the hyperparameters as a dictionary to use for training.

        The fit() method, that does the model training, calls this method to
        find the hyperparameters you specified.
        """
        return self._hyperparameters

    def training_image_uri(self):
        """Returns the docker image to use for training.

        The fit() method, that does the model training, calls this method to
        find the image to use for model training.
        """
        raise RuntimeError("training_image_uri is never meant to be called on Algorithm Estimators")

    def enable_network_isolation(self):
        """Return True if this Estimator will need network isolation to run.

        On Algorithm Estimators this depends on the algorithm being used. If
        this is algorithm owned by your account it will be False. If this is an
        an algorithm consumed from Marketplace it will be True.

        Returns:
            bool: Whether this Estimator needs network isolation or not.
        """
        return self._is_marketplace()

    def create_model(
        self,
        role=None,
        predictor_cls=None,
        serializer=IdentitySerializer(),
        deserializer=BytesDeserializer(),
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        **kwargs
    ):
        """Create a model to deploy.

        The serializer and deserializer are only used to define a default
        Predictor. They are ignored if an explicit predictor class is passed in.
        Other arguments are passed through to the Model class.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            predictor_cls (Predictor): The predictor class to use when
                deploying the model.
            serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
                serializer object, used to encode data for an inference endpoint
                (default: :class:`~sagemaker.serializers.IdentitySerializer`).
            deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
                deserializer object, used to decode data from an inference
                endpoint (default: :class:`~sagemaker.deserializers.BytesDeserializer`).
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional arguments for creating a :class:`~sagemaker.model.ModelPackage`.

        .. tip::

            You can find additional parameters for using this method at
            :class:`~sagemaker.model.ModelPackage` and
            :class:`~sagemaker.model.Model`.

        Returns:
            a Model ready for deployment.
        """
        removed_kwargs("content_type", kwargs)
        removed_kwargs("accept", kwargs)

        if predictor_cls is None:

            def predict_wrapper(endpoint, session):
                return Predictor(endpoint, session, serializer, deserializer)

            predictor_cls = predict_wrapper

        role = role or self.role

        return sagemaker.ModelPackage(
            role,
            algorithm_arn=self.algorithm_arn,
            model_data=self.model_data,
            vpc_config=self.get_vpc_config(vpc_config_override),
            sagemaker_session=self.sagemaker_session,
            predictor_cls=predictor_cls,
            **kwargs
        )

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
    ):
        """Return a ``Transformer`` that uses a SageMaker Model based on the  training job.

        It reuses the SageMaker Session and base job name used by the Estimator.

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
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            volume_kms_key (str): Optional. KMS key ID for encrypting the volume
                attached to the ML compute instance (default: None).
        """
        role = role or self.role

        if self.latest_training_job is not None:
            model = self.create_model(role=role)
            model._create_sagemaker_model()
            model_name = model.name
            transform_env = {}
            if env is not None:
                transform_env = model.env.copy()
                transform_env.update(env)
            if self._is_marketplace():
                transform_env = None

            tags = tags or self.tags
        else:
            raise RuntimeError("No finished training job found associated with this estimator")

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
            env=transform_env,
            tags=tags,
            base_transform_job_name=self.base_job_name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=self.sagemaker_session,
        )

    def _is_marketplace(self):
        """Placeholder docstring"""
        return "ProductId" in self.algorithm_spec

    def _ensure_base_job_name(self):
        """Set ``self.base_job_name`` if it is not set already."""
        if self.base_job_name is None:
            self.base_job_name = self.algorithm_arn.split("/")[-1]

    def _prepare_for_training(self, job_name=None):
        # Validate hyperparameters
        # an explicit call to set_hyperparameters() will also validate the hyperparameters
        # but it is possible that the user never called it.
        self._validate_and_set_default_hyperparameters()

        super(AlgorithmEstimator, self)._prepare_for_training(job_name)

    @runnable_by_pipeline
    def fit(
        self,
        inputs: Optional[Union[str, Dict, TrainingInput, FileSystemInput]] = None,
        wait: bool = True,
        logs: bool = True,
        job_name: Optional[str] = None,
    ):
        """Placeholder docstring"""
        if inputs:
            self._validate_input_channels(inputs)

        return super(AlgorithmEstimator, self).fit(inputs, wait, logs, job_name)

    def _validate_input_channels(self, channels):
        """Placeholder docstring"""
        train_spec = self.algorithm_spec["TrainingSpecification"]
        algorithm_name = self.algorithm_spec["AlgorithmName"]
        training_channels = {c["Name"]: c for c in train_spec["TrainingChannels"]}

        # check for unknown channels that the algorithm does not support
        for c in channels:
            if c not in training_channels:
                raise ValueError(
                    "Unknown input channel: %s is not supported by: %s" % (c, algorithm_name)
                )

        # check for required channels that were not provided
        for name, channel in training_channels.items():
            if name not in channels and "IsRequired" in channel and channel["IsRequired"]:
                raise ValueError("Required input channel: %s Was not provided." % (name))

    def _validate_and_cast_hyperparameter(self, name, v):
        """Placeholder docstring"""
        algorithm_name = self.algorithm_spec["AlgorithmName"]

        if name not in self.hyperparameter_definitions:
            raise ValueError(
                "Invalid hyperparameter: %s is not supported by %s" % (name, algorithm_name)
            )

        definition = self.hyperparameter_definitions[name]
        if "class" in definition:
            value = definition["class"].cast_to_type(v)
        else:
            value = v

        if "range" in definition and not definition["range"].is_valid(value):
            valid_range = definition["range"].as_tuning_range(name)
            raise ValueError("Invalid value: %s Supported range: %s" % (value, valid_range))
        return value

    def _validate_and_set_default_hyperparameters(self):
        """Placeholder docstring"""
        # Check if all the required hyperparameters are set. If there is a default value
        # for one, set it.
        for name, definition in self.hyperparameter_definitions.items():
            if name not in self._hyperparameters:
                spec = definition["spec"]
                if "DefaultValue" in spec:
                    self._hyperparameters[name] = spec["DefaultValue"]
                elif "IsRequired" in spec and spec["IsRequired"]:
                    raise ValueError("Required hyperparameter: %s is not set" % name)

    def _parse_hyperparameters(self):
        """Placeholder docstring"""
        definitions = {}

        training_spec = self.algorithm_spec["TrainingSpecification"]
        if "SupportedHyperParameters" in training_spec:
            hyperparameters = training_spec["SupportedHyperParameters"]
            for h in hyperparameters:
                parameter_type = h["Type"]
                name = h["Name"]
                parameter_class, parameter_range = self._hyperparameter_range_and_class(
                    parameter_type, h
                )

                definitions[name] = {"spec": h}
                if parameter_range:
                    definitions[name]["range"] = parameter_range
                if parameter_class:
                    definitions[name]["class"] = parameter_class

        return definitions

    def _hyperparameter_range_and_class(self, parameter_type, hyperparameter):
        """Placeholder docstring."""
        if parameter_type in self._hyperpameters_with_range:
            range_name = parameter_type + "ParameterRangeSpecification"

        parameter_class = None
        parameter_range = None

        if parameter_type in ("Integer", "Continuous"):
            # Integer and Continuous are handled the same way. We get the min and max values
            # and just create an Instance of Parameter. Note that the range is optional for all
            # the Parameter Types.
            if parameter_type == "Integer":
                parameter_class = sagemaker.parameter.IntegerParameter
            else:
                parameter_class = sagemaker.parameter.ContinuousParameter

            if "Range" in hyperparameter:
                min_value = parameter_class.cast_to_type(
                    hyperparameter["Range"][range_name]["MinValue"]
                )
                max_value = parameter_class.cast_to_type(
                    hyperparameter["Range"][range_name]["MaxValue"]
                )
                parameter_range = parameter_class(min_value, max_value)

        elif parameter_type == "Categorical":
            parameter_class = sagemaker.parameter.CategoricalParameter
            if "Range" in hyperparameter:
                values = hyperparameter["Range"][range_name]["Values"]
                parameter_range = sagemaker.parameter.CategoricalParameter(values)
        elif parameter_type == "FreeText":
            pass
        else:
            raise ValueError(
                "Invalid Hyperparameter type: %s. Valid ones are:"
                "(Integer, Continuous, Categorical, FreeText)" % parameter_type
            )

        return parameter_class, parameter_range

    def _algorithm_training_input_modes(self, training_channels):
        """Placeholder docstring"""
        current_input_modes = {"File", "Pipe"}
        for channel in training_channels:
            supported_input_modes = set(channel["SupportedInputModes"])
            current_input_modes = current_input_modes & supported_input_modes

        return current_input_modes

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details (dict): the returned job details from a DescribeTrainingJob
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dict: The transformed init_params
        """
        init_params = super(AlgorithmEstimator, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        # This hyperparameter is added by Amazon SageMaker Automatic Model Tuning.
        # It cannot be set through instantiating an estimator.
        if "_tuning_objective_metric" in init_params["hyperparameters"]:
            del init_params["hyperparameters"]["_tuning_objective_metric"]

        return init_params
