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
"""Amazon SageMaker Debugger provides full visibility into ML training jobs.

This module provides SageMaker Debugger high-level methods
to set up Debugger objects, such as Debugger built-in rules, tensor collections,
and hook configuration. Use the Debugger objects for parameters when constructing
a SageMaker estimator to initiate a training job.

"""
from __future__ import absolute_import

from abc import ABC

from typing import Union, Optional, List, Dict

import attr

import smdebug_rulesconfig as rule_configs

from sagemaker import image_uris
from sagemaker.utils import build_dict, name_from_base
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.debugger.profiler_constants import (
    DETAIL_PROF_PROCESSING_DEFAULT_INSTANCE_TYPE,
    DETAIL_PROF_PROCESSING_DEFAULT_VOLUME_SIZE,
)

framework_name = "debugger"
detailed_framework_name = "detailed-profiler"
DEBUGGER_FLAG = "USE_SMDEBUG"


class DetailedProfilerProcessingJobConfig:
    """ProfilerRule like class.

    Serves as a vehicle to pass info through to the processing instance.

    """

    def __init__(self):
        self.rule_name = self.__class__.__name__
        self.rule_parameters = {"rule_to_invoke": "DetailedProfilerProcessing"}


def get_rule_container_image_uri(name, region):
    """Return the Debugger rule image URI for the given AWS Region.

    For a full list of rule image URIs,
    see `Use Debugger Docker Images for Built-in or Custom Rules
    <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-docker-images-rules.html>`_.

    Args:
        region (str): A string of AWS Region. For example, ``'us-east-1'``.

    Returns:
        str: Formatted image URI for the given AWS Region and the rule container type.

    """
    if name is not None and name.startswith("DetailedProfilerProcessingJobConfig"):
        # should have the format like "123456789012.dkr.ecr.us-west-2.amazonaws.com/detailed-profiler-processing:latest"
        return image_uris.retrieve(detailed_framework_name, region)

    return image_uris.retrieve(framework_name, region)


def get_default_profiler_processing_job(instance_type=None, volume_size_in_gb=None):
    """Return the default profiler processing job (a rule) with a unique name.

    Returns:
        sagemaker.debugger.ProfilerRule: The instance of the built-in ProfilerRule.

    """
    default_rule = DetailedProfilerProcessingJobConfig()
    custom_name = name_from_base(default_rule.rule_name)
    return ProfilerRule.sagemaker(
        default_rule,
        name=custom_name,
        instance_type=instance_type,
        volume_size_in_gb=volume_size_in_gb,
    )


@attr.s
class RuleBase(ABC):
    """The SageMaker Debugger rule base class that cannot be instantiated directly.

    .. tip::

        Debugger rule classes inheriting this RuleBase class are
        :class:`~sagemaker.debugger.Rule` and :class:`~sagemaker.debugger.ProfilerRule`.
        Do not directly use the rule base class to instantiate a SageMaker Debugger rule.
        Use the :class:`~sagemaker.debugger.Rule` classmethods for debugging
        and the :class:`~sagemaker.debugger.ProfilerRule` classmethods for profiling.

    Attributes:
        name (str): The name of the rule.
        image_uri (str): The image URI to use the rule.
        instance_type (str): Type of EC2 instance to use. For example, 'ml.c4.xlarge'.
        container_local_output_path (str): The local path to store the Rule output.
        s3_output_path (str): The location in S3 to store the output.
        volume_size_in_gb (int): Size in GB of the EBS volume to use for storing data.
        rule_parameters (dict): A dictionary of parameters for the rule.

    """

    name = attr.ib()
    image_uri = attr.ib()
    instance_type = attr.ib()
    container_local_output_path = attr.ib()
    s3_output_path = attr.ib()
    volume_size_in_gb = attr.ib()
    rule_parameters = attr.ib()

    @staticmethod
    def _set_rule_parameters(source, rule_to_invoke, rule_parameters):
        """Create a dictionary of rule parameters.

        Args:
            source (str): Optional. A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str): Optional. The name of the rule to invoke within the source.
                If provided, you must also provide source.
            rule_parameters (dict): Optional. A dictionary of parameters for the rule.

        Returns:
            dict: A dictionary of rule parameters.

        """
        if bool(source) ^ bool(rule_to_invoke):
            raise ValueError(
                "If you provide a source, you must also provide a rule to invoke (and vice versa)."
            )

        merged_rule_params = {}
        merged_rule_params.update(build_dict("source_s3_uri", source))
        merged_rule_params.update(build_dict("rule_to_invoke", rule_to_invoke))
        merged_rule_params.update(rule_parameters or {})

        return merged_rule_params


class Rule(RuleBase):
    """The SageMaker Debugger Rule class configures *debugging* rules to debug your training job.

    The debugging rules analyze tensor outputs from your training job
    and monitor conditions that are critical for the success of the training
    job.

    SageMaker Debugger comes pre-packaged with built-in *debugging* rules.
    For example, the debugging rules can detect whether gradients are getting too large or
    too small, or if a model is overfitting.
    For a full list of built-in rules for debugging, see
    `List of Debugger Built-in Rules
    <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.
    You can also write your own rules using the custom rule classmethod.

    """

    def __init__(
        self,
        name,
        image_uri,
        instance_type,
        container_local_output_path,
        s3_output_path,
        volume_size_in_gb,
        rule_parameters,
        collections_to_save,
        actions=None,
    ):
        """Configure the debugging rules using the following classmethods.

        .. tip::
            Use the following ``Rule.sagemaker`` class method for built-in debugging rules
            or the ``Rule.custom`` class method for custom debugging rules.
            Do not directly use the :class:`~sagemaker.debugger.Rule`
            initialization method.

        """
        super(Rule, self).__init__(
            name,
            image_uri,
            instance_type,
            container_local_output_path,
            s3_output_path,
            volume_size_in_gb,
            rule_parameters,
        )
        self.collection_configs = collections_to_save
        self.actions = actions

    @classmethod
    def sagemaker(
        cls,
        base_config,
        name=None,
        container_local_output_path=None,
        s3_output_path=None,
        other_trials_s3_input_paths=None,
        rule_parameters=None,
        collections_to_save=None,
        actions=None,
    ):
        """Initialize a ``Rule`` object for a *built-in* debugging rule.

        Args:
            base_config (dict): Required. This is the base rule config dictionary returned from the
                :class:`~sagemaker.debugger.rule_configs` method.
                For example, ``rule_configs.dead_relu()``.
                For a full list of built-in rules for debugging, see
                `List of Debugger Built-in Rules
                <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.
            name (str): Optional. The name of the debugger rule. If one is not provided,
                the name of the base_config will be used.
            container_local_output_path (str): Optional. The local path in the rule processing
                container.
            s3_output_path (str): Optional. The location in Amazon S3 to store the output tensors.
                The default Debugger output path for debugging data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/debug-output/.
            other_trials_s3_input_paths ([str]): Optional. The Amazon S3 input paths
                of other trials to use the SimilarAcrossRuns rule.
            rule_parameters (dict): Optional. A dictionary of parameters for the rule.
            collections_to_save (:class:`~sagemaker.debugger.CollectionConfig`):
                Optional. A list
                of :class:`~sagemaker.debugger.CollectionConfig` objects to be saved.

        Returns:
            :class:`~sagemaker.debugger.Rule`: An instance of the built-in rule.

        **Example of how to create a built-in rule instance:**

        .. code-block:: python

            from sagemaker.debugger import Rule, rule_configs

            built_in_rules = [
                Rule.sagemaker(rule_configs.built_in_rule_name_in_pysdk_format_1()),
                Rule.sagemaker(rule_configs.built_in_rule_name_in_pysdk_format_2()),
                ...
                Rule.sagemaker(rule_configs.built_in_rule_name_in_pysdk_format_n())
            ]

        You need to replace the ``built_in_rule_name_in_pysdk_format_*`` with the
        names of built-in rules. You can find the rule names at `List of Debugger Built-in
        Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.

        **Example of creating a built-in rule instance with adjusting parameter values:**

        .. code-block:: python

            from sagemaker.debugger import Rule, rule_configs

            built_in_rules = [
                Rule.sagemaker(
                    base_config=rule_configs.built_in_rule_name_in_pysdk_format(),
                    rule_parameters={
                            "key": "value"
                    }
                    collections_to_save=[
                        CollectionConfig(
                            name="tensor_collection_name",
                            parameters={
                                "key": "value"
                            }
                        )
                    ]
                )
            ]

        For more information about setting up the ``rule_parameters`` parameter,
        see `List of Debugger Built-in
        Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.

        For more information about setting up the ``collections_to_save`` parameter,
        see the :class:`~sagemaker.debugger.CollectionConfig` class.

        """
        merged_rule_params = {}

        if rule_parameters is not None and rule_parameters.get("rule_to_invoke") is not None:
            raise RuntimeError(
                """You cannot provide a 'rule_to_invoke' for SageMaker rules.
                Either remove the rule_to_invoke or use a custom rule.

                """
            )

        if actions is not None and not rule_configs.is_valid_action_object(actions):
            raise RuntimeError("""`actions` must be of type `Action` or `ActionList`!""")

        if other_trials_s3_input_paths is not None:
            for index, s3_input_path in enumerate(other_trials_s3_input_paths):
                merged_rule_params["other_trial_{}".format(str(index))] = s3_input_path

        default_rule_params = base_config["DebugRuleConfiguration"].get("RuleParameters", {})
        merged_rule_params.update(default_rule_params)
        merged_rule_params.update(rule_parameters or {})

        base_config_collections = []
        for config in base_config.get("CollectionConfigurations", []):
            collection_name = None
            collection_parameters = {}
            for key, value in config.items():
                if key == "CollectionName":
                    collection_name = value
                if key == "CollectionParameters":
                    collection_parameters = value
            base_config_collections.append(
                CollectionConfig(name=collection_name, parameters=collection_parameters)
            )

        return cls(
            name=name or base_config["DebugRuleConfiguration"].get("RuleConfigurationName"),
            image_uri="DEFAULT_RULE_EVALUATOR_IMAGE",
            instance_type=None,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=None,
            rule_parameters=merged_rule_params,
            collections_to_save=collections_to_save or base_config_collections,
            actions=actions,
        )

    @classmethod
    def custom(
        cls,
        name: str,
        image_uri: Union[str, PipelineVariable],
        instance_type: Union[str, PipelineVariable],
        volume_size_in_gb: Union[int, PipelineVariable],
        source: Optional[str] = None,
        rule_to_invoke: Optional[Union[str, PipelineVariable]] = None,
        container_local_output_path: Optional[Union[str, PipelineVariable]] = None,
        s3_output_path: Optional[Union[str, PipelineVariable]] = None,
        other_trials_s3_input_paths: Optional[List[Union[str, PipelineVariable]]] = None,
        rule_parameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        collections_to_save: Optional[List["CollectionConfig"]] = None,
        actions=None,
    ):
        """Initialize a ``Rule`` object for a *custom* debugging rule.

        You can create a custom rule that analyzes tensors emitted
        during the training of a model
        and monitors conditions that are critical for the success of a training
        job. For more information, see `Create Debugger Custom Rules for Training Job
        Analysis
        <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-custom-rules.html>`_.

        Args:
            name (str): Required. The name of the debugger rule.
            image_uri (str or PipelineVariable): Required. The URI of the image to
                be used by the debugger rule.
            instance_type (str or PipelineVariable): Required. Type of EC2 instance to use,
                for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int or PipelineVariable): Required. Size in GB of the
                EBS volume to use for storing data.
            source (str): Optional. A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str or PipelineVariable): Optional. The name of the rule to
                invoke within the source. If provided, you must also provide source.
            container_local_output_path (str or PipelineVariable): Optional. The local path
                in the container.
            s3_output_path (str or PipelineVariable): Optional. The location in Amazon S3
                to store the output tensors.
                The default Debugger output path for debugging data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/debug-output/.
            other_trials_s3_input_paths (list[str] or list[PipelineVariable]: Optional.
                The Amazon S3 input paths of other trials to use the SimilarAcrossRuns rule.
            rule_parameters (dict[str, str] or dict[str, PipelineVariable]): Optional.
                A dictionary of parameters for the rule.
            collections_to_save ([sagemaker.debugger.CollectionConfig]): Optional. A list
                of :class:`~sagemaker.debugger.CollectionConfig` objects to be saved.

        Returns:
            :class:`~sagemaker.debugger.Rule`: The instance of the custom rule.

        """
        if actions is not None and not rule_configs.is_valid_action_object(actions):
            raise RuntimeError("""`actions` must be of type `Action` or `ActionList`!""")

        merged_rule_params = cls._set_rule_parameters(
            source, rule_to_invoke, other_trials_s3_input_paths, rule_parameters
        )

        return cls(
            name=name,
            image_uri=image_uri,
            instance_type=instance_type,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=volume_size_in_gb,
            rule_parameters=merged_rule_params,
            collections_to_save=collections_to_save or [],
            actions=actions,
        )

    def prepare_actions(self, training_job_name):
        """Prepare actions for Debugger Rule.

        Args:
            training_job_name (str): The training job name. To be set as the default training job
                prefix for the StopTraining action if it is specified.
        """
        if self.actions is None:
            # user cannot manually specify action_json in rule_parameters for actions.
            self.rule_parameters.pop("action_json", None)
            return

        self.actions.update_training_job_prefix_if_not_specified(training_job_name)
        action_params = {"action_json": self.actions.serialize()}
        self.rule_parameters.update(action_params)

    @staticmethod
    def _set_rule_parameters(source, rule_to_invoke, other_trials_s3_input_paths, rule_parameters):
        """Set rule parameters for Debugger Rule.

        Args:
            source (str): Optional. A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str): Optional. The name of the rule to invoke within the source.
                If provided, you must also provide source.
            other_trials_s3_input_paths ([str]): Optional. S3 input paths for other trials.
            rule_parameters (dict): Optional. A dictionary of parameters for the rule.

        Returns:
            dict: A dictionary of rule parameters.

        """
        merged_rule_params = {}
        if other_trials_s3_input_paths is not None:
            for index, s3_input_path in enumerate(other_trials_s3_input_paths):
                merged_rule_params["other_trial_{}".format(str(index))] = s3_input_path

        merged_rule_params.update(
            super(Rule, Rule)._set_rule_parameters(source, rule_to_invoke, rule_parameters)
        )
        return merged_rule_params

    def to_debugger_rule_config_dict(self):
        """Generates a request dictionary using the parameters provided when initializing object.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        debugger_rule_config_request = {
            "RuleConfigurationName": self.name,
            "RuleEvaluatorImage": self.image_uri,
        }

        debugger_rule_config_request.update(build_dict("InstanceType", self.instance_type))
        debugger_rule_config_request.update(build_dict("VolumeSizeInGB", self.volume_size_in_gb))
        debugger_rule_config_request.update(
            build_dict("LocalPath", self.container_local_output_path)
        )
        debugger_rule_config_request.update(build_dict("S3OutputPath", self.s3_output_path))
        debugger_rule_config_request.update(build_dict("RuleParameters", self.rule_parameters))

        return debugger_rule_config_request


class ProfilerRule(RuleBase):
    """The SageMaker Debugger ProfilerRule class configures *profiling* rules.

    SageMaker Debugger profiling rules automatically analyze
    hardware system resource utilization and framework metrics of a
    training job to identify performance bottlenecks.

    SageMaker Debugger comes pre-packaged with built-in *profiling* rules.
    For example, the profiling rules can detect if GPUs are underutilized due to CPU bottlenecks or
    IO bottlenecks.
    For a full list of built-in rules for debugging, see
    `List of Debugger Built-in Rules <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.
    You can also write your own profiling rules using the Amazon SageMaker
    Debugger APIs.

    .. tip::
        Use the following ``ProfilerRule.sagemaker`` class method for built-in profiling rules
        or the ``ProfilerRule.custom`` class method for custom profiling rules.
        Do not directly use the `Rule` initialization method.

    """

    @classmethod
    def sagemaker(
        cls,
        base_config,
        name=None,
        container_local_output_path=None,
        s3_output_path=None,
        instance_type=None,
        volume_size_in_gb=None,
    ):
        """Initialize a ``ProfilerRule`` object for a *built-in* profiling rule.

        The rule analyzes system and framework metrics of a given
        training job to identify performance bottlenecks.

        Args:
            base_config (rule_configs.ProfilerRule): The base rule configuration object
                returned from the ``rule_configs`` method.
                For example, 'rule_configs.ProfilerReport()'.
                For a full list of built-in rules for debugging, see
                `List of Debugger Built-in Rules
                <https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html>`_.

            name (str): The name of the profiler rule. If one is not provided,
                the name of the base_config will be used.
            container_local_output_path (str): The path in the container.
            s3_output_path (str): The location in Amazon S3 to store the profiling output data.
                The default Debugger output path for profiling data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/profiler-output/.

        Returns:
            :class:`~sagemaker.debugger.ProfilerRule`:
            The instance of the built-in ProfilerRule.

        """
        used_name = name or base_config.rule_name
        if used_name.startswith("DetailedProfilerProcessingJobConfig"):
            if volume_size_in_gb is None:
                volume_size_in_gb = DETAIL_PROF_PROCESSING_DEFAULT_VOLUME_SIZE
            if instance_type is None:
                instance_type = DETAIL_PROF_PROCESSING_DEFAULT_INSTANCE_TYPE
        return cls(
            name=used_name,
            image_uri="DEFAULT_RULE_EVALUATOR_IMAGE",
            instance_type=instance_type,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=volume_size_in_gb,
            rule_parameters=base_config.rule_parameters,
        )

    @classmethod
    def custom(
        cls,
        name,
        image_uri,
        instance_type,
        volume_size_in_gb,
        source=None,
        rule_to_invoke=None,
        container_local_output_path=None,
        s3_output_path=None,
        rule_parameters=None,
    ):
        """Initialize a ``ProfilerRule`` object for a *custom* profiling rule.

        You can create a rule that
        analyzes system and framework metrics emitted during the training of a model and
        monitors conditions that are critical for the success of a
        training job.

        Args:
            name (str): The name of the profiler rule.
            image_uri (str): The URI of the image to be used by the proflier rule.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data.
            source (str): A source file containing a rule to invoke. If provided,
                you must also provide rule_to_invoke. This can either be an S3 uri or
                a local path.
            rule_to_invoke (str): The name of the rule to invoke within the source.
                If provided, you must also provide the source.
            container_local_output_path (str): The path in the container.
            s3_output_path (str): The location in Amazon S3 to store the output.
                The default Debugger output path for profiling data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/profiler-output/.
            rule_parameters (dict): A dictionary of parameters for the rule.

        Returns:
            :class:`~sagemaker.debugger.ProfilerRule`:
            The instance of the custom ProfilerRule.

        """
        merged_rule_params = super()._set_rule_parameters(source, rule_to_invoke, rule_parameters)

        return cls(
            name=name,
            image_uri=image_uri,
            instance_type=instance_type,
            container_local_output_path=container_local_output_path,
            s3_output_path=s3_output_path,
            volume_size_in_gb=volume_size_in_gb,
            rule_parameters=merged_rule_params,
        )

    def to_profiler_rule_config_dict(self):
        """Generates a request dictionary using the parameters provided when initializing object.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        profiler_rule_config_request = {
            "RuleConfigurationName": self.name,
            "RuleEvaluatorImage": self.image_uri,
        }

        profiler_rule_config_request.update(build_dict("InstanceType", self.instance_type))
        profiler_rule_config_request.update(build_dict("VolumeSizeInGB", self.volume_size_in_gb))
        profiler_rule_config_request.update(
            build_dict("LocalPath", self.container_local_output_path)
        )
        profiler_rule_config_request.update(build_dict("S3OutputPath", self.s3_output_path))

        if self.rule_parameters:
            profiler_rule_config_request["RuleParameters"] = self.rule_parameters
            for k, v in profiler_rule_config_request["RuleParameters"].items():
                profiler_rule_config_request["RuleParameters"][k] = str(v)

        return profiler_rule_config_request


class DebuggerHookConfig(object):
    """Create a Debugger hook configuration object to save the tensor for debugging.

    DebuggerHookConfig provides options to customize how debugging
    information is emitted and saved. This high-level DebuggerHookConfig class
    runs based on the `smdebug.SaveConfig
    <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/
    api.md#saveconfig>`_ class.

    """

    def __init__(
        self,
        s3_output_path: Optional[Union[str, PipelineVariable]] = None,
        container_local_output_path: Optional[Union[str, PipelineVariable]] = None,
        hook_parameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        collection_configs: Optional[List["CollectionConfig"]] = None,
    ):
        """Initialize the DebuggerHookConfig instance.

        Args:
            s3_output_path (str or PipelineVariable): Optional. The location in Amazon S3 to
                store the output tensors. The default Debugger output path is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/debug-output/.
            container_local_output_path (str or PipelineVariable): Optional. The local path
                in the container.
            hook_parameters (dict[str, str] or dict[str, PipelineVariable]): Optional.
                A dictionary of parameters.
            collection_configs ([sagemaker.debugger.CollectionConfig]): Required. A list
                of :class:`~sagemaker.debugger.CollectionConfig` objects to be saved
                at the **s3_output_path**.

        **Example of creating a DebuggerHookConfig object:**

        .. code-block:: python

            from sagemaker.debugger import CollectionConfig, DebuggerHookConfig

            collection_configs=[
                CollectionConfig(name="tensor_collection_1")
                CollectionConfig(name="tensor_collection_2")
                ...
                CollectionConfig(name="tensor_collection_n")
            ]

            hook_config = DebuggerHookConfig(
                collection_configs=collection_configs
            )

        """
        self.s3_output_path = s3_output_path
        self.container_local_output_path = container_local_output_path
        self.hook_parameters = hook_parameters
        self.collection_configs = collection_configs

    def _to_request_dict(self):
        """Generate a request dictionary using the parameters when initializing the object.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        debugger_hook_config_request = {"S3OutputPath": self.s3_output_path}

        if self.container_local_output_path is not None:
            debugger_hook_config_request["LocalPath"] = self.container_local_output_path

        if self.hook_parameters is not None:
            debugger_hook_config_request["HookParameters"] = self.hook_parameters

        if self.collection_configs is not None:
            debugger_hook_config_request["CollectionConfigurations"] = [
                collection_config._to_request_dict()
                for collection_config in self.collection_configs
            ]

        return debugger_hook_config_request


class TensorBoardOutputConfig(object):
    """Create a tensor ouput configuration object for debugging visualizations on TensorBoard."""

    def __init__(
        self,
        s3_output_path: Union[str, PipelineVariable],
        container_local_output_path: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Initialize the TensorBoardOutputConfig instance.

        Args:
            s3_output_path (str or PipelineVariable): Optional. The location in Amazon S3
                to store the output.
            container_local_output_path (str or PipelineVariable): Optional. The local path
                in the container.

        """
        self.s3_output_path = s3_output_path
        self.container_local_output_path = container_local_output_path

    def _to_request_dict(self):
        """Generate a request dictionary using the instances attributes.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        tensorboard_output_config_request = {"S3OutputPath": self.s3_output_path}

        if self.container_local_output_path is not None:
            tensorboard_output_config_request["LocalPath"] = self.container_local_output_path

        return tensorboard_output_config_request


class CollectionConfig(object):
    """Creates tensor collections for SageMaker Debugger."""

    def __init__(
        self,
        name: Union[str, PipelineVariable],
        parameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    ):
        """Constructor for collection configuration.

        Args:
            name (str or PipelineVariable): Required. The name of the collection configuration.
            parameters (dict[str, str] or dict[str, PipelineVariable]): Optional. The parameters
                for the collection configuration.

        **Example of creating a CollectionConfig object:**

        .. code-block:: python

            from sagemaker.debugger import CollectionConfig

            collection_configs=[
                CollectionConfig(name="tensor_collection_1")
                CollectionConfig(name="tensor_collection_2")
                ...
                CollectionConfig(name="tensor_collection_n")
            ]

        For a full list of Debugger built-in collection, see
        `Debugger Built in Collections
        <https://github.com/awslabs/sagemaker-debugger/blob/master
        /docs/api.md#built-in-collections>`_.

        **Example of creating a CollectionConfig object with parameter adjustment:**

        You can use the following CollectionConfig template in two ways:
        (1) to adjust the parameters of the built-in tensor collections,
        and (2) to create custom tensor collections.

        If you put the built-in collection names to the ``name`` parameter,
        ``CollectionConfig`` takes it to match the built-in collections and adjust parameters.
        If you specify a new name to the ``name`` parameter,
        ``CollectionConfig`` creates a new tensor collection, and you must use
        ``include_regex`` parameter to specify regex of tensors you want to collect.

        .. code-block:: python

            from sagemaker.debugger import CollectionConfig

            collection_configs=[
                CollectionConfig(
                    name="tensor_collection",
                    parameters={
                        "key_1": "value_1",
                        "key_2": "value_2"
                        ...
                        "key_n": "value_n"
                    }
                )
            ]

        The following list shows the available CollectionConfig parameters.

        +--------------------------+---------------------------------------------------------+
        | Parameter Key            | Descriptions                                            |
        +==========================+=========================================================+
        |``include_regex``         |  Specify a list of regex patterns of tensors to save.   |
        |                          |                                                         |
        |                          |  Tensors whose names match these patterns will be saved.|
        +--------------------------+---------------------------------------------------------+
        |``save_histogram``        |  Set *True* if want to save histogram output data for   |
        |                          |                                                         |
        |                          |  TensorFlow visualization.                              |
        +--------------------------+---------------------------------------------------------+
        |``reductions``            |  Specify certain reduction values of tensors.           |
        |                          |                                                         |
        |                          |  This helps reduce the amount of data saved and         |
        |                          |                                                         |
        |                          |  increase training speed.                               |
        |                          |                                                         |
        |                          |  Available values are ``min``, ``max``, ``median``,     |
        |                          |                                                         |
        |                          |  ``mean``, ``std``, ``variance``, ``sum``, and ``prod``.|
        +--------------------------+---------------------------------------------------------+
        |``save_interval``         |  Specify how often to save tensors in steps.            |
        |                          |                                                         |
        |``train.save_interval``   |  You can also specify the save intervals                |
        |                          |                                                         |
        |``eval.save_interval``    |  in TRAIN, EVAL, PREDICT, and GLOBAL modes.             |
        |                          |                                                         |
        |``predict.save_interval`` |  The default value is 500 steps.                        |
        |                          |                                                         |
        |``global.save_interval``  |                                                         |
        +--------------------------+---------------------------------------------------------+
        |``save_steps``            |  Specify the exact step numbers to save tensors.        |
        |                          |                                                         |
        |``train.save_steps``      |  You can also specify the save steps                    |
        |                          |                                                         |
        |``eval.save_steps``       |  in TRAIN, EVAL, PREDICT, and GLOBAL modes.             |
        |                          |                                                         |
        |``predict.save_steps``    |                                                         |
        |                          |                                                         |
        |``global.save_steps``     |                                                         |
        +--------------------------+---------------------------------------------------------+
        |``start_step``            |  Specify the exact start step to save tensors.          |
        |                          |                                                         |
        |``train.start_step``      |  You can also specify the start steps                   |
        |                          |                                                         |
        |``eval.start_step``       |  in TRAIN, EVAL, PREDICT, and GLOBAL modes.             |
        |                          |                                                         |
        |``predict.start_step``    |                                                         |
        |                          |                                                         |
        |``global.start_step``     |                                                         |
        +--------------------------+---------------------------------------------------------+
        |``end_step``              |  Specify the exact end step to save tensors.            |
        |                          |                                                         |
        |``train.end_step``        |  You can also specify the end steps                     |
        |                          |                                                         |
        |``eval.end_step``         |  in TRAIN, EVAL, PREDICT, and GLOBAL modes.             |
        |                          |                                                         |
        |``predict.end_step``      |                                                         |
        |                          |                                                         |
        |``global.end_step``       |                                                         |
        +--------------------------+---------------------------------------------------------+

        For example, the following code shows how to control the save_interval parameters
        of the built-in ``losses`` tensor collection. With the following collection configuration,
        Debugger collects loss values every 100 steps from training loops and every 10 steps
        from evaluation loops.

        .. code-block:: python

            collection_configs=[
                CollectionConfig(
                    name="losses",
                    parameters={
                        "train.save_interval": "100",
                        "eval.save_interval": "10"
                    }
                )
            ]

        """
        self.name = name
        self.parameters = parameters

    def __eq__(self, other):
        """Equal method override.

        Args:
            other: Object to test equality against.

        """
        if not isinstance(other, CollectionConfig):
            raise TypeError(
                "CollectionConfig is only comparable with other CollectionConfig objects."
            )

        return self.name == other.name and self.parameters == other.parameters

    def __ne__(self, other):
        """Not-equal method override.

        Args:
            other: Object to test equality against.

        """
        if not isinstance(other, CollectionConfig):
            raise TypeError(
                "CollectionConfig is only comparable with other CollectionConfig objects."
            )

        return self.name != other.name or self.parameters != other.parameters

    def __hash__(self):
        """Hash method override."""
        return hash((self.name, tuple(sorted((self.parameters or {}).items()))))

    def _to_request_dict(self):
        """Generate a request dictionary using the parameters initializing the object.

        Returns:
            dict: A portion of an API request as a dictionary.

        """
        collection_config_request = {"CollectionName": self.name}

        if self.parameters is not None:
            collection_config_request["CollectionParameters"] = self.parameters

        return collection_config_request
