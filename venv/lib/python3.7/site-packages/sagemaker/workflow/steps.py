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
"""The `Step` definitions for SageMaker Pipelines Workflows."""
from __future__ import absolute_import

import abc
import warnings

from enum import Enum
from typing import Dict, List, Set, Union, Optional, Any, TYPE_CHECKING
from urllib.parse import urlparse

import attr

from sagemaker import Session
from sagemaker.estimator import EstimatorBase, _TrainingJob
from sagemaker.inputs import CreateModelInput, TrainingInput, TransformInput, FileSystemInput
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.processing import (
    ProcessingInput,
    ProcessingJob,
    ProcessingOutput,
    Processor,
)
from sagemaker.transformer import Transformer, _TransformJob
from sagemaker.tuner import HyperparameterTuner, _TuningJob
from sagemaker.workflow.conditions import Condition
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import (
    DefaultEnumMeta,
    Entity,
    RequestType,
)
from sagemaker.workflow.pipeline_context import _JobStepArguments
from sagemaker.workflow.properties import (
    PropertyFile,
    Properties,
)
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.retry import RetryPolicy
from sagemaker.workflow.utilities import trim_request_dict

if TYPE_CHECKING:
    from sagemaker.workflow.step_collections import StepCollection


class StepTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """Enum of `Step` types."""

    CONDITION = "Condition"
    CREATE_MODEL = "Model"
    PROCESSING = "Processing"
    REGISTER_MODEL = "RegisterModel"
    TRAINING = "Training"
    TRANSFORM = "Transform"
    CALLBACK = "Callback"
    TUNING = "Tuning"
    LAMBDA = "Lambda"
    QUALITY_CHECK = "QualityCheck"
    CLARIFY_CHECK = "ClarifyCheck"
    EMR = "EMR"
    FAIL = "Fail"
    AUTOML = "AutoML"


@attr.s
class Step(Entity):
    """Pipeline `Step` for SageMaker Pipelines Workflows.

    Attributes:
        name (str): The name of the `Step`.
        display_name (str): The display name of the `Step`.
        description (str): The description of the `Step`.
        step_type (StepTypeEnum): The type of the `Step`.
        depends_on (List[Union[str, Step, StepCollection]]): The list of `Step`/`StepCollection`
            names or `Step` instances or `StepCollection` instances that the current `Step`
            depends on.
    """

    name: str = attr.ib(factory=str)
    display_name: Optional[str] = attr.ib(default=None)
    description: Optional[str] = attr.ib(default=None)
    step_type: StepTypeEnum = attr.ib(factory=StepTypeEnum.factory)
    depends_on: Optional[List[Union[str, "Step", "StepCollection"]]] = attr.ib(default=None)

    @property
    @abc.abstractmethod
    def arguments(self) -> RequestType:
        """The arguments to the particular `Step` service call."""

    @property
    def step_only_arguments(self) -> RequestType:
        """The arguments to this Step only.

        Compound Steps such as the ConditionStep will have to
        override this method to return arguments pertaining to only that step.
        """
        return self.arguments

    @property
    @abc.abstractmethod
    def properties(self):
        """The properties of the particular `Step`."""

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        request_dict = {
            "Name": self.name,
            "Type": self.step_type.value,
            "Arguments": self.arguments,
        }
        if self.depends_on:
            request_dict["DependsOn"] = self._resolve_depends_on(self.depends_on)
        if self.display_name:
            request_dict["DisplayName"] = self.display_name
        if self.description:
            request_dict["Description"] = self.description

        return request_dict

    def add_depends_on(self, step_names: List[Union[str, "Step", "StepCollection"]]):
        """Add `Step` names or `Step` instances to the current `Step` depends on list."""

        if not step_names:
            return

        if not self.depends_on:
            self.depends_on = []
        self.depends_on.extend(step_names)

    @property
    def ref(self) -> Dict[str, str]:
        """Gets a reference dictionary for `Step` instances."""
        return {"Name": self.name}

    @staticmethod
    def _resolve_depends_on(
        depends_on_list: List[Union[str, "Step", "StepCollection"]]
    ) -> List[str]:
        """Resolve the `Step` depends on list."""
        from sagemaker.workflow.step_collections import StepCollection

        depends_on = []
        for step in depends_on_list:
            # As for StepCollection, the names of its sub steps will be interpolated
            # when generating the pipeline definition
            if isinstance(step, (Step, StepCollection)):
                depends_on.append(step.name)
            elif isinstance(step, str):
                depends_on.append(step)
            else:
                raise ValueError(f"Invalid input step type: {type(step)}")
        return depends_on

    def _find_step_dependencies(
        self, step_map: Dict[str, Union["Step", "StepCollection"]]
    ) -> List[str]:
        """Find the all step names this step is dependent on."""
        step_dependencies = set()
        if self.depends_on:
            step_dependencies.update(self._find_dependencies_in_depends_on_list(step_map))
        step_dependencies.update(
            self._find_dependencies_in_step_arguments(self.step_only_arguments, step_map)
        )
        return list(step_dependencies)

    def _find_dependencies_in_depends_on_list(
        self, step_map: Dict[str, Union["Step", "StepCollection"]]
    ) -> Set[str]:
        """Find dependency steps referenced in the depends-on field of this step."""
        # import here to prevent circular import
        from sagemaker.workflow.step_collections import StepCollection

        dependencies = set()
        for step in self.depends_on:
            if isinstance(step, Step):
                dependencies.add(step.name)
            elif isinstance(step, StepCollection):
                dependencies.add(step.steps[-1].name)
            elif isinstance(step, str):
                # step could be the name of a `Step` or a `StepCollection`
                dependencies.add(self._get_step_name_from_str(step, step_map))
        return dependencies

    def _find_dependencies_in_step_arguments(
        self, obj: Any, step_map: Dict[str, Union["Step", "StepCollection"]]
    ):
        """Find the step dependencies referenced in the arguments of this step."""
        dependencies = set()
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, (PipelineVariable, Condition)):
                    for referenced_step in value._referenced_steps:
                        dependencies.add(self._get_step_name_from_str(referenced_step, step_map))
                    if isinstance(value, JsonGet):
                        self._validate_json_get_function(value, step_map)
                dependencies.update(self._find_dependencies_in_step_arguments(value, step_map))
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (PipelineVariable, Condition)):
                    for referenced_step in item._referenced_steps:
                        dependencies.add(self._get_step_name_from_str(referenced_step, step_map))
                    if isinstance(item, JsonGet):
                        self._validate_json_get_function(item, step_map)
                dependencies.update(self._find_dependencies_in_step_arguments(item, step_map))
        return dependencies

    def _validate_json_get_function(
        self, json_get: JsonGet, step_map: Dict[str, Union["Step", "StepCollection"]]
    ):
        """Validate the JsonGet function inputs."""
        property_file_reference = json_get.property_file
        processing_step = step_map[json_get.step_name]
        property_file = None
        if isinstance(property_file_reference, str):
            if not isinstance(processing_step, ProcessingStep):
                raise ValueError(
                    f"Invalid JsonGet function {json_get.expr} in step '{self.name}'. JsonGet "
                    f"function can only be evaluated on processing step outputs."
                )
            for file in processing_step.property_files:
                if file.name == property_file_reference:
                    property_file = file
                    break
        elif isinstance(property_file_reference, PropertyFile):
            property_file = property_file_reference
        if property_file is None:
            raise ValueError(
                f"Invalid JsonGet function {json_get.expr} in step '{self.name}'. Property file "
                f"reference '{property_file_reference}' is undefined in step "
                f"'{processing_step.name}'."
            )
        property_file_output = None
        if "ProcessingOutputConfig" in processing_step.arguments:
            for output in processing_step.arguments["ProcessingOutputConfig"]["Outputs"]:
                if output["OutputName"] == property_file.output_name:
                    property_file_output = output
        if property_file_output is None:
            raise ValueError(
                f"Processing output name '{property_file.output_name}' defined in property file "
                f"'{property_file.name}' not found in processing step '{processing_step.name}'."
            )

    @staticmethod
    def _get_step_name_from_str(
        str_input: str, step_map: Dict[str, Union["Step", "StepCollection"]]
    ) -> str:
        """Convert a Step or StepCollection name input to step name."""
        from sagemaker.workflow.step_collections import StepCollection

        if str_input not in step_map:
            raise ValueError(f"Step {str_input} is undefined.")
        if isinstance(step_map[str_input], StepCollection):
            return step_map[str_input].steps[-1].name
        return str_input

    @staticmethod
    def _trim_experiment_config(request_dict: Dict):
        """For job steps, trim the experiment config to keep the trial component display name."""
        if request_dict.get("ExperimentConfig", {}).get("TrialComponentDisplayName"):
            request_dict["ExperimentConfig"] = {
                "TrialComponentDisplayName": request_dict["ExperimentConfig"][
                    "TrialComponentDisplayName"
                ]
            }
        else:
            request_dict.pop("ExperimentConfig", None)


@attr.s
class CacheConfig:
    """Configuration class to enable caching in SageMaker Pipelines Workflows.

    If caching is enabled, the pipeline attempts to find a previous execution of a `Step`
    that was called with the same arguments. `Step` caching only considers successful execution.
    If a successful previous execution is found, the pipeline propagates the values
    from the previous execution rather than recomputing the `Step`.
    When multiple successful executions exist within the timeout period,
    it uses the result for the most recent successful execution.


    Attributes:
        enable_caching (bool): To enable `Step` caching. Defaults to `False`.
        expire_after (str): If `Step` caching is enabled, a timeout also needs to defined.
            It defines how old a previous execution can be to be considered for reuse.
            Value should be an ISO 8601 duration string. Defaults to `None`.

            Examples::

                'p30d' # 30 days
                'P4DT12H' # 4 days and 12 hours
                'T12H' # 12 hours
    """

    enable_caching: bool = attr.ib(default=False)
    expire_after = attr.ib(
        default=None, validator=attr.validators.optional(attr.validators.instance_of(str))
    )

    @property
    def config(self):
        """Configures `Step` caching for SageMaker Pipelines Workflows."""
        config = {"Enabled": self.enable_caching}
        if self.expire_after is not None:
            config["ExpireAfter"] = self.expire_after
        return {"CacheConfig": config}


class ConfigurableRetryStep(Step):
    """`ConfigurableRetryStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_type: StepTypeEnum,
        display_name: str = None,
        description: str = None,
        depends_on: Optional[List[Union[str, Step, "StepCollection"]]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        super().__init__(
            name=name,
            display_name=display_name,
            step_type=step_type,
            description=description,
            depends_on=depends_on,
        )
        self.retry_policies = [] if not retry_policies else retry_policies

    def add_retry_policy(self, retry_policy: RetryPolicy):
        """Add a policy to the current `ConfigurableRetryStep` retry policies list."""
        if not retry_policy:
            return

        if not self.retry_policies:
            self.retry_policies = []
        self.retry_policies.append(retry_policy)

    def to_request(self) -> RequestType:
        """Gets the request structure for `ConfigurableRetryStep`."""
        step_dict = super().to_request()
        if self.retry_policies:
            step_dict["RetryPolicies"] = self._resolve_retry_policy(self.retry_policies)
        return step_dict

    @staticmethod
    def _resolve_retry_policy(retry_policy_list: List[RetryPolicy]) -> List[RequestType]:
        """Resolve the `ConfigurableRetryStep` retry policy list."""
        return [retry_policy.to_request() for retry_policy in retry_policy_list]


class TrainingStep(ConfigurableRetryStep):
    """`TrainingStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments = None,
        estimator: EstimatorBase = None,
        display_name: str = None,
        description: str = None,
        inputs: Union[TrainingInput, dict, str, FileSystemInput] = None,
        cache_config: CacheConfig = None,
        depends_on: Optional[List[Union[str, Step, "StepCollection"]]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        """Construct a `TrainingStep`, given an `EstimatorBase` instance.

        In addition to the `EstimatorBase` instance, the other arguments are those
        that are supplied to the `fit` method of the `sagemaker.estimator.Estimator`.

        Args:
            name (str): The name of the `TrainingStep`.
            step_args (_JobStepArguments): The arguments for the `TrainingStep` definition.
            estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
            display_name (str): The display name of the `TrainingStep`.
            description (str): The description of the `TrainingStep`.
            inputs (Union[str, dict, TrainingInput, FileSystemInput]): Information
                about the training data. This can be one of three types:

                * (str) the S3 location where training data is saved, or a file:// path in
                  local mode.
                * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) If using multiple
                  channels for training data, you can specify a dictionary mapping channel names to
                  strings or :func:`~sagemaker.inputs.TrainingInput` objects.
                * (sagemaker.inputs.TrainingInput) - channel configuration for S3 data sources
                  that can provide additional information as well as the path to the training
                  dataset.
                  See :func:`sagemaker.inputs.TrainingInput` for full details.
                * (sagemaker.inputs.FileSystemInput) - channel configuration for
                  a file system data source that can provide additional information as well as
                  the path to the training dataset.

            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `TrainingStep`
                depends on.
            retry_policies (List[RetryPolicy]):  A list of retry policies.
        """
        super(TrainingStep, self).__init__(
            name, StepTypeEnum.TRAINING, display_name, description, depends_on, retry_policies
        )

        if not (step_args is not None) ^ (estimator is not None):
            raise ValueError("Either step_args or estimator need to be given.")

        if step_args:
            from sagemaker.workflow.utilities import validate_step_args_input

            validate_step_args_input(
                step_args=step_args,
                expected_caller={Session.train.__name__},
                error_message="The step_args of TrainingStep must be obtained from estimator.fit().",
            )

        self.step_args = step_args
        self.estimator = estimator
        self.inputs = inputs

        self._properties = Properties(step_name=name, shape_name="DescribeTrainingJobResponse")
        self.cache_config = cache_config

        if self.cache_config:
            if (self.step_args and "ProfilerConfig" in self.step_args.func_kwargs) or (
                self.estimator is not None and not self.estimator.disable_profiler
            ):
                msg = (
                    "Profiling is enabled on the provided estimator. "
                    "The default profiler rule includes a timestamp "
                    "which will change each time the pipeline is "
                    "upserted, causing cache misses. If profiling "
                    "is not needed, set disable_profiler to True on the estimator."
                )
                warnings.warn(msg)

        if not self.step_args:
            warnings.warn(
                (
                    'We are deprecating the instantiation of TrainingStep using "estimator".'
                    'Instead, simply using "step_args".'
                ),
                DeprecationWarning,
            )

        self.job_name = None
        if estimator and (estimator.source_dir or estimator.entry_point):
            # By default, `Estimator` will upload the local code to an S3 path
            # containing a timestamp. This causes cache misses whenever a
            # pipeline is updated, even if the underlying script hasn't changed.
            # To avoid this, hash the contents of the training script and include it
            # in the `job_name` passed to the `Estimator`, which will be used
            # instead of the timestamped path.
            if not is_pipeline_variable(estimator.source_dir) and not is_pipeline_variable(
                estimator.entry_point
            ):
                self.job_name = self._generate_code_upload_path()

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_training_job`.

        NOTE: The `CreateTrainingJob` request is not quite the args list that workflow needs.
        `ExperimentConfig` attribute cannot be included.
        """
        from sagemaker.workflow.utilities import execute_job_functions
        from sagemaker.workflow.utilities import _pipeline_config

        if self.step_args:
            # execute fit function with saved parameters,
            # and store args in PipelineSession's _context
            execute_job_functions(self.step_args)

            # populate request dict with args
            estimator = self.step_args.func_args[0]
            request_dict = estimator.sagemaker_session.context.args
        else:
            self.estimator._prepare_for_training(self.job_name)
            train_args = _TrainingJob._get_train_args(
                self.estimator, self.inputs, experiment_config=dict()
            )
            request_dict = self.estimator.sagemaker_session._get_train_request(**train_args)

        if "HyperParameters" in request_dict:
            request_dict["HyperParameters"].pop("sagemaker_job_name", None)

        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "TrainingJobName", _pipeline_config)

        Step._trim_experiment_config(request_dict)

        return request_dict

    @property
    def properties(self):
        """A `Properties` object representing the `DescribeTrainingJobResponse` data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the request dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict

    def _generate_code_upload_path(self) -> str or None:
        """Generate an upload path for local training scripts based on their content."""
        from sagemaker.workflow.utilities import hash_files_or_dirs

        if self.estimator.source_dir:
            source_dir_url = urlparse(self.estimator.source_dir)
            if source_dir_url.scheme == "" or source_dir_url.scheme == "file":
                code_hash = hash_files_or_dirs(
                    [self.estimator.source_dir] + self.estimator.dependencies
                )
                return f"{self.name}-{code_hash}"[:1024]
        elif self.estimator.entry_point:
            entry_point_url = urlparse(self.estimator.entry_point)
            if entry_point_url.scheme == "" or entry_point_url.scheme == "file":
                code_hash = hash_files_or_dirs(
                    [self.estimator.entry_point] + self.estimator.dependencies
                )
                return f"{self.name}-{code_hash}"[:1024]
        return None


class CreateModelStep(ConfigurableRetryStep):
    """`CreateModelStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: Optional[dict] = None,
        model: Optional[Union[Model, PipelineModel]] = None,
        inputs: Optional[CreateModelInput] = None,
        depends_on: Optional[List[Union[str, Step, "StepCollection"]]] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Construct a `CreateModelStep`, given an `sagemaker.model.Model` instance.

        In addition to the `Model` instance, the other arguments are those that are supplied to
        the `_create_sagemaker_model` method of the `sagemaker.model.Model._create_sagemaker_model`.

        Args:
            name (str): The name of the `CreateModelStep`.
            step_args (dict): The arguments for the `CreateModelStep` definition (default: None).
            model (Model or PipelineModel): A `sagemaker.model.Model`
                or `sagemaker.pipeline.PipelineModel` instance (default: None).
            inputs (CreateModelInput): A `sagemaker.inputs.CreateModelInput` instance.
                (default: None).
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `CreateModelStep`
                depends on (default: None).
            retry_policies (List[RetryPolicy]):  A list of retry policies (default: None).
            display_name (str): The display name of the `CreateModelStep` (default: None).
            description (str): The description of the `CreateModelStep` (default: None).
        """
        super(CreateModelStep, self).__init__(
            name, StepTypeEnum.CREATE_MODEL, display_name, description, depends_on, retry_policies
        )
        if not (step_args is None) ^ (model is None):
            raise ValueError(
                "step_args and model are mutually exclusive. Either of them should be provided."
            )

        self.step_args = step_args
        self.model = model
        self.inputs = inputs or CreateModelInput()

        self._properties = Properties(step_name=name, shape_name="DescribeModelOutput")

        warnings.warn(
            (
                "We are deprecating the use of CreateModelStep. "
                "Instead, please use the ModelStep, which simply takes in the step arguments "
                "generated by model.create(). For more, see: "
                "https://sagemaker.readthedocs.io/en/stable/"
                "amazon_sagemaker_model_building_pipeline.html#model-step"
            ),
            DeprecationWarning,
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_model`.

        NOTE: The `CreateModelRequest` is not quite the args list that workflow needs.
        """
        from sagemaker.workflow.utilities import _pipeline_config

        if self.step_args:
            request_dict = self.step_args
        else:
            if isinstance(self.model, PipelineModel):
                request_dict = self.model.sagemaker_session._create_model_request(
                    name="",
                    role=self.model.role,
                    container_defs=self.model.pipeline_container_def(self.inputs.instance_type),
                    vpc_config=self.model.vpc_config,
                    enable_network_isolation=self.model.enable_network_isolation,
                )
            else:
                request_dict = self.model.sagemaker_session._create_model_request(
                    name="",
                    role=self.model.role,
                    container_defs=self.model.prepare_container_def(
                        instance_type=self.inputs.instance_type,
                        accelerator_type=self.inputs.accelerator_type,
                    ),
                    vpc_config=self.model.vpc_config,
                    enable_network_isolation=self.model.enable_network_isolation(),
                )

        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "ModelName", _pipeline_config)

        return request_dict

    @property
    def properties(self):
        """A `Properties` object representing the `DescribeModelResponse` data model."""
        return self._properties


class TransformStep(ConfigurableRetryStep):
    """`TransformStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments = None,
        transformer: Transformer = None,
        inputs: TransformInput = None,
        display_name: str = None,
        description: str = None,
        cache_config: CacheConfig = None,
        depends_on: Optional[List[Union[str, Step, "StepCollection"]]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        """Constructs a `TransformStep`, given a `Transformer` instance.

        In addition to the `Transformer` instance, the other arguments are those
        that are supplied to the `transform` method of the `sagemaker.transformer.Transformer`.

        Args:
            name (str): The name of the `TransformStep`.
            step_args (_JobStepArguments): The arguments for the `TransformStep` definition.
            transformer (Transformer): A `sagemaker.transformer.Transformer` instance.
            inputs (TransformInput): A `sagemaker.inputs.TransformInput` instance.
            cache_config (CacheConfig): A `sagemaker.workflow.steps.CacheConfig` instance.
            display_name (str): The display name of the `TransformStep`.
            description (str): The description of the `TransformStep`.
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `TransformStep`
                depends on.
            retry_policies (List[RetryPolicy]): A list of retry policies.
        """
        super(TransformStep, self).__init__(
            name, StepTypeEnum.TRANSFORM, display_name, description, depends_on, retry_policies
        )

        if not (step_args is not None) ^ (transformer is not None):
            raise ValueError("either step_args or transformer need to be given, but not both.")

        if step_args:
            from sagemaker.workflow.utilities import validate_step_args_input

            validate_step_args_input(
                step_args=step_args,
                expected_caller={Session.transform.__name__},
                error_message="The step_args of TransformStep must be obtained "
                "from transformer.transform().",
            )

        self.step_args = step_args
        self.transformer = transformer
        self.inputs = inputs
        self.cache_config = cache_config
        self._properties = Properties(step_name=name, shape_name="DescribeTransformJobResponse")

        if not self.step_args:
            if inputs is None:
                raise ValueError("Inputs can't be None when transformer is given.")
            warnings.warn(
                (
                    'We are deprecating the instantiation of TransformStep using "transformer".'
                    'Instead, simply using "step_args".'
                ),
                DeprecationWarning,
            )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_transform_job`.

        NOTE: The `CreateTransformJob` request is not quite the args list that workflow needs.
        `ExperimentConfig` cannot be included in the arguments.
        """
        from sagemaker.workflow.utilities import execute_job_functions
        from sagemaker.workflow.utilities import _pipeline_config

        if self.step_args:
            # execute transform function with saved parameters,
            # and store args in PipelineSession's _context
            execute_job_functions(self.step_args)

            # populate request dict with args
            transformer = self.step_args.func_args[0]
            request_dict = transformer.sagemaker_session.context.args
        else:
            transform_args = _TransformJob._get_transform_args(
                transformer=self.transformer,
                data=self.inputs.data,
                data_type=self.inputs.data_type,
                content_type=self.inputs.content_type,
                compression_type=self.inputs.compression_type,
                split_type=self.inputs.split_type,
                input_filter=self.inputs.input_filter,
                output_filter=self.inputs.output_filter,
                join_source=self.inputs.join_source,
                model_client_config=self.inputs.model_client_config,
                experiment_config=dict(),
                batch_data_capture_config=self.inputs.batch_data_capture_config,
            )
            request_dict = self.transformer.sagemaker_session._get_transform_request(
                **transform_args
            )

        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "TransformJobName", _pipeline_config)

        Step._trim_experiment_config(request_dict)

        return request_dict

    @property
    def properties(self):
        """A `Properties` object representing the `DescribeTransformJobResponse` data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict


class ProcessingStep(ConfigurableRetryStep):
    """`ProcessingStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments = None,
        processor: Processor = None,
        display_name: str = None,
        description: str = None,
        inputs: List[ProcessingInput] = None,
        outputs: List[ProcessingOutput] = None,
        job_arguments: List[str] = None,
        code: str = None,
        property_files: List[PropertyFile] = None,
        cache_config: CacheConfig = None,
        depends_on: Optional[List[Union[str, Step, "StepCollection"]]] = None,
        retry_policies: List[RetryPolicy] = None,
        kms_key=None,
    ):
        """Construct a `ProcessingStep`, given a `Processor` instance.

        In addition to the `Processor` instance, the other arguments are those that are supplied to
        the `process` method of the `sagemaker.processing.Processor`.

        Args:
            name (str): The name of the `ProcessingStep`.
            step_args (_JobStepArguments): The arguments for the `ProcessingStep` definition.
            processor (Processor): A `sagemaker.processing.Processor` instance.
            display_name (str): The display name of the `ProcessingStep`.
            description (str): The description of the `ProcessingStep`
            inputs (List[ProcessingInput]): A list of `sagemaker.processing.ProcessorInput`
                instances. Defaults to `None`.
            outputs (List[ProcessingOutput]): A list of `sagemaker.processing.ProcessorOutput`
                instances. Defaults to `None`.
            job_arguments (List[str]): A list of strings to be passed into the processing job.
                Defaults to `None`.
            code (str): This can be an S3 URI or a local path to a file with the framework
                script to run. Defaults to `None`.
            property_files (List[PropertyFile]): A list of property files that workflow looks
                for and resolves from the configured processing output list.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `ProcessingStep`
                depends on.
            retry_policies (List[RetryPolicy]):  A list of retry policies.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file. Defaults to `None`.
        """
        super(ProcessingStep, self).__init__(
            name, StepTypeEnum.PROCESSING, display_name, description, depends_on, retry_policies
        )

        if not (step_args is not None) ^ (processor is not None):
            raise ValueError("either step_args or processor need to be given, but not both.")

        if step_args:
            from sagemaker.workflow.utilities import validate_step_args_input

            validate_step_args_input(
                step_args=step_args,
                expected_caller={Session.process.__name__},
                error_message="The step_args of ProcessingStep must be obtained from processor.run().",
            )

        self.step_args = step_args
        self.processor = processor
        self.inputs = inputs
        self.outputs = outputs
        self.job_arguments = job_arguments
        self.code = code
        self.property_files = property_files or []
        self.job_name = None
        self.kms_key = kms_key
        self.cache_config = cache_config
        self._properties = Properties(step_name=name, shape_name="DescribeProcessingJobResponse")

        if not self.step_args:
            # Examine why run method in `sagemaker.processing.Processor`
            # mutates the processor instance by setting the instance's
            # arguments attribute. Refactor `Processor.run`, if possible.
            self.processor.arguments = job_arguments

            if code:
                if is_pipeline_variable(code):
                    raise ValueError(
                        "code argument has to be a valid S3 URI or local file path "
                        + "rather than a pipeline variable"
                    )
                code_url = urlparse(code)
                if code_url.scheme == "" or code_url.scheme == "file":
                    # By default, `Processor` will upload the local code to an S3 path
                    # containing a timestamp. This causes cache misses whenever a
                    # pipeline is updated, even if the underlying script hasn't changed.
                    # To avoid this, hash the contents of the script and include it
                    # in the `job_name` passed to the `Processor`, which will be used
                    # instead of the timestamped path.
                    self.job_name = self._generate_code_upload_path()

            warnings.warn(
                (
                    'We are deprecating the instantiation of ProcessingStep using "processor".'
                    'Instead, simply using "step_args".'
                ),
                DeprecationWarning,
            )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_processing_job`.

        NOTE: The `CreateProcessingJob` request is not quite the args list that workflow needs.
        `ExperimentConfig` cannot be included in the arguments.
        """
        from sagemaker.workflow.utilities import execute_job_functions
        from sagemaker.workflow.utilities import _pipeline_config

        if self.step_args:
            # execute run function with saved parameters,
            # and store args in PipelineSession's _context
            execute_job_functions(self.step_args)

            # populate request dict with args
            processor = self.step_args.func_args[0]
            request_dict = processor.sagemaker_session.context.args
        else:
            normalized_inputs, normalized_outputs = self.processor._normalize_args(
                job_name=self.job_name,
                arguments=self.job_arguments,
                inputs=self.inputs,
                outputs=self.outputs,
                code=self.code,
                kms_key=self.kms_key,
            )
            process_args = ProcessingJob._get_process_args(
                self.processor, normalized_inputs, normalized_outputs, experiment_config=dict()
            )
            request_dict = self.processor.sagemaker_session._get_process_request(**process_args)

        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "ProcessingJobName", _pipeline_config)

        Step._trim_experiment_config(request_dict)

        return request_dict

    @property
    def properties(self):
        """A `Properties` object representing the `DescribeProcessingJobResponse` data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        request_dict = super(ProcessingStep, self).to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)
        if self.property_files:
            request_dict["PropertyFiles"] = [
                property_file.expr for property_file in self.property_files
            ]
        return request_dict

    def _generate_code_upload_path(self) -> str:
        """Generate an upload path for local processing scripts based on its contents."""
        from sagemaker.workflow.utilities import hash_file

        code_hash = hash_file(self.code)
        return f"{self.name}-{code_hash}"[:1024]


class TuningStep(ConfigurableRetryStep):
    """`TuningStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments = None,
        tuner: HyperparameterTuner = None,
        display_name: str = None,
        description: str = None,
        inputs=None,
        job_arguments: List[str] = None,
        cache_config: CacheConfig = None,
        depends_on: Optional[List[Union[str, Step, "StepCollection"]]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        """Construct a `TuningStep`, given a `HyperparameterTuner` instance.

        In addition to the `HyperparameterTuner` instance, the other arguments are those
        that are supplied to the `fit` method of the `sagemaker.tuner.HyperparameterTuner`.

        Args:
            name (str): The name of the `TuningStep`.
            step_args (_JobStepArguments): The arguments for the `TuningStep` definition.
            tuner (HyperparameterTuner): A `sagemaker.tuner.HyperparameterTuner` instance.
            display_name (str): The display name of the `TuningStep`.
            description (str): The description of the `TuningStep`.
            inputs: Information about the training data. Please refer to the
                `fit()` method of the associated estimator, as this can take
                any of the following forms:

                * (str) - The S3 location where training data is saved.
                * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) -
                    If using multiple channels for training data, you can specify
                    a dictionary mapping channel names to strings or
                    :func:`~sagemaker.inputs.TrainingInput` objects.
                * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources
                    that can provide additional information about the training dataset.
                    See :func:`sagemaker.inputs.TrainingInput` for full details.
                * (sagemaker.session.FileSystemInput) - channel configuration for
                    a file system data source that can provide additional information as well as
                    the path to the training dataset.
                * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                    Amazon :class:~`Record` objects serialized and stored in S3.
                    For use with an estimator for an Amazon algorithm.
                * (sagemaker.amazon.amazon_estimator.FileSystemRecordSet) -
                    Amazon SageMaker channel configuration for a file system data source for
                    Amazon algorithms.
                * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
                    where each instance is a different channel of training data.
                * (list[sagemaker.amazon.amazon_estimator.FileSystemRecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.FileSystemRecordSet` objects,
                    where each instance is a different channel of training data.
            job_arguments (List[str]): A list of strings to be passed into the processing job.
                Defaults to `None`.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `TuningStep`
                depends on.
            retry_policies (List[RetryPolicy]):  A list of retry policies.
        """
        super(TuningStep, self).__init__(
            name, StepTypeEnum.TUNING, display_name, description, depends_on, retry_policies
        )

        if not (step_args is not None) ^ (tuner is not None):
            raise ValueError("either step_args or tuner need to be given, but not both.")

        if step_args:
            from sagemaker.workflow.utilities import validate_step_args_input

            validate_step_args_input(
                step_args=step_args,
                expected_caller={Session.create_tuning_job.__name__},
                error_message="The step_args of TuningStep must be obtained from tuner.fit().",
            )

        self.step_args = step_args
        self.tuner = tuner
        self.inputs = inputs
        self.job_arguments = job_arguments
        self._properties = Properties(
            step_name=name,
            shape_names=[
                "DescribeHyperParameterTuningJobResponse",
                "ListTrainingJobsForHyperParameterTuningJobResponse",
            ],
        )
        self.cache_config = cache_config

        if not self.step_args:
            warnings.warn(
                (
                    'We are deprecating the instantiation of TuningStep using "tuner".'
                    'Instead, simply using "step_args".'
                ),
                DeprecationWarning,
            )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_hyper_parameter_tuning_job`.

        NOTE: The `CreateHyperParameterTuningJob` request is not quite the
            args list that workflow needs.
        """
        from sagemaker.workflow.utilities import execute_job_functions
        from sagemaker.workflow.utilities import _pipeline_config

        if self.step_args:
            # execute fit function with saved parameters,
            # and store args in PipelineSession's _context
            execute_job_functions(self.step_args)

            # populate request dict with args
            tuner = self.step_args.func_args[0]
            request_dict = tuner.sagemaker_session.context.args
        else:
            if self.tuner.estimator is not None:
                self.tuner.estimator._prepare_for_training()
            else:
                for _, estimator in self.tuner.estimator_dict.items():
                    estimator._prepare_for_training()

            self.tuner._prepare_for_tuning()
            tuner_args = _TuningJob._get_tuner_args(self.tuner, self.inputs)
            request_dict = self.tuner.sagemaker_session._get_tuning_request(**tuner_args)

        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(
            request_dict, "HyperParameterTuningJobName", _pipeline_config
        )

        return request_dict

    @property
    def properties(self):
        """A `Properties` object

        A `Properties` object representing `DescribeHyperParameterTuningJobResponse` and
        `ListTrainingJobsForHyperParameterTuningJobResponse` data model.
        """
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict

    def get_top_model_s3_uri(self, top_k: int, s3_bucket: str, prefix: str = "") -> Join:
        """Get the model artifact S3 URI from the top performing training jobs.

        Args:
            top_k (int): The index of the top performing training job
                tuning step stores up to 50 top performing training jobs.
                A valid top_k value is from 0 to 49. The best training job
                model is at index 0.
            s3_bucket (str): The S3 bucket to store the training job output artifact.
            prefix (str): The S3 key prefix to store the training job output artifact.
        """
        values = ["s3:/", s3_bucket]
        if prefix != "" and prefix is not None:
            values.append(prefix)

        return Join(
            on="/",
            values=values
            + [
                self.properties.TrainingJobSummaries[top_k].TrainingJobName,
                "output/model.tar.gz",
            ],
        )
