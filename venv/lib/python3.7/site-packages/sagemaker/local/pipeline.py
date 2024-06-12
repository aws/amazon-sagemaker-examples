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
"""Local Pipeline Executor"""
from __future__ import absolute_import
from abc import ABC, abstractmethod

import json
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Union
from botocore.exceptions import ClientError

from sagemaker.workflow.conditions import ConditionTypeEnum
from sagemaker.workflow.steps import StepTypeEnum, Step
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.functions import Join, JsonGet, PropertyFile
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.execution_variables import ExecutionVariable, ExecutionVariables
from sagemaker.workflow.pipeline import PipelineGraph
from sagemaker.local.exceptions import StepExecutionException
from sagemaker.local.utils import get_using_dot_notation
from sagemaker.utils import unique_name_from_base
from sagemaker.s3 import parse_s3_url, s3_path_join


PRIMITIVES = (str, int, bool, float)
BINARY_CONDITION_TYPES = (
    ConditionTypeEnum.EQ.value,
    ConditionTypeEnum.GT.value,
    ConditionTypeEnum.GTE.value,
    ConditionTypeEnum.LT.value,
    ConditionTypeEnum.LTE.value,
)


class LocalPipelineExecutor(object):
    """An executor that executes SageMaker Pipelines locally."""

    def __init__(self, execution, sagemaker_session):
        """Initialize StepExecutor.

        Args:
            sagemaker_session (sagemaker.session.Session): a session to use to read configurations
                from, and use its boto client.
        """
        self.sagemaker_session = sagemaker_session
        self.execution = execution
        self.pipeline_dag = PipelineGraph.from_pipeline(self.execution.pipeline)
        self.local_sagemaker_client = self.sagemaker_session.sagemaker_client
        self._blocked_steps = set()
        self._step_executor_factory = _StepExecutorFactory(self)

    def execute(self):
        """Execute a local pipeline."""
        try:
            for step in self.pipeline_dag:
                if step.name not in self._blocked_steps:
                    self._execute_step(step)
        except StepExecutionException as e:
            self.execution.update_execution_failure(e.step_name, e.message)
        else:
            self.execution.update_execution_success()
        return self.execution

    def _execute_step(self, step):
        """Execute a local pipeline step."""
        self.execution.mark_step_executing(step.name)
        step_properties = self._step_executor_factory.get(step).execute()
        self.execution.update_step_properties(step.name, step_properties)

    def evaluate_step_arguments(self, step):
        """Parses and evaluate step arguments."""
        return self._parse_arguments(step.arguments, step.name)

    def _parse_arguments(self, obj, step_name):
        """Parse and evaluate arguments field"""
        if isinstance(obj, dict):
            obj_copy = deepcopy(obj)
            for k, v in obj.items():
                obj_copy[k] = self._parse_arguments(v, step_name)
            return obj_copy
        if isinstance(obj, list):
            list_copy = []
            for item in obj:
                list_copy.append(self._parse_arguments(item, step_name))
            return list_copy
        if isinstance(obj, PipelineVariable):
            return self.evaluate_pipeline_variable(obj, step_name)
        return obj

    def evaluate_pipeline_variable(self, pipeline_variable, step_name):
        """Evaluate pipeline variable runtime value."""
        value = None
        if isinstance(pipeline_variable, PRIMITIVES):
            value = pipeline_variable
        elif isinstance(pipeline_variable, Parameter):
            value = self.execution.pipeline_parameters.get(pipeline_variable.name)
        elif isinstance(pipeline_variable, Join):
            evaluated = [
                str(self.evaluate_pipeline_variable(v, step_name)) for v in pipeline_variable.values
            ]
            value = pipeline_variable.on.join(evaluated)
        elif isinstance(pipeline_variable, Properties):
            value = self._evaluate_property_reference(pipeline_variable, step_name)
        elif isinstance(pipeline_variable, ExecutionVariable):
            value = self._evaluate_execution_variable(pipeline_variable)
        elif isinstance(pipeline_variable, JsonGet):
            value = self._evaluate_json_get_function(pipeline_variable, step_name)
        else:
            self.execution.update_step_failure(
                step_name, f"Unrecognized pipeline variable {pipeline_variable.expr}."
            )

        if value is None:
            self.execution.update_step_failure(step_name, f"{pipeline_variable.expr} is undefined.")
        return value

    def _evaluate_property_reference(self, pipeline_variable, step_name):
        """Evaluate property reference runtime value."""
        try:
            referenced_step_name = pipeline_variable.step_name
            step_properties = self.execution.step_execution.get(referenced_step_name).properties
            return get_using_dot_notation(step_properties, pipeline_variable.path)
        except ValueError:
            self.execution.update_step_failure(step_name, f"{pipeline_variable.expr} is undefined.")

    def _evaluate_execution_variable(self, pipeline_variable):
        """Evaluate pipeline execution variable runtime value."""
        if pipeline_variable in (
            ExecutionVariables.PIPELINE_NAME,
            ExecutionVariables.PIPELINE_ARN,
        ):
            return self.execution.pipeline.name
        if pipeline_variable in (
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            ExecutionVariables.PIPELINE_EXECUTION_ARN,
        ):
            return self.execution.pipeline_execution_name
        if pipeline_variable == ExecutionVariables.START_DATETIME:
            return self.execution.creation_time
        if pipeline_variable == ExecutionVariables.CURRENT_DATETIME:
            return datetime.now()
        return None

    def _evaluate_json_get_function(self, pipeline_variable, step_name):
        """Evaluate join function runtime value."""
        property_file_reference = pipeline_variable.property_file
        property_file = None
        if isinstance(property_file_reference, str):
            processing_step = self.pipeline_dag.step_map[pipeline_variable.step_name]
            for file in processing_step.property_files:
                if file.name == property_file_reference:
                    property_file = file
                    break
        elif isinstance(property_file_reference, PropertyFile):
            property_file = property_file_reference
        processing_step_response = self.execution.step_execution.get(
            pipeline_variable.step_name
        ).properties
        if (
            "ProcessingOutputConfig" not in processing_step_response
            or "Outputs" not in processing_step_response["ProcessingOutputConfig"]
        ):
            self.execution.update_step_failure(
                step_name,
                f"Step '{pipeline_variable.step_name}' does not yet contain processing outputs.",
            )
        processing_output_s3_bucket = processing_step_response["ProcessingOutputConfig"]["Outputs"][
            property_file.output_name
        ]["S3Output"]["S3Uri"]
        try:
            s3_bucket, s3_key_prefix = parse_s3_url(processing_output_s3_bucket)
            file_content = self.sagemaker_session.read_s3_file(
                s3_bucket, s3_path_join(s3_key_prefix, property_file.path)
            )
            file_json = json.loads(file_content)
            return get_using_dot_notation(file_json, pipeline_variable.json_path)
        except ClientError as e:
            self.execution.update_step_failure(
                step_name,
                f"Received an error while file reading file '{property_file.path}' from S3: "
                f"{e.response.get('Code')}: {e.response.get('Message')}",
            )
        except json.JSONDecodeError:
            self.execution.update_step_failure(
                step_name,
                f"Contents of property file '{property_file.name}' are not in valid JSON format.",
            )
        except ValueError:
            self.execution.update_step_failure(
                step_name, f"Invalid json path '{pipeline_variable.json_path}'"
            )


class _StepExecutor(ABC):
    """An abstract base class for step executors running steps locally"""

    def __init__(self, pipeline_executor: LocalPipelineExecutor, step: Step):
        self.pipline_executor = pipeline_executor
        self.step = step

    @abstractmethod
    def execute(self) -> Dict:
        """Execute a pipeline step locally

        Returns:
            A dictionary as properties of the current step
        """

    def _convert_list_to_dict(self, dictionary: dict, path_to_list: str, reducing_key: str):
        """Convert list into dictionary using a field inside list elements as the keys.

        Raises RuntimeError if given list not able to be converted into a map based on given key.
        """

        try:
            list_to_convert = get_using_dot_notation(dictionary, path_to_list)
        except ValueError:
            raise RuntimeError(f"{path_to_list} does not exist in {dictionary}")
        if not isinstance(list_to_convert, list):
            raise RuntimeError(
                f"Element at path {path_to_list} is not a list. Actual type {type(list_to_convert)}"
            )
        converted_map = {}
        for element in list_to_convert:
            if not isinstance(element, dict):
                raise RuntimeError(
                    f"Cannot convert element of type {type(element)} into dictionary entry"
                )
            converted_map[element[reducing_key]] = element
        return converted_map


class _TrainingStepExecutor(_StepExecutor):
    """Executor class to execute TrainingStep locally"""

    def execute(self):
        job_name = unique_name_from_base(self.step.name)
        step_arguments = self.pipline_executor.evaluate_step_arguments(self.step)
        try:
            self.pipline_executor.local_sagemaker_client.create_training_job(
                job_name, **step_arguments
            )
            return self.pipline_executor.local_sagemaker_client.describe_training_job(job_name)
        except Exception as e:  # pylint: disable=W0703
            self.pipline_executor.execution.update_step_failure(
                self.step.name, f"{type(e).__name__}: {str(e)}"
            )


class _ProcessingStepExecutor(_StepExecutor):
    """Executor class to execute ProcessingStep locally"""

    def execute(self):
        job_name = unique_name_from_base(self.step.name)
        step_arguments = self.pipline_executor.evaluate_step_arguments(self.step)
        try:
            self.pipline_executor.local_sagemaker_client.create_processing_job(
                job_name, **step_arguments
            )
            job_describe_response = (
                self.pipline_executor.local_sagemaker_client.describe_processing_job(job_name)
            )
            if (
                "ProcessingOutputConfig" in job_describe_response
                and "Outputs" in job_describe_response["ProcessingOutputConfig"]
            ):
                job_describe_response["ProcessingOutputConfig"][
                    "Outputs"
                ] = self._convert_list_to_dict(
                    job_describe_response, "ProcessingOutputConfig.Outputs", "OutputName"
                )
            if "ProcessingInputs" in job_describe_response:
                job_describe_response["ProcessingInputs"] = self._convert_list_to_dict(
                    job_describe_response, "ProcessingInputs", "InputName"
                )
            return job_describe_response

        except Exception as e:  # pylint: disable=W0703
            self.pipline_executor.execution.update_step_failure(
                self.step.name, f"{type(e).__name__}: {str(e)}"
            )


class _ConditionStepExecutor(_StepExecutor):
    """Executor class to execute ConditionStep locally"""

    def execute(self):
        def _block_all_downstream_steps(steps: List[Union[Step, StepCollection]]):
            steps_to_block = set()
            for step in steps:
                steps_to_block.update(self.pipline_executor.pipeline_dag.get_steps_in_sub_dag(step))
            self.pipline_executor._blocked_steps.update(steps_to_block)

        if_steps = self.step.if_steps
        else_steps = self.step.else_steps
        step_only_arguments = self.pipline_executor._parse_arguments(
            self.step.step_only_arguments, self.step.name
        )

        outcome = self._evaluate_conjunction(step_only_arguments["Conditions"])

        if not outcome:
            _block_all_downstream_steps(if_steps)
        else:
            _block_all_downstream_steps(else_steps)

        return dict(Outcome=outcome)

    def _evaluate_conjunction(self, conditions: List[Dict]) -> bool:
        """Evaluate conditions of current conditionStep.

        Args:
            List of dictionaries representing conditions as request

        Returns:
            True if the conjunction expression is true,
            False otherwise.
        """
        for condition in conditions:
            if not self._resolve_condition(condition):
                return False
        return True

    def _resolve_condition(self, condition: dict) -> bool:
        """Resolve given condition.

        Args:
            Dictionary representing given condition as request

        Returns:
            True if given condition evaluated as true,
            False otherwise.
        """

        condition_type = condition["Type"]
        outcome = None
        if condition_type in BINARY_CONDITION_TYPES:
            outcome = self._resolve_binary_condition(condition, condition_type)
        elif condition_type == ConditionTypeEnum.NOT.value:
            outcome = self._resolve_not_condition(condition)
        elif condition_type == ConditionTypeEnum.OR.value:
            outcome = self._resolve_or_condition(condition)
        elif condition_type == ConditionTypeEnum.IN.value:
            outcome = self._resolve_in_condition(condition)
        else:
            raise NotImplementedError(f"Condition of type [{condition_type}] is not supported.")

        return outcome

    def _resolve_binary_condition(self, binary_condition: dict, binary_condition_type: str):
        """Resolve given binary condition.

        Args:
            Dictionary representing given binary condition as request

        Returns:
            True if given binary condition evaluated as true,
            False otherwise.
        """

        left_value = binary_condition["LeftValue"]
        right_value = binary_condition["RightValue"]
        try:
            outcome = None
            if binary_condition_type == ConditionTypeEnum.EQ.value:
                if not isinstance(left_value, type(right_value)) and not isinstance(
                    right_value, type(left_value)
                ):
                    self.pipline_executor.execution.update_step_failure(
                        self.step.name,
                        f"LeftValue [{left_value}] of type [{type(left_value)}] and "
                        + f"RightValue [{right_value}] of type [{type(right_value)}] "
                        + "are not of the same type.",
                    )
                outcome = left_value == right_value
            elif binary_condition_type == ConditionTypeEnum.GT.value:
                outcome = left_value > right_value
            elif binary_condition_type == ConditionTypeEnum.GTE.value:
                outcome = left_value >= right_value
            elif binary_condition_type == ConditionTypeEnum.LT.value:
                outcome = left_value < right_value
            elif binary_condition_type == ConditionTypeEnum.LTE.value:
                outcome = left_value <= right_value
            else:
                raise NotImplementedError(
                    f"Binary condition of type [{binary_condition_type}] is not supported"
                )
            return outcome

        except TypeError:
            self.pipline_executor.execution.update_step_failure(
                self.step.name,
                f"Condition of type [{binary_condition_type}] not supported between "
                + f"[{left_value}] of type [{type(left_value)}] and [{right_value}] "
                + f"of type [{type(right_value)}]",
            )

    def _resolve_not_condition(self, not_condition: dict):
        """Resolve given ConditionNot.

        Args:
            Dictionary representing given ConditionNot as request

        Returns:
            True if given ConditionNot evaluated as true,
            False otherwise.
        """
        return not self._resolve_condition(not_condition["Expression"])

    def _resolve_or_condition(self, or_condition: dict):
        """Resolve given ConditionOr.

        Args:
            Dictionary representing given ConditionOr as request

        Returns:
            True if given ConditionOr evaluated as true,
            False otherwise.
        """

        for condition in or_condition["Conditions"]:
            if self._resolve_condition(condition):
                return True
        return False

    def _resolve_in_condition(self, in_condition: dict):
        """Resolve given ConditionIn.

        Args:
            Dictionary representing given ConditionIn as request

        Returns:
            True if given ConditionIn evaluated as true,
            False otherwise.
        """

        query_value = in_condition["QueryValue"]
        values = in_condition["Values"]
        return query_value in values


class _TransformStepExecutor(_StepExecutor):
    """Executor class to execute TransformStep locally"""

    def execute(self):
        job_name = unique_name_from_base(self.step.name)
        step_arguments = self.pipline_executor.evaluate_step_arguments(self.step)
        try:
            self.pipline_executor.local_sagemaker_client.create_transform_job(
                job_name, **step_arguments
            )
            return self.pipline_executor.local_sagemaker_client.describe_transform_job(job_name)
        except Exception as e:  # pylint: disable=W0703
            self.pipline_executor.execution.update_step_failure(
                self.step.name, f"{type(e).__name__}: {str(e)}"
            )


class _CreateModelStepExecutor(_StepExecutor):
    """Executor class to execute CreateModelStep locally"""

    def execute(self):
        model_name = unique_name_from_base(self.step.name)
        step_arguments = self.pipline_executor.evaluate_step_arguments(self.step)
        try:
            self.pipline_executor.local_sagemaker_client.create_model(model_name, **step_arguments)
            return self.pipline_executor.local_sagemaker_client.describe_model(model_name)
        except Exception as e:  # pylint: disable=W0703
            self.pipline_executor.execution.update_step_failure(
                self.step.name, f"{type(e).__name__}: {str(e)}"
            )


class _FailStepExecutor(_StepExecutor):
    """Executor class to execute FailStep locally"""

    def execute(self):
        step_arguments = self.pipline_executor.evaluate_step_arguments(self.step)

        error_message = step_arguments.get("ErrorMessage")
        self.pipline_executor.execution.update_step_properties(
            self.step.name, {"ErrorMessage": error_message}
        )
        self.pipline_executor.execution.update_step_failure(
            self.step.name, step_arguments.get("ErrorMessage")
        )


class _StepExecutorFactory:
    """Factory class to generate executors for given step based on their types"""

    def __init__(self, pipeline_executor: LocalPipelineExecutor):
        self.pipeline_executor = pipeline_executor

    def get(self, step: Step) -> _StepExecutor:
        """Return corresponding step executor for given step"""

        step_type = step.step_type
        step_executor = None
        if step_type == StepTypeEnum.TRAINING:
            step_executor = _TrainingStepExecutor(self.pipeline_executor, step)
        elif step_type == StepTypeEnum.PROCESSING:
            step_executor = _ProcessingStepExecutor(self.pipeline_executor, step)
        elif step_type == StepTypeEnum.TRANSFORM:
            step_executor = _TransformStepExecutor(self.pipeline_executor, step)
        elif step_type == StepTypeEnum.CREATE_MODEL:
            step_executor = _CreateModelStepExecutor(self.pipeline_executor, step)
        elif step_type == StepTypeEnum.FAIL:
            step_executor = _FailStepExecutor(self.pipeline_executor, step)
        elif step_type == StepTypeEnum.CONDITION:
            step_executor = _ConditionStepExecutor(self.pipeline_executor, step)
        else:
            self.pipeline_executor.execution.update_step_failure(
                step.name, f"Unsupported step type {step_type} to execute."
            )
        return step_executor
