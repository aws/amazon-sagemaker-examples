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
"""The `MonitorBatchTransform` definition for SageMaker Pipelines Workflows"""
from __future__ import absolute_import
import logging
from typing import Union, Optional, List

from sagemaker.session import Session
from sagemaker.workflow.pipeline_context import _JobStepArguments
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.quality_check_step import (
    QualityCheckStep,
    QualityCheckConfig,
    DataQualityCheckConfig,
)
from sagemaker.workflow.clarify_check_step import (
    ClarifyCheckStep,
    ClarifyCheckConfig,
    ModelExplainabilityCheckConfig,
)
from sagemaker.workflow.steps import Step
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.steps import TransformStep
from sagemaker.workflow.utilities import validate_step_args_input


class MonitorBatchTransformStep(StepCollection):
    """Creates a Transformer step with Quality or Clarify check step

    Used to monitor the inputs and outputs of the batch transform job.
    """

    def __init__(
        self,
        name: str,
        transform_step_args: _JobStepArguments,
        monitor_configuration: Union[QualityCheckConfig, ClarifyCheckConfig],
        check_job_configuration: CheckJobConfig,
        monitor_before_transform: bool = False,
        fail_on_violation: Union[bool, PipelineVariable] = True,
        supplied_baseline_statistics: Union[str, PipelineVariable] = None,
        supplied_baseline_constraints: Union[str, PipelineVariable] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Construct a step collection of `TransformStep`, `QualityCheckStep` or `ClarifyCheckStep`

        Args:
            name (str): The name of the `MonitorBatchTransformStep`.
                The corresponding transform step will be named `{name}-transform`;
                and the corresponding check step will be named `{name}-monitoring`
            transform_step_args (_JobStepArguments): the transform step transform arguments.
            monitor_configuration (Union[
                `sagemaker.workflow.quality_check_step.QualityCheckConfig`,
                `sagemaker.workflow.quality_check_step.ClarifyCheckConfig`
            ]): the monitoring configuration used for run model monitoring.
            check_job_configuration (`sagemaker.workflow.check_job_config.CheckJobConfig`):
                the check job (processing job) cluster resource configuration.
            monitor_before_transform (bool): If to run data quality or model explainability
                monitoring type, a true value of this flag indicates
                running the check step before the transform job.
            fail_on_violation (Union[bool, PipelineVariable]): A opt-out flag to not to fail the
                check step when a violation is detected.
            supplied_baseline_statistics (Union[str, PipelineVariable]): The S3 path
                to the supplied statistics object representing the statistics JSON file
                which will be used for drift to check (default: None).
            supplied_baseline_constraints (Union[str, PipelineVariable]): The S3 path
                to the supplied constraints object representing the constraints JSON file
                which will be used for drift to check (default: None).
            display_name (str): The display name of the `MonitorBatchTransformStep`.
                The display name provides better UI readability.
                The corresponding transform step will be
                named `{display_name}-transform`;  and the corresponding check step
                will be named `{display_name}-monitoring` (default: None).
            description (str): The description of the `MonitorBatchTransformStep` (default: None).
        """
        self.name: str = name
        self.steps: List[Step] = []

        validate_step_args_input(
            step_args=transform_step_args,
            expected_caller={
                Session.transform.__name__,
            },
            error_message="The transform_step_args of MonitorBatchTransformStep"
            "must be obtained from transformer.transform()",
        )
        transform_step = TransformStep(
            name=f"{name}-transform",
            display_name=f"{display_name}-transform" if display_name else None,
            description=description,
            step_args=transform_step_args,
        )

        self.steps.append(transform_step)

        monitoring_step_name = f"{name}-monitoring"
        monitoring_step_display_name = f"{display_name}-monitoring" if display_name else None
        if isinstance(monitor_configuration, QualityCheckConfig):
            monitoring_step = QualityCheckStep(
                name=monitoring_step_name,
                display_name=monitoring_step_display_name,
                description=description,
                quality_check_config=monitor_configuration,
                check_job_config=check_job_configuration,
                skip_check=False,
                supplied_baseline_statistics=supplied_baseline_statistics,
                supplied_baseline_constraints=supplied_baseline_constraints,
                fail_on_violation=fail_on_violation,
            )
        elif isinstance(monitor_configuration, ClarifyCheckConfig):
            if supplied_baseline_statistics:
                logging.warning(
                    "supplied_baseline_statistics will be ignored if monitor_configuration "
                    "is a ClarifyCheckConfig"
                )
            monitoring_step = ClarifyCheckStep(
                name=monitoring_step_name,
                display_name=monitoring_step_display_name,
                description=description,
                clarify_check_config=monitor_configuration,
                check_job_config=check_job_configuration,
                skip_check=False,
                supplied_baseline_constraints=supplied_baseline_constraints,
                fail_on_violation=fail_on_violation,
            )
        else:
            raise ValueError(
                f"Unrecognized monitoring configuration: {monitor_configuration}"
                f"Should be an instance of either QualityCheckConfig or ClarifyCheckConfig"
            )

        self.steps.append(monitoring_step)

        if monitor_before_transform and not (
            isinstance(
                monitor_configuration, (DataQualityCheckConfig, ModelExplainabilityCheckConfig)
            )
        ):
            raise ValueError(
                "monitor_before_transform only take effect when the monitor_configuration "
                "is one of [DataQualityCheckConfig, ModelExplainabilityCheckConfig]"
            )

        if monitor_before_transform:
            transform_step.add_depends_on([monitoring_step])
        else:
            monitoring_step.add_depends_on([transform_step])
