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
"""Pipeline experiment config for SageMaker pipeline."""
from __future__ import absolute_import

from typing import Union

from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.execution_variables import ExecutionVariable
from sagemaker.workflow.entities import (
    Entity,
    Expression,
    RequestType,
)


class PipelineExperimentConfig(Entity):
    """Experiment config for SageMaker pipeline."""

    def __init__(
        self,
        experiment_name: Union[str, Parameter, ExecutionVariable, Expression],
        trial_name: Union[str, Parameter, ExecutionVariable, Expression],
    ):
        """Create a PipelineExperimentConfig

        Examples:
        Use pipeline name as the experiment name and pipeline execution id as the trial name::

            PipelineExperimentConfig(
                ExecutionVariables.PIPELINE_NAME, ExecutionVariables.PIPELINE_EXECUTION_ID)

        Use a customized experiment name and pipeline execution id as the trial name::

            PipelineExperimentConfig(
                'MyExperiment', ExecutionVariables.PIPELINE_EXECUTION_ID)

        Args:
            experiment_name (Union[str, Parameter, ExecutionVariable, Expression]):
                the name of the experiment that will be created.
            trial_name (Union[str, Parameter, ExecutionVariable, Expression]):
                the name of the trial that will be created.
        """
        self.experiment_name = experiment_name
        self.trial_name = trial_name

    def to_request(self) -> RequestType:
        """Returns: the request structure."""

        return {
            "ExperimentName": self.experiment_name,
            "TrialName": self.trial_name,
        }


class PipelineExperimentConfigProperty(Expression):
    """Reference to pipeline experiment config property."""

    def __init__(self, name: str):
        """Create a reference to pipeline experiment property.

        Args:
            name (str): The name of the pipeline experiment config property.
        """
        super(PipelineExperimentConfigProperty, self).__init__()
        self.name = name

    @property
    def expr(self) -> RequestType:
        """The 'Get' expression dict for a pipeline experiment config property."""

        return {"Get": f"PipelineExperimentConfig.{self.name}"}


class PipelineExperimentConfigProperties:
    """Enum-like class for all pipeline experiment config property references."""

    EXPERIMENT_NAME = PipelineExperimentConfigProperty("ExperimentName")
    TRIAL_NAME = PipelineExperimentConfigProperty("TrialName")
