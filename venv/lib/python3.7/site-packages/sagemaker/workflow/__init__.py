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
"""Defines Types etc. used in workflow."""
from __future__ import absolute_import

from sagemaker.workflow.entities import Expression
from sagemaker.workflow.parameters import ParameterString


def is_pipeline_variable(var: object) -> bool:
    """Check if the variable is a pipeline variable

    Args:
        var (object): The variable to be verified.
    Returns:
         bool: True if it is, False otherwise.
    """

    # Currently Expression is on top of all kinds of pipeline variables
    # as well as PipelineExperimentConfigProperty and PropertyFile
    # TODO: We should deprecate the Expression and replace it with PipelineVariable
    return isinstance(var, Expression)


def is_pipeline_parameter_string(var: object) -> bool:
    """Check if the variable is a pipeline parameter string

    Args:
        var (object): The variable to be verified.
    Returns:
         bool: True if it is, False otherwise.
    """
    return isinstance(var, ParameterString)
