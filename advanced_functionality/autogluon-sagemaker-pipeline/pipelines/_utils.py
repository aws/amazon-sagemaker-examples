# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Provides utilities for SageMaker Pipeline CLI."""
from __future__ import absolute_import

import ast


def get_pipeline_driver(module_name, passed_args=None):
    """Gets the driver for generating your pipeline definition.

    Pipeline modules must define a get_pipeline() module-level method.

    Args:
        module_name: The module name of your pipeline.
        passed_args: Optional passed arguments that your pipeline may be templated by.

    Returns:
        The SageMaker Workflow pipeline.
    """
    _imports = __import__(module_name, fromlist=["get_pipeline"])
    kwargs = convert_struct(passed_args)
    return _imports.get_pipeline(**kwargs)


def convert_struct(str_struct=None):
    return ast.literal_eval(str_struct) if str_struct else {}

def get_pipeline_custom_tags(module_name, args, tags):
    """Gets the custom tags for pipeline

    Returns:
        Custom tags to be added to the pipeline
    """
    try:
        _imports = __import__(module_name, fromlist=["get_pipeline_custom_tags"])
        kwargs = convert_struct(args)
        return _imports.get_pipeline_custom_tags(tags, kwargs['region'], kwargs['sagemaker_project_arn'])
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return tags
