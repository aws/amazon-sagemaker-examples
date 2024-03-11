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
"""This module is used to define the Spark job config to remote function."""
from __future__ import absolute_import

from typing import Optional, List, Dict, Union
import attr
from sagemaker.spark.processing import SparkConfigUtils


def _validate_configuration(instance, attribute, configuration):
    # pylint: disable=unused-argument
    """This is the helper method to validate the spark configuration"""
    if configuration:
        SparkConfigUtils.validate_configuration(configuration=configuration)


def _validate_s3_uri(instance, attribute, s3_uri):
    # pylint: disable=unused-argument
    """This is the helper method to validate the s3 uri"""
    if s3_uri:
        SparkConfigUtils.validate_s3_uri(s3_uri)


@attr.s(frozen=True)
class SparkConfig:
    """This is the class to initialize the spark configurations for remote function

    Attributes:
        submit_jars (Optional[List[str]]): A list which contains paths to the jars which
            are going to be submitted to Spark job. The location can be a valid s3 uri or
            local path to the jar. Defaults to ``None``.
        submit_py_files (Optional[List[str]]): A list which contains paths to the python
            files which are going to be submitted to Spark job. The location can be a
            valid s3 uri or local path to the python file. Defaults to ``None``.
        submit_files (Optional[List[str]]): A list which contains paths to the files which
            are going to be submitted to Spark job. The location can be a valid s3 uri or
            local path to the python file. Defaults to ``None``.
        configuration (list[dict] or dict): Configuration for Hadoop, Spark, or Hive.
            List or dictionary of EMR-style classifications.
            https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-configure-apps.html
        spark_event_logs_s3_uri (str): S3 path where Spark application events will
            be published to.
    """

    submit_jars: Optional[List[str]] = attr.ib(default=None)
    submit_py_files: Optional[List[str]] = attr.ib(default=None)
    submit_files: Optional[List[str]] = attr.ib(default=None)
    configuration: Optional[Union[List[Dict], Dict]] = attr.ib(
        default=None, validator=_validate_configuration
    )
    spark_event_logs_uri: Optional[str] = attr.ib(default=None, validator=_validate_s3_uri)
