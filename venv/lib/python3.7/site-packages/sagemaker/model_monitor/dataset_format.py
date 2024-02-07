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
"""This module contains code related to the DatasetFormat class.

Codes are used for managing the constraints JSON file generated and consumed by Amazon SageMaker
Model Monitoring Schedules.
"""
from __future__ import print_function, absolute_import


class DatasetFormat(object):
    """Represents a Dataset Format that is used when calling a DefaultModelMonitor."""

    @staticmethod
    def csv(header=True, output_columns_position="START"):
        """Returns a DatasetFormat JSON string for use with a DefaultModelMonitor.

        Args:
            header (bool): Whether the csv dataset to baseline and monitor has a header.
                Default: True.
            output_columns_position (str): The position of the output columns.
                Must be one of ("START", "END"). Default: "START".

        Returns:
            dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.

        """
        return {"csv": {"header": header, "output_columns_position": output_columns_position}}

    @staticmethod
    def json(lines=True):
        """Returns a DatasetFormat JSON string for use with a DefaultModelMonitor.

        Args:
            lines (bool): Whether the file should be read as a json object per line. Default: True.

        Returns:
            dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.

        """
        return {"json": {"lines": lines}}

    @staticmethod
    def sagemaker_capture_json():
        """Returns a DatasetFormat SageMaker Capture Json string for use with a DefaultModelMonitor.

        Returns:
            dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.

        """
        return {"sagemakerCaptureJson": {}}


class MonitoringDatasetFormat(object):
    """Represents a Dataset Format that is used when calling a DefaultModelMonitor."""

    @staticmethod
    def csv(header=True):
        """Returns a DatasetFormat JSON string for use with a DefaultModelMonitor.

        Args:
            header (bool): Whether the csv dataset to baseline and monitor has a header.
                Default: True.

        Returns:
            dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.

        """
        return {"Csv": {"Header": header}}

    @staticmethod
    def json(lines=True):
        """Returns a DatasetFormat JSON string for use with a DefaultModelMonitor.

        Args:
            lines (bool): Whether the file should be read as a json object per line. Default: True.

        Returns:
            dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.

        """
        return {"Json": {"Line": lines}}

    @staticmethod
    def parquet():
        """Returns a DatasetFormat SageMaker Capture Json string for use with a DefaultModelMonitor.

        Returns:
            dict: JSON string containing DatasetFormat to be used by DefaultModelMonitor.

        """
        return {"Parquet": {}}
