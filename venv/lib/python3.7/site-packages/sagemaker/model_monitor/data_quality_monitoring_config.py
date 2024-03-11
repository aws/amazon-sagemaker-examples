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
"""This module contains code related to the MonitoringConfig of constraints file.

Code is used to represent the Monitoring Config object and its parameters suggested
in constraints file by Model Monitor Container in data quality analysis.
"""
from __future__ import print_function, absolute_import

CHI_SQUARED_METHOD = "ChiSquared"
L_INFINITY_METHOD = "LInfinity"


class DataQualityDistributionConstraints:
    """Represents the distribution_constraints object of monitoring_config in constraints file."""

    def __init__(self, categorical_drift_method: str = None):
        self.categorical_drift_method = categorical_drift_method

    @staticmethod
    def valid_distribution_constraints(distribution_constraints):
        """Checks whether distribution_constraints are valid or not."""

        if not distribution_constraints:
            return True

        return DataQualityDistributionConstraints.valid_categorical_drift_method(
            distribution_constraints.categorical_drift_method
        )

    @staticmethod
    def valid_categorical_drift_method(categorical_drift_method):
        """Checks whether categorical_drift_method is valid or not."""

        if not categorical_drift_method:
            return True

        return categorical_drift_method in [CHI_SQUARED_METHOD, L_INFINITY_METHOD]


class DataQualityMonitoringConfig:
    """Represents monitoring_config object in constraints file."""

    def __init__(self, distribution_constraints: DataQualityDistributionConstraints = None):
        self.distribution_constraints = distribution_constraints

    @staticmethod
    def valid_monitoring_config(monitoring_config):
        """Checks whether monitoring_config is valid or not."""

        if not monitoring_config:
            return True

        return DataQualityDistributionConstraints.valid_distribution_constraints(
            monitoring_config.distribution_constraints
        )
