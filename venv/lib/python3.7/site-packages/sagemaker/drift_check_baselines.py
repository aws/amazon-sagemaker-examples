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
"""This file contains code related to drift check baselines"""
from __future__ import absolute_import

from typing import Optional

from sagemaker.model_metrics import MetricsSource, FileSource


class DriftCheckBaselines(object):
    """Accepts drift check baselines parameters for conversion to request dict."""

    def __init__(
        self,
        model_statistics: Optional[MetricsSource] = None,
        model_constraints: Optional[MetricsSource] = None,
        model_data_statistics: Optional[MetricsSource] = None,
        model_data_constraints: Optional[MetricsSource] = None,
        bias_config_file: Optional[FileSource] = None,
        bias_pre_training_constraints: Optional[MetricsSource] = None,
        bias_post_training_constraints: Optional[MetricsSource] = None,
        explainability_constraints: Optional[MetricsSource] = None,
        explainability_config_file: Optional[FileSource] = None,
    ):
        """Initialize a ``DriftCheckBaselines`` instance and turn parameters into dict.

        Args:
            model_statistics (MetricsSource): A metric source object that represents
                model statistics (default: None).
            model_constraints (MetricsSource): A metric source object that represents
                model constraints (default: None).
            model_data_statistics (MetricsSource): A metric source object that represents
                model data statistics (default: None).
            model_data_constraints (MetricsSource): A metric source object that represents
                model data constraints (default: None).
            bias_config_file (FileSource): A file source object that represents bias config
                (default: None).
            bias_pre_training_constraints (MetricsSource):
                A metric source object that represents Pre-training constraints (default: None).
            bias_post_training_constraints (MetricsSource):
                A metric source object that represents Post-training constraits (default: None).
            explainability_constraints (MetricsSource):
                A metric source object that represents explainability constraints (default: None).
            explainability_config_file (FileSource): A file source object that represents
                explainability config (default: None).
        """
        self.model_statistics = model_statistics
        self.model_constraints = model_constraints
        self.model_data_statistics = model_data_statistics
        self.model_data_constraints = model_data_constraints
        self.bias_config_file = bias_config_file
        self.bias_pre_training_constraints = bias_pre_training_constraints
        self.bias_post_training_constraints = bias_post_training_constraints
        self.explainability_constraints = explainability_constraints
        self.explainability_config_file = explainability_config_file

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        drift_check_baselines_request = {}

        model_quality = {}
        if self.model_statistics is not None:
            model_quality["Statistics"] = self.model_statistics._to_request_dict()
        if self.model_constraints is not None:
            model_quality["Constraints"] = self.model_constraints._to_request_dict()
        if model_quality:
            drift_check_baselines_request["ModelQuality"] = model_quality

        model_data_quality = {}
        if self.model_data_statistics is not None:
            model_data_quality["Statistics"] = self.model_data_statistics._to_request_dict()
        if self.model_data_constraints is not None:
            model_data_quality["Constraints"] = self.model_data_constraints._to_request_dict()
        if model_data_quality:
            drift_check_baselines_request["ModelDataQuality"] = model_data_quality

        bias = {}
        if self.bias_config_file is not None:
            bias["ConfigFile"] = self.bias_config_file._to_request_dict()
        if self.bias_pre_training_constraints is not None:
            bias["PreTrainingConstraints"] = self.bias_pre_training_constraints._to_request_dict()
        if self.bias_post_training_constraints is not None:
            bias["PostTrainingConstraints"] = self.bias_post_training_constraints._to_request_dict()
        if bias:
            drift_check_baselines_request["Bias"] = bias

        explainability = {}
        if self.explainability_constraints is not None:
            explainability["Constraints"] = self.explainability_constraints._to_request_dict()
        if self.explainability_config_file is not None:
            explainability["ConfigFile"] = self.explainability_config_file._to_request_dict()
        if explainability:
            drift_check_baselines_request["Explainability"] = explainability

        return drift_check_baselines_request
