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
"""This file contains code related to model metrics, including metric source and file source."""
from __future__ import absolute_import

from typing import Optional, Union

from sagemaker.workflow.entities import PipelineVariable


class ModelMetrics(object):
    """Accepts model metrics parameters for conversion to request dict."""

    def __init__(
        self,
        model_statistics: Optional["MetricsSource"] = None,
        model_constraints: Optional["MetricsSource"] = None,
        model_data_statistics: Optional["MetricsSource"] = None,
        model_data_constraints: Optional["MetricsSource"] = None,
        bias: Optional["MetricsSource"] = None,
        explainability: Optional["MetricsSource"] = None,
        bias_pre_training: Optional["MetricsSource"] = None,
        bias_post_training: Optional["MetricsSource"] = None,
    ):
        """Initialize a ``ModelMetrics`` instance and turn parameters into dict.

        Args:
            model_statistics (MetricsSource): A metric source object that represents
                model statistics (default: None).
            model_constraints (MetricsSource): A metric source object that represents
                model constraints (default: None).
            model_data_statistics (MetricsSource): A metric source object that represents
                model data statistics (default: None).
            model_data_constraints (MetricsSource): A metric source object that represents
                model data constraints (default: None).
            bias (MetricsSource): A metric source object that represents bias report
                (default: None).
            explainability (MetricsSource): A metric source object that represents
                explainability report (default: None).
            bias_pre_training (MetricsSource): A metric source object that represents
                Pre-training report (default: None).
            bias_post_training (MetricsSource): A metric source object that represents
                Post-training report (default: None).
        """
        self.model_statistics = model_statistics
        self.model_constraints = model_constraints
        self.model_data_statistics = model_data_statistics
        self.model_data_constraints = model_data_constraints
        self.bias = bias
        self.bias_pre_training = bias_pre_training
        self.bias_post_training = bias_post_training
        self.explainability = explainability

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        model_metrics_request = {}

        model_quality = {}
        if self.model_statistics is not None:
            model_quality["Statistics"] = self.model_statistics._to_request_dict()
        if self.model_constraints is not None:
            model_quality["Constraints"] = self.model_constraints._to_request_dict()
        if model_quality:
            model_metrics_request["ModelQuality"] = model_quality

        model_data_quality = {}
        if self.model_data_statistics is not None:
            model_data_quality["Statistics"] = self.model_data_statistics._to_request_dict()
        if self.model_data_constraints is not None:
            model_data_quality["Constraints"] = self.model_data_constraints._to_request_dict()
        if model_data_quality:
            model_metrics_request["ModelDataQuality"] = model_data_quality

        bias = {}
        if self.bias is not None:
            bias["Report"] = self.bias._to_request_dict()
        if self.bias_pre_training is not None:
            bias["PreTrainingReport"] = self.bias_pre_training._to_request_dict()
        if self.bias_post_training is not None:
            bias["PostTrainingReport"] = self.bias_post_training._to_request_dict()
        model_metrics_request["Bias"] = bias

        explainability = {}
        if self.explainability is not None:
            explainability["Report"] = self.explainability._to_request_dict()
        model_metrics_request["Explainability"] = explainability

        return model_metrics_request


class MetricsSource(object):
    """Accepts metrics source parameters for conversion to request dict."""

    def __init__(
        self,
        content_type: Union[str, PipelineVariable],
        s3_uri: Union[str, PipelineVariable],
        content_digest: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Initialize a ``MetricsSource`` instance and turn parameters into dict.

        Args:
            content_type (str or PipelineVariable): Specifies the type of content
                in S3 URI
            s3_uri (str or PipelineVariable): The S3 URI of the metric
            content_digest (str or PipelineVariable): The digest of the metric
                (default: None)
        """
        self.content_type = content_type
        self.s3_uri = s3_uri
        self.content_digest = content_digest

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        metrics_source_request = {"ContentType": self.content_type, "S3Uri": self.s3_uri}
        if self.content_digest is not None:
            metrics_source_request["ContentDigest"] = self.content_digest
        return metrics_source_request


class FileSource(object):
    """Accepts file source parameters for conversion to request dict."""

    def __init__(
        self,
        s3_uri: Union[str, PipelineVariable],
        content_digest: Optional[Union[str, PipelineVariable]] = None,
        content_type: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Initialize a ``FileSource`` instance and turn parameters into dict.

        Args:
            s3_uri (str or PipelineVariable): The S3 URI of the metric
            content_digest (str or PipelineVariable): The digest of the metric
                (default: None)
            content_type (str or PipelineVariable): Specifies the type of content
                in S3 URI (default: None)
        """
        self.content_type = content_type
        self.s3_uri = s3_uri
        self.content_digest = content_digest

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        file_source_request = {"S3Uri": self.s3_uri}
        if self.content_digest is not None:
            file_source_request["ContentDigest"] = self.content_digest
        if self.content_type is not None:
            file_source_request["ContentType"] = self.content_type
        return file_source_request
