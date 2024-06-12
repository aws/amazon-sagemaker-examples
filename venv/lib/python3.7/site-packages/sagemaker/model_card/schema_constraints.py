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
"""Schema constraints for model card attribute value."""
from __future__ import absolute_import

from enum import Enum


class ModelCardStatusEnum(str, Enum):
    """Model card status enumerator"""

    DRAFT = "Draft"
    PENDING_REVIEW = "PendingReview"
    APPROVED = "Approved"
    ARCHIVED = "Archived"


class RiskRatingEnum(str, Enum):
    """Risk rating enumerator"""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class ObjectiveFunctionEnum(str, Enum):
    """Objective function enumerator"""

    MAXIMIZE = "Maximize"
    MINIMIZE = "Minimize"


class FacetEnum(str, Enum):
    """Objective function facet enumerator"""

    LOSS = "Loss"
    ACCURACY = "Accuracy"
    RMSE = "RMSE"
    MAE = "MAE"
    AUC = "AUC"


class MetricTypeEnum(str, Enum):
    """Metric type enumerator"""

    NUMBER = "number"
    LINEAR_GRAPH = "linear_graph"
    STRING = "string"
    BOOLEAN = "boolean"
    MATRIX = "matrix"
    BAR_CHART = "bar_chart"


class ModelPackageStatusEnum(str, Enum):
    """Model package status enumerator"""

    PENDING = "Pending"
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    DELETING = "Deleting"


class ModelApprovalStatusEnum(str, Enum):
    """Model package approval status enumerator"""

    APPROVED = "Approved"
    REJECTED = "Rejected"
    PENDING_MANUAL_APPROVAL = "PendingManualApproval"


METRIC_VALUE_TYPE_MAP = {
    MetricTypeEnum.NUMBER: [int, float],
    MetricTypeEnum.LINEAR_GRAPH: [list],
    MetricTypeEnum.STRING: [str],
    MetricTypeEnum.BOOLEAN: [bool],
    MetricTypeEnum.MATRIX: [list],
    MetricTypeEnum.BAR_CHART: [list],
}


PYTHON_TYPE_TO_METRIC_VALUE_TYPE = {
    int: MetricTypeEnum.NUMBER,
    float: MetricTypeEnum.NUMBER,
    str: MetricTypeEnum.STRING,
    bool: MetricTypeEnum.BOOLEAN,
}

MODEL_ARTIFACT_MAX_SIZE = 15
ENVIRONMENT_CONTAINER_IMAGES_MAX_SIZE = 15
TRAINING_DATASETS_MAX_SIZE = 15
TRAINING_METRICS_MAX_SIZE = 50
USER_PROVIDED_TRAINING_METRICS_MAX_SIZE = 50
HYPER_PARAMETERS_MAX_SIZE = 100
USER_PROVIDED_HYPER_PARAMETERS_MAX_SIZE = 100
EVALUATION_DATASETS_MAX_SIZE = 10
SOURCE_ALGORITHMS_MAX_SIZE = 15
