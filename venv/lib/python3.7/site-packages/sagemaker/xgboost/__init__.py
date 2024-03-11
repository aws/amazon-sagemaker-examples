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
"""Classes for using XGBoost with Amazon SageMaker."""
from __future__ import absolute_import

from sagemaker.xgboost.defaults import XGBOOST_NAME  # noqa: F401
from sagemaker.xgboost.estimator import XGBoost  # noqa: F401
from sagemaker.xgboost.model import XGBoostModel, XGBoostPredictor  # noqa: F401
from sagemaker.xgboost.processing import XGBoostProcessor  # noqa: F401
