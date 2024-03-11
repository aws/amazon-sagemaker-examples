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
"""Classes for using Scikit-Learn with Amazon SageMaker."""
from __future__ import absolute_import

from sagemaker.sklearn.estimator import SKLearn  # noqa: F401
from sagemaker.sklearn.model import SKLearnModel, SKLearnPredictor  # noqa: F401
from sagemaker.sklearn.processing import SKLearnProcessor  # noqa: F401
