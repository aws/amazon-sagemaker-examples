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
"""Models that can be deployed to serverless compute."""
from __future__ import absolute_import
from sagemaker.deprecations import deprecated


@deprecated(sdk_version="v2.66.3")
class LambdaModel:
    """A model that can be deployed to Lambda.

    note:: Deprecated in versions >= v2.66.3. An alternative support will be added in near future.
    """
