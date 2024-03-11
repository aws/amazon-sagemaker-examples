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
"""This module stores parameters related to SageMaker JumpStart."""
from __future__ import absolute_import
import datetime

JUMPSTART_DEFAULT_MAX_S3_CACHE_ITEMS = 20
JUMPSTART_DEFAULT_MAX_SEMANTIC_VERSION_CACHE_ITEMS = 20
JUMPSTART_DEFAULT_S3_CACHE_EXPIRATION_HORIZON = datetime.timedelta(hours=6)
JUMPSTART_DEFAULT_SEMANTIC_VERSION_CACHE_EXPIRATION_HORIZON = datetime.timedelta(hours=6)
