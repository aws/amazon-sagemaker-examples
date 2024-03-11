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
"""Contains class to store Transformation Code"""
from __future__ import absolute_import
from typing import Optional
import attr


@attr.s
class TransformationCode:
    """A Transformation Code definition for FeatureProcessor Lineage.

    Attributes:
        s3_uri (str): The S3 URI of the code.
        name (Optional[str]): The name of the code Artifact object.
        author (Optional[str]): The author of the code.
    """

    s3_uri: str = attr.ib()
    name: Optional[str] = attr.ib(default=None)
    author: Optional[str] = attr.ib(default=None)
