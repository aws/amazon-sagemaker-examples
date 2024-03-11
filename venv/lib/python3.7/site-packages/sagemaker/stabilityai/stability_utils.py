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
"""Utility functions."""

from __future__ import absolute_import

from typing import Optional

from sagemaker import image_uris
from sagemaker.session import Session


def get_stabilityai_image_uri(
    session: Optional[Session] = None,
    region: Optional[str] = None,
    version: Optional[str] = None,
    image_scope: Optional[str] = "inference",
) -> str:
    """Very basic utility function to fetch image URI of StabilityAI images.

    Args:
        session (Session): SageMaker session.
        region (str): AWS region of image URI.
        version (str): Framework version. Latest version used if not specified.
        image_scope (str): Image type. e.g. inference, training
    Returns:
        Image URI string.
    """

    if region is None:
        if session is None:
            region = Session().boto_session.region_name
        else:
            region = session.boto_session.region_name
    return image_uris.retrieve(
        framework="stabilityai",
        region=region,
        version=version,
        image_scope=image_scope,
    )
