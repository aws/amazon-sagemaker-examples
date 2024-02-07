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
"""Functions for generating ECR image URIs for pre-built SageMaker Docker images."""
from __future__ import absolute_import

from typing import Optional

from sagemaker import image_uris
from sagemaker.session import Session


def get_huggingface_llm_image_uri(
    backend: str,
    session: Optional[Session] = None,
    region: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    """Retrieves the image URI for inference.

    Args:
        backend (str): The backend to use. Valid values include "huggingface" and "lmi".
        session (Session): The SageMaker Session to use. (Default: None).
        region (str): The AWS region to use for image URI. (default: None).
        version (str): The framework version for which to retrieve an
            image URI. If no version is set, defaults to latest version. (default: None).

    Returns:
        str: The image URI string.
    """

    if region is None:
        if session is None:
            region = Session().boto_session.region_name
        else:
            region = session.boto_session.region_name
    if backend == "huggingface":
        return image_uris.retrieve(
            "huggingface-llm",
            region=region,
            version=version,
            image_scope="inference",
        )
    if backend == "lmi":
        version = version or "0.23.0"
        return image_uris.retrieve(framework="djl-deepspeed", region=region, version=version)
    raise ValueError("Unsupported backend: %s" % backend)
