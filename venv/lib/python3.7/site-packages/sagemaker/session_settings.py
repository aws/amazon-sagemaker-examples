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
"""Defines classes to parametrize a SageMaker ``Session``."""

from __future__ import absolute_import


class SessionSettings(object):
    """Optional container class for settings to apply to a SageMaker session."""

    def __init__(
        self,
        encrypt_repacked_artifacts=True,
        local_download_dir=None,
    ) -> None:
        """Initialize the ``SessionSettings`` of a SageMaker ``Session``.

        Args:
            encrypt_repacked_artifacts (bool): Flag to indicate whether to encrypt the artifacts
                at rest in S3 using the default AWS managed KMS key for S3 when a custom KMS key
                is not provided (Default: True).
            local_download_dir (str): Optional. A path specifying the local directory
                for downloading artifacts. (Default: None).
        """
        self._encrypt_repacked_artifacts = encrypt_repacked_artifacts
        self._local_download_dir = local_download_dir

    @property
    def encrypt_repacked_artifacts(self) -> bool:
        """Return True if repacked artifacts at rest in S3 should be encrypted by default."""
        return self._encrypt_repacked_artifacts

    @property
    def local_download_dir(self) -> str:
        """Return path specifying the local directory for downloading artifacts."""
        return self._local_download_dir
