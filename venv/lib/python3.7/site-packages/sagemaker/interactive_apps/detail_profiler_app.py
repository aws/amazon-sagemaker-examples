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
"""Methods for starting up and accessing DetailProfiler apps

This module contains methods for starting up and accessing
DetailProfiler apps hosted on SageMaker
"""
from __future__ import absolute_import

import json
import logging
import os
import re

from typing import Optional
from sagemaker.session import Session, NOTEBOOK_METADATA_FILE

logger = logging.getLogger(__name__)


class DetailProfilerApp(object):
    """Class for creating/accessing a DetailProfiler app hosted on SageMaker."""

    def __init__(self, region: Optional[str] = None):
        """Initialize a DetailProfilerApp object.

        Args:
            region (str): The name of the region e.g. us-east-1. If not specified,
                one is created using the default AWS configuration chain.
        """
        if region:
            self.region = region
        else:
            try:
                self.region = Session().boto_region_name
            except ValueError:
                raise ValueError(
                    "Failed to get region from default config. Please eihter pass region "
                    "as an input argument or setup the local AWS config."
                )

        self._domain_id = None
        self._user_profile_name = None
        self._valid_domain_and_user = False
        self._get_domain_and_user()

    def __str__(self):
        """Return str(self)."""
        return f"DetailProfilerApp(region={self.region})"

    def __repr__(self):
        """Return repr(self)."""
        return self.__str__()

    def get_app_url(self, training_job_name: Optional[str] = None):
        """Get an unsigned URL for DetailProfiler app hosted on SageMaker.

           For users that are already in SM Studio notebook instance, the method tries to
           get domain id and user profile from the Studio environment. If succeeded, the
           generated URL will direct to SM DetailProfiler app. Otherwise it will direct to
           DetailProfiler landing page on SageMaker console. For non Studio users, the
           URL will direct to the DetailProfiler landing page on SageMaker console.

        Args:
            training_job_name (str): Optional. The name of the training job to pre-load in
                DetailProfiler.
                If not provided, no job will be automatically loaded when the URL is opened.

        Returns:
            str: An unsigned URL for DetailProfiler hosted on SageMaker.
        """
        if self._valid_domain_and_user:
            url = f"https://{self._domain_id}.studio.{self.region}.sagemaker.aws/profiler/default"
            if training_job_name is not None:
                self._validate_job_name(training_job_name)
                url += f"/#!/welcome?profile={training_job_name}"
        else:
            dp = "profiler-landing"
            url = "https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/{dp}".format(
                region=self.region, dp=dp
            )

        return url

    def _get_domain_and_user(self):
        """Validate studio domain id and user profile

        Get and validate studio domain id and user profile
        from NOTEBOOK_METADATA_FILE in studio environment.

        Set _valid_domain_and_user to True if validation succeeded.
        """
        if not os.path.isfile(NOTEBOOK_METADATA_FILE):
            return

        with open(NOTEBOOK_METADATA_FILE, "rb") as f:
            metadata = json.loads(f.read())
            self._domain_id = metadata.get("DomainId")
            self._user_profile_name = metadata.get("UserProfileName")
            if self._validate_domain_id() is True and self._validate_user_profile_name() is True:
                self._valid_domain_and_user = True
            else:
                logger.warning(
                    "NOTEBOOK_METADATA_FILE detected but failed to get"
                    "valid domain and user from it."
                )

    def _validate_job_name(self, job_name: str):
        """Validate training job name format."""
        job_name_regex = "^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}"
        if not re.fullmatch(job_name_regex, job_name):
            raise ValueError(
                f"Invalid job name. " f"Job name must match regular expression {job_name_regex}"
            )

    def _validate_domain_id(self):
        """Validate domain id format."""
        if self._domain_id is None or len(self._domain_id) > 63:
            return False
        return True

    def _validate_user_profile_name(self):
        """Validate user profile name format."""
        user_profile_name_regex = "^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}"
        if self._user_profile_name is None or not re.fullmatch(
            user_profile_name_regex, self._user_profile_name
        ):
            return False
        return True
