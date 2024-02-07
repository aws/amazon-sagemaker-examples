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
"""Contains class that determines the current execution environment."""
from __future__ import absolute_import


from typing import Dict, Optional
from datetime import datetime, timezone
import json
import logging
import os
import attr
from sagemaker.feature_store.feature_processor._constants import (
    EXECUTION_TIME_PIPELINE_PARAMETER,
    EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT,
)


logger = logging.getLogger("sagemaker")


@attr.s
class EnvironmentHelper:
    """Helper class to retrieve info from environment.

    Attributes:
        current_time (datetime): The current datetime.
    """

    current_time = attr.ib(default=datetime.now(timezone.utc))

    def is_training_job(self) -> bool:
        """Determine if the current execution environment is inside a SageMaker Training Job"""
        return self.load_training_resource_config() is not None

    def get_instance_count(self) -> int:
        """Determine the number of instances for the current execution environment."""
        resource_config = self.load_training_resource_config()
        return len(resource_config["hosts"]) if resource_config else 1

    def load_training_resource_config(self) -> Optional[Dict]:
        """Load the contents of resourceconfig.json contents.

        Returns:
            Optional[Dict]: None if not found.
        """
        SM_TRAINING_CONFIG_FILE_PATH = "/opt/ml/input/config/resourceconfig.json"
        try:
            with open(SM_TRAINING_CONFIG_FILE_PATH, "r") as cfgfile:
                resource_config = json.load(cfgfile)
                logger.debug("Contents of %s: %s", SM_TRAINING_CONFIG_FILE_PATH, resource_config)
                return resource_config
        except FileNotFoundError:
            return None

    def get_job_scheduled_time(self) -> str:
        """Get the job scheduled time.

        Returns:
            str: Timestamp when the job is scheduled.
        """

        scheduled_time = self.current_time.strftime(EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT)
        if self.is_training_job():
            envs = dict(os.environ)
            return envs.get(EXECUTION_TIME_PIPELINE_PARAMETER, scheduled_time)

        return scheduled_time
