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
"""Contains the _RunEnvironment class."""
from __future__ import absolute_import

import enum
import json
import logging
import os

from sagemaker import Session
from sagemaker.experiments import trial_component
from sagemaker.utils import retry_with_backoff

TRAINING_JOB_ARN_ENV = "TRAINING_JOB_ARN"
PROCESSING_JOB_CONFIG_PATH = "/opt/ml/config/processingjobconfig.json"
TRANSFORM_JOB_ARN_ENV = "TRANSFORM_JOB_ARN"
MAX_RETRY_ATTEMPTS = 7

logger = logging.getLogger(__name__)


class _EnvironmentType(enum.Enum):
    """SageMaker jobs which data can be pulled from the environment."""

    SageMakerTrainingJob = 1
    SageMakerProcessingJob = 2
    SageMakerTransformJob = 3


class _RunEnvironment(object):
    """Retrieves job specific data from the environment."""

    def __init__(self, environment_type: _EnvironmentType, source_arn: str):
        """Init for _RunEnvironment.

        Args:
            environment_type (_EnvironmentType): The environment type.
            source_arn (str): The ARN of the current job.
        """
        self.environment_type = environment_type
        self.source_arn = source_arn

    @classmethod
    def load(
        cls,
        training_job_arn_env: str = TRAINING_JOB_ARN_ENV,
        processing_job_config_path: str = PROCESSING_JOB_CONFIG_PATH,
        transform_job_arn_env: str = TRANSFORM_JOB_ARN_ENV,
    ):
        """Loads source arn of current job from environment.

        Args:
            training_job_arn_env (str): The environment key for training job ARN
                (default: `TRAINING_JOB_ARN`).
            processing_job_config_path (str): The processing job config path
                (default: `/opt/ml/config/processingjobconfig.json`).
            transform_job_arn_env (str): The environment key for transform job ARN
                (default: `TRANSFORM_JOB_ARN_ENV`).

        Returns:
            _RunEnvironment: Job data loaded from the environment. None if config does not exist.
        """
        if training_job_arn_env in os.environ:
            environment_type = _EnvironmentType.SageMakerTrainingJob
            source_arn = os.environ.get(training_job_arn_env)
            return _RunEnvironment(environment_type, source_arn)
        if os.path.exists(processing_job_config_path):
            environment_type = _EnvironmentType.SageMakerProcessingJob
            source_arn = json.loads(open(processing_job_config_path).read())["ProcessingJobArn"]
            return _RunEnvironment(environment_type, source_arn)
        if transform_job_arn_env in os.environ:
            environment_type = _EnvironmentType.SageMakerTransformJob
            # TODO: need to update to get source_arn from config file once Transform side ready
            source_arn = os.environ.get(transform_job_arn_env)
            return _RunEnvironment(environment_type, source_arn)

        return None

    def get_trial_component(self, sagemaker_session: Session):
        """Retrieves the trial component from the job in the environment.

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            _TrialComponent: The trial component created from the job. None if not found.
        """

        def _get_trial_component():
            summaries = list(
                trial_component._TrialComponent.list(
                    source_arn=self.source_arn.lower(), sagemaker_session=sagemaker_session
                )
            )
            if summaries:
                summary = summaries[0]
                return trial_component._TrialComponent.load(
                    trial_component_name=summary.trial_component_name,
                    sagemaker_session=sagemaker_session,
                )
            return None

        job_tc = None
        try:
            job_tc = retry_with_backoff(_get_trial_component, MAX_RETRY_ATTEMPTS)
        except Exception as ex:  # pylint: disable=broad-except
            logger.error(
                "Failed to get trail component in the current environment due to %s", str(ex)
            )
        return job_tc
