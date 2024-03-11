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
"""Contains the SageMaker Experiment class."""
from __future__ import absolute_import

import time

from botocore.exceptions import ClientError

from sagemaker.apiutils import _base_types
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent


class Experiment(_base_types.Record):
    """An Amazon SageMaker experiment, which is a collection of related trials.

    New experiments are created by calling `experiments.experiment.Experiment.create`.
    Existing experiments can be reloaded by calling `experiments.experiment.Experiment.load`.

    Attributes:
        experiment_name (str): The name of the experiment. The name must be unique
            within an account.
        display_name (str): Name of the experiment that will appear in UI,
            such as SageMaker Studio.
        description (str): A description of the experiment.
        tags (List[Dict[str, str]]): A list of tags to associate with the experiment.
    """

    experiment_name = None
    display_name = None
    description = None
    tags = None

    _boto_create_method = "create_experiment"
    _boto_load_method = "describe_experiment"
    _boto_update_method = "update_experiment"
    _boto_delete_method = "delete_experiment"

    _boto_update_members = ["experiment_name", "description", "display_name"]
    _boto_delete_members = ["experiment_name"]

    _MAX_DELETE_ALL_ATTEMPTS = 3

    def save(self):
        """Save the state of this Experiment to SageMaker.

        Returns:
            dict: Update experiment API response.
        """
        return self._invoke_api(self._boto_update_method, self._boto_update_members)

    def delete(self):
        """Delete this Experiment from SageMaker.

        Deleting an Experiment does not delete associated Trials and their Trial Components.
        It requires that each Trial in the Experiment is first deleted.

        Returns:
            dict: Delete experiment API response.
        """
        return self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    @classmethod
    def load(cls, experiment_name, sagemaker_session=None):
        """Load an existing experiment and return an `Experiment` object representing it.

        Args:
            experiment_name: (str): Name of the experiment
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            experiments.experiment.Experiment: A SageMaker `Experiment` object
        """
        return cls._construct(
            cls._boto_load_method,
            experiment_name=experiment_name,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def create(
        cls,
        experiment_name,
        display_name=None,
        description=None,
        tags=None,
        sagemaker_session=None,
    ):
        """Create a new experiment in SageMaker and return an `Experiment` object.

        Args:
            experiment_name: (str): Name of the experiment. Must be unique. Required.
            display_name: (str): Name of the experiment that will appear in UI,
                such as SageMaker Studio (default: None).
            description: (str): Description of the experiment (default: None).
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
            tags (List[Dict[str, str]]): A list of tags to associate with the experiment
                (default: None).

        Returns:
            experiments.experiment.Experiment: A SageMaker `Experiment` object
        """
        return cls._construct(
            cls._boto_create_method,
            experiment_name=experiment_name,
            display_name=display_name,
            description=description,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def _load_or_create(
        cls,
        experiment_name,
        display_name=None,
        description=None,
        tags=None,
        sagemaker_session=None,
    ):
        """Load an experiment by name and create a new one if it does not exist.

        Args:
            experiment_name: (str): Name of the experiment. Must be unique. Required.
            display_name: (str): Name of the experiment that will appear in UI,
                such as SageMaker Studio (default: None). This is used only when the
                given `experiment_name` does not exist and a new experiment has to be created.
            description: (str): Description of the experiment (default: None).
                This is used only when the given `experiment_name` does not exist and
                a new experiment has to be created.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
            tags (List[Dict[str, str]]): A list of tags to associate with the experiment
                (default: None). This is used only when the given `experiment_name` does not
                exist and a new experiment has to be created.

        Returns:
            experiments.experiment.Experiment: A SageMaker `Experiment` object
        """
        try:
            experiment = Experiment.create(
                experiment_name=experiment_name,
                display_name=display_name,
                description=description,
                tags=tags,
                sagemaker_session=sagemaker_session,
            )
        except ClientError as ce:
            error_code = ce.response["Error"]["Code"]
            error_message = ce.response["Error"]["Message"]
            if not (error_code == "ValidationException" and "already exists" in error_message):
                raise ce
            # already exists
            experiment = Experiment.load(experiment_name, sagemaker_session)
        return experiment

    def list_trials(self, created_before=None, created_after=None, sort_by=None, sort_order=None):
        """List trials in this experiment matching the specified criteria.

        Args:
            created_before (datetime.datetime): Return trials created before this instant
                (default: None).
            created_after (datetime.datetime): Return trials created after this instant
                (default: None).
            sort_by (str): Which property to sort results by. One of 'Name', 'CreationTime'
                (default: None).
            sort_order (str): One of 'Ascending', or 'Descending' (default: None).

        Returns:
            collections.Iterator[experiments._api_types.TrialSummary] :
                An iterator over trials matching the criteria.
        """
        return _Trial.list(
            experiment_name=self.experiment_name,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            sagemaker_session=self.sagemaker_session,
        )

    def _delete_all(self, action):
        """Force to delete the experiment and associated trials, trial components.

        Args:
            action (str): The string '--force' is required to pass in to confirm recursively
                delete the experiments, and all its trials and trial components.
        """
        if action != "--force":
            raise ValueError(
                "Must confirm with string '--force' in order to delete the experiment and "
                "associated trials, trial components."
            )

        delete_attempt_count = 0
        last_exception = None
        while True:
            if delete_attempt_count == self._MAX_DELETE_ALL_ATTEMPTS:
                raise Exception("Failed to delete, please try again.") from last_exception
            try:
                for trial_summary in self.list_trials():
                    trial = _Trial.load(
                        sagemaker_session=self.sagemaker_session,
                        trial_name=trial_summary.trial_name,
                    )
                    for (
                        trial_component_summary
                    ) in trial.list_trial_components():  # pylint: disable=no-member
                        tc = _TrialComponent.load(
                            sagemaker_session=self.sagemaker_session,
                            trial_component_name=trial_component_summary.trial_component_name,
                        )
                        tc.delete(force_disassociate=True)
                        # to prevent throttling
                        time.sleep(1.2)
                    trial.delete()  # pylint: disable=no-member
                    # to prevent throttling
                    time.sleep(1.2)
                self.delete()
                break
            except Exception as ex:  # pylint: disable=broad-except
                last_exception = ex
            finally:
                delete_attempt_count = delete_attempt_count + 1
