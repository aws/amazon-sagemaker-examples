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
"""Contains the Trial class."""
from __future__ import absolute_import

from botocore.exceptions import ClientError

from sagemaker.apiutils import _base_types
from sagemaker.experiments import _api_types
from sagemaker.experiments.trial_component import _TrialComponent


class _Trial(_base_types.Record):
    """An execution of a data-science workflow with an experiment.

    Consists of a list of trial component objects, which document individual
    activities within the workflow.

    Attributes:
        trial_name (str): The name of the trial.
        experiment_name (str): The name of the trial's experiment.
        display_name (str): The name of the trial that will appear in UI,
            such as SageMaker Studio.
        tags (List[Dict[str, str]]): A list of tags to associate with the trial.
    """

    trial_name = None
    experiment_name = None
    display_name = None
    tags = None

    _boto_create_method = "create_trial"
    _boto_load_method = "describe_trial"
    _boto_delete_method = "delete_trial"
    _boto_update_method = "update_trial"

    _boto_update_members = ["trial_name", "display_name"]
    _boto_delete_members = ["trial_name"]

    @classmethod
    def _boto_ignore(cls):
        """Response fields to ignore by default."""
        return super(_Trial, cls)._boto_ignore() + ["CreatedBy"]

    def save(self):
        """Save the state of this Trial to SageMaker.

        Returns:
            dict: Update trial response.
        """
        return self._invoke_api(self._boto_update_method, self._boto_update_members)

    def delete(self):
        """Delete this Trial from SageMaker.

        Does not delete associated Trial Components.

        Returns:
            dict: Delete trial response.
        """
        return self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    @classmethod
    def load(cls, trial_name, sagemaker_session=None):
        """Load an existing trial and return a `_Trial` object.

        Args:
            trial_name: (str): Name of the Trial.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            experiments.trial._Trial: A SageMaker `_Trial` object
        """
        return super(_Trial, cls)._construct(
            cls._boto_load_method,
            trial_name=trial_name,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def create(
        cls, experiment_name, trial_name, display_name=None, tags=None, sagemaker_session=None
    ):
        """Create a new trial and return a `_Trial` object.

        Args:
            experiment_name: (str): Name of the experiment to create this trial in.
            trial_name: (str): Name of the Trial.
            display_name (str): Name of the trial that will appear in UI,
                such as SageMaker Studio (default: None).
            tags (List[dict]): A list of tags to associate with the trial (default: None).
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            experiments.trial._Trial: A SageMaker `_Trial` object
        """
        trial = super(_Trial, cls)._construct(
            cls._boto_create_method,
            trial_name=trial_name,
            experiment_name=experiment_name,
            display_name=display_name,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )
        return trial

    @classmethod
    def list(
        cls,
        experiment_name=None,
        trial_component_name=None,
        created_before=None,
        created_after=None,
        sort_by=None,
        sort_order=None,
        sagemaker_session=None,
    ):
        """List all trials matching the specified criteria.

        Args:
            experiment_name (str): Name of the experiment. If specified, only trials in
                the experiment will be returned (default: None).
            trial_component_name (str): Name of the trial component. If specified, only
                trials with this trial component name will be returned (default: None).
            created_before (datetime.datetime): Return trials created before this instant
                (default: None).
            created_after (datetime.datetime): Return trials created after this instant
                (default: None).
            sort_by (str): Which property to sort results by. One of 'Name', 'CreationTime'
                (default: None).
            sort_order (str): One of 'Ascending', or 'Descending' (default: None).
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
        Returns:
            collections.Iterator[experiments._api_types.TrialSummary]: An iterator over trials
                matching the specified criteria.
        """
        return super(_Trial, cls)._list(
            "list_trials",
            _api_types.TrialSummary.from_boto,
            "TrialSummaries",
            experiment_name=experiment_name,
            trial_component_name=trial_component_name,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            sagemaker_session=sagemaker_session,
        )

    def add_trial_component(self, trial_component):
        """Add the specified trial component to this trial.

        A trial component may belong to many trials and a trial may have many trial components.

        Args:
            trial_component (str or _TrialComponent): The trial component to add.
                Can be one of a _TrialComponent instance, or a string containing
                the name of the trial component to add.
        """
        if isinstance(trial_component, _TrialComponent):
            trial_component_name = trial_component.trial_component_name
        elif isinstance(trial_component, str):
            trial_component_name = trial_component
        else:
            raise TypeError(
                "Unsupported type of trail component {}. "
                "It has to be one type of _TrialComponent or str".format(trial_component)
            )
        self.sagemaker_session.sagemaker_client.associate_trial_component(
            TrialName=self.trial_name, TrialComponentName=trial_component_name
        )

    def remove_trial_component(self, trial_component):
        """Remove the specified trial component from this trial.

        Args:
            trial_component (str or _TrialComponent): The trial component to add.
                Can be one of a _TrialComponent instance, or a string containing
                the name of the trial component to add.
        """
        if isinstance(trial_component, _TrialComponent):
            trial_component_name = trial_component.trial_component_name
        elif isinstance(trial_component, str):
            trial_component_name = trial_component
        else:
            raise TypeError(
                "Unsupported type of trail component {}. "
                "It has to be one type of _TrialComponent or str".format(trial_component)
            )
        self.sagemaker_session.sagemaker_client.disassociate_trial_component(
            TrialName=self.trial_name, TrialComponentName=trial_component_name
        )

    def list_trial_components(
        self,
        created_before=None,
        created_after=None,
        sort_by=None,
        sort_order=None,
        max_results=None,
        next_token=None,
    ):
        """List trial components in this trial matching the specified criteria.

        Args:
            created_before (datetime.datetime): Return trials created before this instant
                (default: None).
            created_after (datetime.datetime): Return trials created after this instant
                (default: None).
            sort_by (str): Which property to sort results by. One of 'Name',
                'CreationTime' (default: None).
            sort_order (str): One of 'Ascending', or 'Descending' (default: None).
            max_results (int): maximum number of trial components to retrieve (default: None).
            next_token (str): token for next page of results (default: None).

        Returns:
            collections.Iterator[experiments._api_types.TrialComponentSummary] : An iterator over
                trials matching the criteria.
        """
        return _TrialComponent.list(
            trial_name=self.trial_name,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            max_results=max_results,
            next_token=next_token,
            sagemaker_session=self.sagemaker_session,
        )

    @classmethod
    def _load_or_create(
        cls, experiment_name, trial_name, display_name=None, tags=None, sagemaker_session=None
    ):
        """Load a trial by name and create a new one if it does not exist.

        Args:
            experiment_name: (str): Name of the experiment to create this trial in.
            trial_name: (str): Name of the Trial.
            display_name (str): Name of the trial that will appear in UI,
                such as SageMaker Studio (default: None). This is used only when the given
                `trial_name` does not exist and a new trial has to be created.
            tags (List[dict]): A list of tags to associate with the trial (default: None).
                This is used only when the given `trial_name` does not exist and
                a new trial has to be created.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            experiments.trial._Trial: A SageMaker `_Trial` object
        """
        try:
            trial = _Trial.create(
                experiment_name=experiment_name,
                trial_name=trial_name,
                display_name=display_name,
                tags=tags,
                sagemaker_session=sagemaker_session,
            )
        except ClientError as ce:
            error_code = ce.response["Error"]["Code"]
            error_message = ce.response["Error"]["Message"]
            if not (error_code == "ValidationException" and "already exists" in error_message):
                raise ce
            # already exists
            trial = _Trial.load(trial_name, sagemaker_session)
            if trial.experiment_name != experiment_name:  # pylint: disable=no-member
                raise ValueError(
                    "The given experiment_name {} ".format(experiment_name)
                    + "does not match that in the loaded trial {}".format(
                        trial.experiment_name  # pylint: disable=no-member
                    )
                )
        return trial
