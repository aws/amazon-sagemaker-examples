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
"""Contains the SageMaker Experiment Run class."""
from __future__ import absolute_import

import datetime
import logging
from enum import Enum
from math import isnan, isinf
from numbers import Number
from typing import Optional, List, Dict, TYPE_CHECKING, Union

import dateutil
from numpy import array

from sagemaker.apiutils import _utils
from sagemaker.experiments import _api_types
from sagemaker.experiments._api_types import (
    TrialComponentArtifact,
    _TrialComponentStatusType,
)
from sagemaker.experiments._helper import (
    _ArtifactUploader,
    _LineageArtifactTracker,
    _DEFAULT_ARTIFACT_PREFIX,
)
from sagemaker.experiments._environment import _RunEnvironment
from sagemaker.experiments._run_context import _RunContext
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments._metrics import _MetricsManager
from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent

from sagemaker.utils import (
    get_module,
    unique_name_from_base,
)

from sagemaker.experiments._utils import (
    guess_media_type,
    resolve_artifact_name,
    verify_length_of_true_and_predicted,
    validate_invoked_inside_run_context,
    get_tc_and_exp_config_from_job_env,
    verify_load_input_names,
    is_run_trial_component,
)

if TYPE_CHECKING:
    from sagemaker import Session

logger = logging.getLogger(__name__)

RUN_NAME_BASE = "Sagemaker-Run".lower()
TRIAL_NAME_TEMPLATE = "Default-Run-Group-{}"
MAX_RUN_TC_ARTIFACTS_LEN = 30
MAX_NAME_LEN_IN_BACKEND = 120
EXPERIMENT_NAME = "ExperimentName"
TRIAL_NAME = "TrialName"
RUN_NAME = "RunName"
DELIMITER = "-"
RUN_TC_TAG_KEY = "sagemaker:trial-component-source"
RUN_TC_TAG_VALUE = "run"
RUN_TC_TAG = {"Key": RUN_TC_TAG_KEY, "Value": RUN_TC_TAG_VALUE}


class SortByType(Enum):
    """The type of property by which to sort the `list_runs` results."""

    CREATION_TIME = "CreationTime"
    NAME = "Name"


class SortOrderType(Enum):
    """The type of order to sort the list or search results."""

    ASCENDING = "Ascending"
    DESCENDING = "Descending"


class Run(object):
    """A collection of parameters, metrics, and artifacts to create a ML model."""

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        experiment_display_name: Optional[str] = None,
        run_display_name: Optional[str] = None,
        tags: Optional[List[Dict[str, str]]] = None,
        sagemaker_session: Optional["Session"] = None,
        artifact_bucket: Optional[str] = None,
        artifact_prefix: Optional[str] = None,
    ):
        """Construct a `Run` instance.

        SageMaker Experiments automatically tracks the inputs, parameters, configurations,
        and results of your iterations as runs.
        You can assign, group, and organize these runs into experiments.
        You can also create, compare, and evaluate runs.

        The code sample below shows how to initialize a run, log parameters to the Run object
        and invoke a training job under the context of this Run object, which automatically
        passes the run's ``experiment_config`` (including the experiment name, run name etc.)
        to the training job.

        Note:
            All log methods (e.g. ``log_parameter``, ``log_metric``, etc.) have to be called within
            the run context (i.e. the ``with`` statement). Otherwise, a ``RuntimeError`` is thrown.

        .. code:: python

            with Run(experiment_name="my-exp", run_name="my-run", ...) as run:
                run.log_parameter(...)
                ...
                estimator.fit(job_name="my-job")  # Create a training job

        In order to reuse an existing run to log extra data, ``load_run`` is recommended.
        For example, instead of the ``Run`` constructor, the ``load_run`` is recommended to use
        in a job script to load the existing run created before the job launch.
        Otherwise, a new run may be created each time you launch a job.

        The code snippet below displays how to load the run initialized above
        in a custom training job script, where no ``run_name`` or ``experiment_name``
        is presented as they are automatically retrieved from the experiment config
        in the job environment.

        .. code:: python

            with load_run(sagemaker_session=sagemaker_session) as run:
                run.log_metric(...)
                ...

        Args:
            experiment_name (str): The name of the experiment. The name must be unique
                within an account.
            run_name (str): The name of the run. If it is not specified, one is auto generated.
            experiment_display_name (str): Name of the experiment that will appear in UI,
                such as SageMaker Studio. (default: None). This display name is used in
                a create experiment call. If an experiment with the specified name already exists,
                this display name won't take effect.
            run_display_name (str): The display name of the run used in UI (default: None).
                This display name is used in a create run call. If a run with the
                specified name already exists, this display name won't take effect.
            tags (List[Dict[str, str]]): A list of tags to be used for all create calls,
                e.g. to create an experiment, a run group, etc. (default: None).
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
            artifact_bucket (str): The S3 bucket to upload the artifact to.
                If not specified, the default bucket defined in `sagemaker_session`
                will be used.
            artifact_prefix (str): The S3 key prefix used to generate the S3 path
                to upload the artifact to (default: "trial-component-artifacts").
        """
        # TODO: we should revert the lower casting once backend fix reaches prod
        self.experiment_name = experiment_name.lower()
        sagemaker_session = sagemaker_session or _utils.default_session()
        self.run_name = run_name or unique_name_from_base(RUN_NAME_BASE)

        # avoid confusion due to mis-match in casing between run name and TC name
        self.run_name = self.run_name.lower()

        trial_component_name = Run._generate_trial_component_name(
            run_name=self.run_name, experiment_name=self.experiment_name
        )
        self.run_group_name = Run._generate_trial_name(self.experiment_name)

        self._experiment = Experiment._load_or_create(
            experiment_name=self.experiment_name,
            display_name=experiment_display_name,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

        self._trial = _Trial._load_or_create(
            experiment_name=self.experiment_name,
            trial_name=self.run_group_name,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

        self._trial_component, is_existed = _TrialComponent._load_or_create(
            trial_component_name=trial_component_name,
            display_name=run_display_name,
            tags=Run._append_run_tc_label_to_tags(tags),
            sagemaker_session=sagemaker_session,
        )
        if is_existed:
            logger.info(
                "The run (%s) under experiment (%s) already exists. Loading it.",
                self.run_name,
                self.experiment_name,
            )

        if not _TrialComponent._trial_component_is_associated_to_trial(
            self._trial_component.trial_component_name, self._trial.trial_name, sagemaker_session
        ):
            self._trial.add_trial_component(self._trial_component)

        self._artifact_uploader = _ArtifactUploader(
            trial_component_name=self._trial_component.trial_component_name,
            sagemaker_session=sagemaker_session,
            artifact_bucket=artifact_bucket,
            artifact_prefix=_DEFAULT_ARTIFACT_PREFIX
            if artifact_prefix is None
            else artifact_prefix,
        )
        self._lineage_artifact_tracker = _LineageArtifactTracker(
            trial_component_arn=self._trial_component.trial_component_arn,
            sagemaker_session=sagemaker_session,
        )
        self._metrics_manager = _MetricsManager(
            trial_component_name=self._trial_component.trial_component_name,
            sagemaker_session=sagemaker_session,
        )
        self._inside_init_context = False
        self._inside_load_context = False
        self._in_load = False

    @property
    def experiment_config(self) -> dict:
        """Get experiment config from run attributes."""
        return {
            EXPERIMENT_NAME: self.experiment_name,
            TRIAL_NAME: self.run_group_name,
            RUN_NAME: self._trial_component.trial_component_name,
        }

    @validate_invoked_inside_run_context
    def log_parameter(self, name: str, value: Union[str, int, float]):
        """Record a single parameter value for this run.

        Overwrites any previous value recorded for the specified parameter name.

        Args:
            name (str): The name of the parameter.
            value (str or int or float): The value of the parameter.
        """
        if self._is_input_valid("parameter", name, value):
            self._trial_component.parameters[name] = value

    @validate_invoked_inside_run_context
    def log_parameters(self, parameters: Dict[str, Union[str, int, float]]):
        """Record a collection of parameter values for this run.

        Args:
            parameters (dict[str, str or int or float]): The parameters to record.
        """
        filtered_parameters = {
            key: value
            for (key, value) in parameters.items()
            if self._is_input_valid("parameter", key, value)
        }
        self._trial_component.parameters.update(filtered_parameters)

    @validate_invoked_inside_run_context
    def log_metric(
        self,
        name: str,
        value: float,
        timestamp: Optional[datetime.datetime] = None,
        step: Optional[int] = None,
    ):
        """Record a custom scalar metric value for this run.

        Note:
             This method is for manual custom metrics, for automatic metrics see the
             ``enable_sagemaker_metrics`` parameter on the ``estimator`` class.

        Args:
            name (str): The name of the metric.
            value (float): The value of the metric.
            timestamp (datetime.datetime): The timestamp of the metric.
                If not specified, the current UTC time will be used.
            step (int): The integer iteration number of the metric value (default: None).
        """
        if self._is_input_valid("metric", name, value):
            self._metrics_manager.log_metric(
                metric_name=name, value=value, timestamp=timestamp, step=step
            )

    @validate_invoked_inside_run_context
    def log_precision_recall(
        self,
        y_true: Union[list, array],
        predicted_probabilities: Union[list, array],
        positive_label: Optional[Union[str, int]] = None,
        title: Optional[str] = None,
        is_output: bool = True,
        no_skill: Optional[int] = None,
    ):
        """Create and log a precision recall graph artifact for Studio UI to render.

        The artifact is stored in S3 and represented as a lineage artifact
        with an association with the run.

        You can view the artifact in the UI.
        If your job is created by a pipeline execution you can view the artifact
        by selecting the corresponding step in the pipelines UI.
        See also `SageMaker Pipelines <https://aws.amazon.com/sagemaker/pipelines/>`_

        This method requires sklearn library.

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            predicted_probabilities (list or array): Estimated/predicted probabilities.
            positive_label (str or int): Label of the positive class (default: None).
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.
            no_skill (int): The precision threshold under which the classifier cannot discriminate
                between the classes and would predict a random class or a constant class in
                all cases (default: None).
        """

        verify_length_of_true_and_predicted(
            true_labels=y_true,
            predicted_attrs=predicted_probabilities,
            predicted_attrs_name="predicted probabilities",
        )

        get_module("sklearn")
        from sklearn.metrics import precision_recall_curve, average_precision_score

        kwargs = {}
        if positive_label is not None:
            kwargs["pos_label"] = positive_label

        precision, recall, _ = precision_recall_curve(y_true, predicted_probabilities, **kwargs)

        kwargs["average"] = "micro"
        ap = average_precision_score(y_true, predicted_probabilities, **kwargs)

        data = {
            "type": "PrecisionRecallCurve",
            "version": 0,
            "title": title,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "averagePrecisionScore": ap,
            "noSkill": no_skill,
        }
        self._log_graph_artifact(
            artifact_name=title,
            data=data,
            graph_type="PrecisionRecallCurve",
            is_output=is_output,
        )

    @validate_invoked_inside_run_context
    def log_roc_curve(
        self,
        y_true: Union[list, array],
        y_score: Union[list, array],
        title: Optional[str] = None,
        is_output: bool = True,
    ):
        """Create and log a receiver operating characteristic (ROC curve) artifact.

        The artifact is stored in S3 and represented as a lineage artifact
        with an association with the run.

        You can view the artifact in the UI.
        If your job is created by a pipeline execution you can view the artifact
        by selecting the corresponding step in the pipelines UI.
        See also `SageMaker Pipelines <https://aws.amazon.com/sagemaker/pipelines/>`_

        This method requires sklearn library.

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            y_score (list or array): Estimated/predicted probabilities.
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        verify_length_of_true_and_predicted(
            true_labels=y_true,
            predicted_attrs=y_score,
            predicted_attrs_name="predicted scores",
        )

        get_module("sklearn")
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_score)

        auc = auc(fpr, tpr)

        data = {
            "type": "ROCCurve",
            "version": 0,
            "title": title,
            "falsePositiveRate": fpr.tolist(),
            "truePositiveRate": tpr.tolist(),
            "areaUnderCurve": auc,
        }
        self._log_graph_artifact(
            artifact_name=title, data=data, graph_type="ROCCurve", is_output=is_output
        )

    @validate_invoked_inside_run_context
    def log_confusion_matrix(
        self,
        y_true: Union[list, array],
        y_pred: Union[list, array],
        title: Optional[str] = None,
        is_output: bool = True,
    ):
        """Create and log a confusion matrix artifact.

        The artifact is stored in S3 and represented as a lineage artifact
        with an association with the run.

        You can view the artifact in the UI.
        If your job is created by a pipeline execution you can view the
        artifact by selecting the corresponding step in the pipelines UI.
        See also `SageMaker Pipelines <https://aws.amazon.com/sagemaker/pipelines/>`_
        This method requires sklearn library.

        Args:
            y_true (list or array): True labels. If labels are not binary
                then positive_label should be given.
            y_pred (list or array): Predicted labels.
            title (str): Title of the graph (default: None).
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        verify_length_of_true_and_predicted(
            true_labels=y_true,
            predicted_attrs=y_pred,
            predicted_attrs_name="predicted labels",
        )

        get_module("sklearn")
        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(y_true, y_pred)

        data = {
            "type": "ConfusionMatrix",
            "version": 0,
            "title": title,
            "confusionMatrix": matrix.tolist(),
        }
        self._log_graph_artifact(
            artifact_name=title,
            data=data,
            graph_type="ConfusionMatrix",
            is_output=is_output,
        )

    @validate_invoked_inside_run_context
    def log_artifact(
        self,
        name: str,
        value: str,
        media_type: Optional[str] = None,
        is_output: bool = True,
    ):
        """Record a single artifact for this run.

        Overwrites any previous value recorded for the specified name.

        Args:
            name (str): The name of the artifact.
            value (str): The value.
            media_type (str): The MediaType (MIME type) of the value (default: None).
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        self._verify_trial_component_artifacts_length(is_output=is_output)
        if is_output:
            self._trial_component.output_artifacts[name] = TrialComponentArtifact(
                value, media_type=media_type
            )
        else:
            self._trial_component.input_artifacts[name] = TrialComponentArtifact(
                value, media_type=media_type
            )

    @validate_invoked_inside_run_context
    def log_file(
        self,
        file_path: str,
        name: Optional[str] = None,
        media_type: Optional[str] = None,
        is_output: bool = True,
    ):
        """Upload a file to s3 and store it as an input/output artifact in this run.

        Args:
            file_path (str): The path of the local file to upload.
            name (str): The name of the artifact (default: None).
            media_type (str): The MediaType (MIME type) of the file.
                If not specified, this library will attempt to infer the media type
                from the file extension of ``file_path``.
            is_output (bool): Determines direction of association to the
                run. Defaults to True (output artifact).
                If set to False then represented as input association.
        """
        self._verify_trial_component_artifacts_length(is_output)
        media_type = media_type or guess_media_type(file_path)
        name = name or resolve_artifact_name(file_path)
        s3_uri, _ = self._artifact_uploader.upload_artifact(file_path)
        if is_output:
            self._trial_component.output_artifacts[name] = TrialComponentArtifact(
                value=s3_uri, media_type=media_type
            )
        else:
            self._trial_component.input_artifacts[name] = TrialComponentArtifact(
                value=s3_uri, media_type=media_type
            )

    def close(self):
        """Persist any data saved locally."""
        try:
            # Update the trial component with additions from the Run object
            self._trial_component.save()
            # Create Lineage entities for the artifacts
            self._lineage_artifact_tracker.save()
        finally:
            if self._metrics_manager:
                self._metrics_manager.close()

    @staticmethod
    def _generate_trial_name(base_name) -> str:
        """Generate the reserved trial name based on experiment name

        Args:
            base_name (str): The ``experiment_name`` of this ``Run`` object.
        """
        available_length = MAX_NAME_LEN_IN_BACKEND - len(TRIAL_NAME_TEMPLATE)
        return TRIAL_NAME_TEMPLATE.format(base_name[:available_length])

    @staticmethod
    def _is_input_valid(input_type, field_name, field_value) -> bool:
        """Check if the input is valid or not

        Args:
            input_type (str): The type of the input, one of ``parameter``, ``metric``.
            field_name (str): The name of the field to be checked.
            field_value (str or int or float): The value of the field to be checked.
        """
        if isinstance(field_value, Number) and (isnan(field_value) or isinf(field_value)):
            logger.warning(
                "Failed to log %s %s. Received invalid value: %s.",
                input_type,
                field_name,
                field_value,
            )
            return False
        return True

    def _log_graph_artifact(self, data, graph_type, is_output, artifact_name=None):
        """Log an artifact.

        Logs an artifact by uploading data to S3, creating an artifact, and associating that
        artifact with the run trial component.

        Args:
            data (dict): Artifacts data that will be saved to S3.
            graph_type (str):  The type of the artifact.
            is_output (bool): Determines direction of association to the
                trial component. Defaults to True (output artifact).
                If set to False then represented as input association.
            artifact_name (str): Name of the artifact (default: None).
        """
        # generate an artifact name
        if not artifact_name:
            unique_name_from_base(graph_type)

        # create a json file in S3
        s3_uri, etag = self._artifact_uploader.upload_object_artifact(
            artifact_name, data, file_extension="json"
        )

        # create an artifact and association for the table
        if is_output:
            self._lineage_artifact_tracker.add_output_artifact(
                name=artifact_name,
                source_uri=s3_uri,
                etag=etag,
                artifact_type=graph_type,
            )
        else:
            self._lineage_artifact_tracker.add_input_artifact(
                name=artifact_name,
                source_uri=s3_uri,
                etag=etag,
                artifact_type=graph_type,
            )

    def _verify_trial_component_artifacts_length(self, is_output):
        """Verify the length of trial component artifacts

        Args:
            is_output (bool): Determines direction of association to the
                trial component.

        Raises:
            ValueError: If the length of trial component artifacts exceeds the limit.
        """
        err_msg_template = "Cannot add more than {} {}_artifacts under run"
        if is_output:
            if len(self._trial_component.output_artifacts) >= MAX_RUN_TC_ARTIFACTS_LEN:
                raise ValueError(err_msg_template.format(MAX_RUN_TC_ARTIFACTS_LEN, "output"))
        else:
            if len(self._trial_component.input_artifacts) >= MAX_RUN_TC_ARTIFACTS_LEN:
                raise ValueError(err_msg_template.format(MAX_RUN_TC_ARTIFACTS_LEN, "input"))

    @staticmethod
    def _generate_trial_component_name(run_name: str, experiment_name: str) -> str:
        """Generate the TrialComponentName based on run_name and experiment_name

        Args:
            run_name (str): The run_name supplied by the user.
            experiment_name (str): The experiment_name supplied by the user,
                which is prepended to the run_name to generate the TrialComponentName.

        Returns:
            str: The TrialComponentName used to create a trial component
                which is unique in an account.

        Raises:
            ValueError: If either the run_name or the experiment_name exceeds
                the length limit.
        """
        buffer = 1  # leave length buffers for delimiters
        max_len = int(MAX_NAME_LEN_IN_BACKEND / 2) - buffer
        err_msg_template = "The {} (length: {}) must have length less than or equal to {}"
        if len(run_name) > max_len:
            raise ValueError(err_msg_template.format("run_name", len(run_name), max_len))
        if len(experiment_name) > max_len:
            raise ValueError(
                err_msg_template.format("experiment_name", len(experiment_name), max_len)
            )
        trial_component_name = "{}{}{}".format(experiment_name, DELIMITER, run_name)
        # due to mixed-case concerns on the backend
        trial_component_name = trial_component_name.lower()
        return trial_component_name

    @staticmethod
    def _extract_run_name_from_tc_name(trial_component_name: str, experiment_name: str) -> str:
        """Extract the user supplied run name from a trial component name.

        Args:
            trial_component_name (str): The name of a run trial component.
            experiment_name (str): The experiment_name supplied by the user,
                which was prepended to the run_name to generate the trial_component_name.

        Returns:
            str: The name of the Run object supplied by a user.
        """
        # TODO: we should revert the lower casting once backend fix reaches prod
        return trial_component_name.replace(
            "{}{}".format(experiment_name.lower(), DELIMITER), "", 1
        )

    @staticmethod
    def _append_run_tc_label_to_tags(tags: Optional[List[Dict[str, str]]] = None) -> list:
        """Append the run trial component label to tags used to create a trial component.

        Args:
            tags (List[Dict[str, str]]): The tags supplied by users to initialize a Run object.

        Returns:
            list: The updated tags with the appended run trial component label.
        """
        if not tags:
            tags = []
        if RUN_TC_TAG not in tags:
            tags.append(RUN_TC_TAG)
        return tags

    def __enter__(self):
        """Updates the start time of the run.

        Returns:
            object: self.
        """
        nested_with_err_msg_template = (
            "It is not allowed to use nested 'with' statements on the {}."
        )
        if self._in_load:
            if self._inside_load_context:
                raise RuntimeError(nested_with_err_msg_template.format("load_run"))
            self._inside_load_context = True
            if not self._inside_init_context:
                # Add to run context only if the load_run is called separately
                # without under a Run init context
                _RunContext.add_run_object(self)
        else:
            if _RunContext.get_current_run():
                raise RuntimeError(nested_with_err_msg_template.format("Run"))
            self._inside_init_context = True
            _RunContext.add_run_object(self)

        if not self._trial_component.start_time:
            start_time = datetime.datetime.now(dateutil.tz.tzlocal())
            self._trial_component.start_time = start_time
        self._trial_component.status = _api_types.TrialComponentStatus(
            primary_status=_TrialComponentStatusType.InProgress.value,
            message="Within a run context",
        )
        # Save the start_time and status changes to backend
        self._trial_component.save()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Updates the end time of the run.

        Args:
            exc_type (str): The exception type.
            exc_value (str): The exception value.
            exc_traceback (str): The stack trace of the exception.
        """
        if self._in_load:
            self._inside_load_context = False
            self._in_load = False
            if not self._inside_init_context:
                _RunContext.drop_current_run()
        else:
            self._inside_init_context = False
            _RunContext.drop_current_run()

        end_time = datetime.datetime.now(dateutil.tz.tzlocal())
        self._trial_component.end_time = end_time
        if exc_value:
            self._trial_component.status = _api_types.TrialComponentStatus(
                primary_status=_TrialComponentStatusType.Failed.value,
                message=str(exc_value),
            )
        else:
            self._trial_component.status = _api_types.TrialComponentStatus(
                primary_status=_TrialComponentStatusType.Completed.value
            )

        self.close()

    def __getstate__(self):
        """Overriding this method to prevent instance of Run from being pickled.

        Raise:
            NotImplementedError: If attempting to pickle this instance.
        """
        raise NotImplementedError("Instance of Run type is not allowed to be pickled.")


def load_run(
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    sagemaker_session: Optional["Session"] = None,
    artifact_bucket: Optional[str] = None,
    artifact_prefix: Optional[str] = None,
) -> Run:
    """Load an existing run.

    In order to reuse an existing run to log extra data, ``load_run`` is recommended.
    It can be used in several ways:

    1. Use ``load_run`` by explicitly passing in ``run_name`` and ``experiment_name``.

    If ``run_name`` and ``experiment_name`` are passed in, they are honored over
    the default experiment config in the job environment or the run context
    (i.e. within the ``with`` block).

    Note:
        Both ``run_name`` and ``experiment_name`` should be supplied to make this usage work.
        Otherwise, you may get a ``ValueError``.

    .. code:: python

        with load_run(experiment_name="my-exp", run_name="my-run") as run:
            run.log_metric(...)
            ...

    2. Use the ``load_run`` in a job script without supplying ``run_name`` and ``experiment_name``.

    In this case, the default experiment config (specified when creating the job) is fetched
    from the job environment to load the run.

    .. code:: python

        # In a job script
        with load_run() as run:
            run.log_metric(...)
            ...

    3. Use the ``load_run`` in a notebook within a run context (i.e. the ``with`` block)
    but without supplying ``run_name`` and ``experiment_name``.

    Every time we call ``with Run(...) as run1:``, the initialized ``run1`` is tracked
    in the run context. Then when we call ``load_run()`` under this with statement, the ``run1``
    in the context is loaded by default.

    .. code:: python

        # In a notebook
        with Run(experiment_name="my-exp", run_name="my-run", ...) as run1:
            run1.log_parameter(...)

            with load_run() as run2: # run2 is the same object as run1
                run2.log_metric(...)
                ...

    Args:
        run_name (str): The name of the run to be loaded (default: None).
            If it is None, the ``RunName`` in the ``ExperimentConfig`` of the job will be
            fetched to load the run.
        experiment_name (str): The name of the Experiment that the to be loaded run
            is associated with (default: None).
            Note: the experiment_name must be supplied along with a valid run_name.
            Otherwise, it will be ignored.
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other
            AWS services needed. If not specified, one is created using the
            default AWS configuration chain.
        artifact_bucket (str): The S3 bucket to upload the artifact to.
                If not specified, the default bucket defined in `sagemaker_session`
                will be used.
        artifact_prefix (str): The S3 key prefix used to generate the S3 path
            to upload the artifact to (default: "trial-component-artifacts").

    Returns:
        Run: The loaded Run object.
    """
    environment = _RunEnvironment.load()

    verify_load_input_names(run_name=run_name, experiment_name=experiment_name)

    if run_name:
        logger.warning(
            "run_name is explicitly supplied in load_run, "
            "which will be prioritized to load the Run object. "
            "In other words, the run name in the experiment config, fetched from the "
            "job environment or the current run context, will be ignored."
        )
        run_instance = Run(
            experiment_name=experiment_name,
            run_name=run_name,
            sagemaker_session=sagemaker_session or _utils.default_session(),
            artifact_bucket=artifact_bucket,
            artifact_prefix=artifact_prefix,
        )
    elif _RunContext.get_current_run():
        run_instance = _RunContext.get_current_run()
    elif environment:
        exp_config = get_tc_and_exp_config_from_job_env(
            environment=environment,
            sagemaker_session=sagemaker_session or _utils.default_session(),
        )
        run_name = Run._extract_run_name_from_tc_name(
            trial_component_name=exp_config[RUN_NAME],
            experiment_name=exp_config[EXPERIMENT_NAME],
        )
        experiment_name = exp_config[EXPERIMENT_NAME]
        run_instance = Run(
            experiment_name=experiment_name,
            run_name=run_name,
            sagemaker_session=sagemaker_session or _utils.default_session(),
            artifact_bucket=artifact_bucket,
            artifact_prefix=artifact_prefix,
        )
    else:
        raise RuntimeError(
            "Failed to load a Run object. "
            "Please make sure a Run object has been initialized already."
        )

    run_instance._in_load = True
    return run_instance


def list_runs(
    experiment_name: str,
    created_before: Optional[datetime.datetime] = None,
    created_after: Optional[datetime.datetime] = None,
    sagemaker_session: Optional["Session"] = None,
    max_results: Optional[int] = None,
    next_token: Optional[str] = None,
    sort_by: SortByType = SortByType.CREATION_TIME,
    sort_order: SortOrderType = SortOrderType.DESCENDING,
) -> list:
    """Return a list of ``Run`` objects matching the given criteria.

    Args:
        experiment_name (str): Only Run objects related to the specified experiment
            are returned.
        created_before (datetime.datetime): Return Run objects created before this instant
            (default: None).
        created_after (datetime.datetime): Return Run objects created after this instant
            (default: None).
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other
            AWS services needed. If not specified, one is created using the
            default AWS configuration chain.
        max_results (int): Maximum number of Run objects to retrieve (default: None).
        next_token (str): Token for next page of results (default: None).
        sort_by (SortByType): The property to sort results by. One of NAME, CREATION_TIME
            (default: CREATION_TIME).
        sort_order (SortOrderType): One of ASCENDING, or DESCENDING (default: DESCENDING).

    Returns:
        list: A list of ``Run`` objects.
    """

    # all trial components retrieved by default
    tc_summaries = _TrialComponent.list(
        experiment_name=experiment_name,
        created_before=created_before,
        created_after=created_after,
        sort_by=sort_by.value,
        sort_order=sort_order.value,
        sagemaker_session=sagemaker_session,
        max_results=max_results,
        next_token=next_token,
    )
    run_list = []
    for tc_summary in tc_summaries:
        if not is_run_trial_component(
            trial_component_name=tc_summary.trial_component_name,
            sagemaker_session=sagemaker_session,
        ):
            continue
        run_instance = Run(
            experiment_name=experiment_name,
            run_name=Run._extract_run_name_from_tc_name(
                trial_component_name=tc_summary.trial_component_name,
                experiment_name=experiment_name,
            ),
            sagemaker_session=sagemaker_session,
        )
        run_list.append(run_instance)
    return run_list
