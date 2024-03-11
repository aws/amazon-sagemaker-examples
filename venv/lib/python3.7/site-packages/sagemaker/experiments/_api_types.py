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
"""Contains API objects for SageMaker experiments."""
from __future__ import absolute_import

import enum
import numbers

from sagemaker.apiutils import _base_types


class TrialComponentMetricSummary(_base_types.ApiObject):
    """Summary model of a trial component.

    Attributes:
        metric_name (str): The name of the metric.
        source_arn (str):  The ARN of the source.
        time_stamp (datetime): Metric last updated value.
        max (float): The max value of the metric.
        min (float):  The min value of the metric.
        last (float):  The last value of the metric.
        count (float):  The number of samples used to generate the metric.
        avg (float):  The average value of the metric.
        std_dev (float):  The standard deviation of the metric.
    """

    metric_name = None
    source_arn = None
    time_stamp = None
    max = None
    min = None
    last = None
    count = None
    avg = None
    std_dev = None

    def __init__(self, metric_name=None, source_arn=None, **kwargs):
        super(TrialComponentMetricSummary, self).__init__(
            metric_name=metric_name, source_arn=source_arn, **kwargs
        )


class TrialComponentParameters(_base_types.ApiObject):
    """A dictionary of TrialComponentParameterValues"""

    @classmethod
    def from_boto(cls, boto_dict, **kwargs):
        """Converts a boto dict to a dictionary of TrialComponentParameterValues

        Args:
            boto_dict (dict): boto response dictionary.
            **kwargs:  Arbitrary keyword arguments.

        Returns:
            dict: Dictionary of parameter values.
        """
        return_map = {}
        for key, value in boto_dict.items():
            return_map[key] = value.get("NumberValue", value.get("StringValue", None))
        return return_map

    @classmethod
    def to_boto(cls, parameters):
        """Converts TrialComponentParameters to dict.

        Args:
            parameters (TrialComponentParameters): Dictionary to convert.

        Returns:
            dict: Dictionary of trial component parameters in boto format.
        """
        boto_map = {}
        for key, value in parameters.items():
            if isinstance(value, numbers.Number):
                boto_map[key] = {"NumberValue": value}
            else:
                boto_map[key] = {"StringValue": str(value)}
        return boto_map


class TrialComponentArtifact(_base_types.ApiObject):
    """Trial component artifact.

    Attributes:
        value (str): The artifact value.
        media_type (str): The media type.
    """

    value = None
    media_type = None

    def __init__(self, value=None, media_type=None, **kwargs):
        super(TrialComponentArtifact, self).__init__(value=value, media_type=media_type, **kwargs)


class _TrialComponentStatusType(enum.Enum):
    """The type of trial component status"""

    InProgress = "InProgress"
    Completed = "Completed"
    Failed = "Failed"


class TrialComponentStatus(_base_types.ApiObject):
    """Status of the trial component.

    Attributes:
        primary_status (str): The status of a trial component.
        message (str): Status message.
    """

    primary_status = None
    message = None

    def __init__(self, primary_status=None, message=None, **kwargs):
        super(TrialComponentStatus, self).__init__(
            primary_status=primary_status, message=message, **kwargs
        )


class TrialComponentSummary(_base_types.ApiObject):
    """Summary model of a trial component.

    Attributes:
        trial_component_name (str): Name of trial component.
        trial_component_arn (str): ARN of the trial component.
        display_name (str): Friendly display name in UI.
        source_arn (str): ARN of the trial component source.
        status (str): Status.
        start_time (datetime): Start time.
        end_time (datetime): End time.
        creation_time (datetime): Creation time.
        created_by (str): Created by.
        last_modified_time (datetime): Date last modified.
        last_modified_by (datetime): User last modified.
    """

    _custom_boto_types = {
        "status": (TrialComponentStatus, False),
    }
    trial_component_name = None
    trial_component_arn = None
    display_name = None
    source_arn = None
    status = None
    start_time = None
    end_time = None
    creation_time = None
    created_by = None
    last_modified_time = None
    last_modified_by = None


class TrialComponentSource(_base_types.ApiObject):
    """Trial Component Source

    Attributes:
        source_arn (str): The ARN of the source.
    """

    source_arn = None

    def __init__(self, source_arn=None, **kwargs):
        super(TrialComponentSource, self).__init__(source_arn=source_arn, **kwargs)


class Parent(_base_types.ApiObject):
    """The trial/experiment/run that a trial component is associated with.

    Attributes:
        trial_name (str): Name of the trial.
        experiment_name (str): Name of the experiment.
        run_name (str): Name of the run.
    """

    trial_name = None
    experiment_name = None
    run_name = None


class TrialComponentSearchResult(_base_types.ApiObject):
    """Summary model of an Trial Component search result.

    Attributes:
        trial_component_arn (str): ARN of the trial component.
        trial_component_name (str): Name of the trial component.
        display_name (str): Display name of the trial component for UI display.
        source (dict): The source of the trial component.
        status (dict): The status of the trial component.
        start_time (datetime): Start time.
        end_time (datetime): End time.
        creation_time (datetime): Creation time.
        created_by (str): Created by.
        last_modified_time (datetime): Date last modified.
        last_modified_by (datetime): User last modified.
        parameters (dict): The hyperparameters of the component.
        input_artifacts (dict): The input artifacts of the component.
        output_artifacts (dict): The output artifacts of the component.
        metrics (list): The metrics for the component.
        source_detail (dict): The source of the trial component.
        tags (list): The list of tags that are associated with the trial component.
        parents (list[Parent]): The parent of trial component.
    """

    _custom_boto_types = {
        "parents": (Parent, True),  # parents is a collection (list) of Parent objects
    }
    trial_component_arn = None
    trial_component_name = None
    display_name = None
    source = None
    status = None
    start_time = None
    end_time = None
    creation_time = None
    created_by = None
    last_modified_time = None
    last_modified_by = None
    parameters = None
    input_artifacts = None
    output_artifacts = None
    metrics = None
    source_detail = None
    tags = None
    parents = None


class TrialSummary(_base_types.ApiObject):
    """Summary model of a trial.

    Attributes:
        trial_arn (str): The ARN of the trial.
        trial_name (str): The name of the trial.
        creation_time (datetime):  When the trial was created.
        last_modified_time (datetime): When the trial was last modified.
    """

    trial_arn = None
    trial_name = None
    creation_time = None
    last_modified_time = None
