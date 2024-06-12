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
"""Placeholder docstring"""
from __future__ import print_function, absolute_import

from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
import datetime
import logging

from six import with_metaclass

from sagemaker.session import Session
from sagemaker.utils import DeferredError
from sagemaker.lineage import artifact

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError as e:
    logger.warning("pandas failed to import. Analytics features will be impaired or broken.")
    # Any subsequent attempt to use pandas will raise the ImportError
    pd = DeferredError(e)

METRICS_PERIOD_DEFAULT = 60  # seconds


class AnalyticsMetricsBase(with_metaclass(ABCMeta, object)):
    """Base class for tuning job or training job analytics classes.

    Understands common functionality like persistence and caching.
    """

    def __init__(self):
        """Initializes ``AnalyticsMetricsBase`` instance."""
        self._dataframe = None

    def export_csv(self, filename):
        """Persists the analytics dataframe to a file.

        Args:
            filename (str): The name of the file to save to.
        """
        self.dataframe().to_csv(filename)

    def dataframe(self, force_refresh=False):
        """A pandas dataframe with lots of interesting results about this object.

        Created by calling SageMaker List and Describe APIs and converting them into a
        convenient tabular summary.

        Args:
            force_refresh (bool): Set to True to fetch the latest data from
                SageMaker API.
        """
        if force_refresh:
            self.clear_cache()
        if self._dataframe is None:
            self._dataframe = self._fetch_dataframe()
        return self._dataframe

    @abstractmethod
    def _fetch_dataframe(self):
        """Sub-class must calculate the dataframe and return it."""

    def clear_cache(self):
        """Clear the object of all local caches of API methods.

        So that the next time any properties are accessed they will be refreshed from the service.
        """
        self._dataframe = None


class HyperparameterTuningJobAnalytics(AnalyticsMetricsBase):
    """Fetch results about a hyperparameter tuning job and make them accessible for analytics."""

    def __init__(self, hyperparameter_tuning_job_name, sagemaker_session=None):
        """Initialize a ``HyperparameterTuningJobAnalytics`` instance.

        Args:
            hyperparameter_tuning_job_name (str): name of the
                HyperparameterTuningJob to analyze.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or Session()
        self._sage_client = sagemaker_session.sagemaker_client
        self._tuning_job_name = hyperparameter_tuning_job_name
        self._tuning_job_describe_result = None
        self._training_job_summaries = None
        super(HyperparameterTuningJobAnalytics, self).__init__()
        self.clear_cache()

    @property
    def name(self):
        """Name of the HyperparameterTuningJob being analyzed"""
        return self._tuning_job_name

    def __repr__(self):
        """Human-readable representation override."""
        return "<sagemaker.HyperparameterTuningJobAnalytics for %s>" % self.name

    def clear_cache(self):
        """Clear the object of all local caches of API methods."""
        super(HyperparameterTuningJobAnalytics, self).clear_cache()
        self._tuning_job_describe_result = None
        self._training_job_summaries = None

    def _fetch_dataframe(self):
        """Return a pandas dataframe with all the training jobs.

        This includes their hyperparameters, results, and metadata, as well as
        a column to indicate if a training job was the best seen so far.
        """

        def reshape(training_summary):
            # Helper method to reshape a single training job summary into a dataframe record
            out = {}
            for k, v in training_summary["TunedHyperParameters"].items():
                # Something (bokeh?) gets confused with ints so convert to float
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    pass
                out[k] = v
            out["TrainingJobName"] = training_summary["TrainingJobName"]
            out["TrainingJobStatus"] = training_summary["TrainingJobStatus"]
            out["FinalObjectiveValue"] = training_summary.get(
                "FinalHyperParameterTuningJobObjectiveMetric", {}
            ).get("Value")

            start_time = training_summary.get("TrainingStartTime", None)
            end_time = training_summary.get("TrainingEndTime", None)
            out["TrainingStartTime"] = start_time
            out["TrainingEndTime"] = end_time
            if start_time and end_time:
                out["TrainingElapsedTimeSeconds"] = (end_time - start_time).total_seconds()
            if "TrainingJobDefinitionName" in training_summary:
                out["TrainingJobDefinitionName"] = training_summary["TrainingJobDefinitionName"]
            return out

        # Run that helper over all the summaries.
        df = pd.DataFrame([reshape(tjs) for tjs in self.training_job_summaries()])
        return df

    @property
    def tuning_ranges(self):
        """A dictionary describing the ranges of all tuned hyperparameters.

        The keys are the names of the hyperparameter, and the values are the ranges.

        The output can take one of two forms:

            * If the 'TrainingJobDefinition' field is present in the job description, the output
                is a dictionary constructed from 'ParameterRanges' in
                'HyperParameterTuningJobConfig' of the job description. The keys are the
                parameter names, while the values are the parameter ranges.
                Example:
                >>> {
                >>>     "eta": {"MaxValue": "1", "MinValue": "0", "Name": "eta"},
                >>>     "gamma": {"MaxValue": "10", "MinValue": "0", "Name": "gamma"},
                >>>     "iterations": {"MaxValue": "100", "MinValue": "50", "Name": "iterations"},
                >>>     "num_layers": {"MaxValue": "30", "MinValue": "5", "Name": "num_layers"},
                >>> }
            * If the 'TrainingJobDefinitions' field (list) is present in the job description,
                the output is a dictionary with keys as the 'DefinitionName' values from
                all items in 'TrainingJobDefinitions', and each value would be a dictionary
                constructed from 'HyperParameterRanges' in each item in 'TrainingJobDefinitions'
                in the same format as above
                Example:
                >>> {
                >>>     "estimator_1": {
                >>>         "eta": {"MaxValue": "1", "MinValue": "0", "Name": "eta"},
                >>>         "gamma": {"MaxValue": "10", "MinValue": "0", "Name": "gamma"},
                >>>     },
                >>>     "estimator_2": {
                >>>         "framework": {"Values": ["TF", "MXNet"], "Name": "framework"},
                >>>         "gamma": {"MaxValue": "1.0", "MinValue": "0.2", "Name": "gamma"}
                >>>     }
                >>> }

        For more details about the 'TrainingJobDefinition' and 'TrainingJobDefinitions' fields
        in job description, see
        https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job
        """
        description = self.description()

        if "TrainingJobDefinition" in description:
            return self._prepare_parameter_ranges(
                description["HyperParameterTuningJobConfig"]["ParameterRanges"]
            )

        return {
            training_job_definition["DefinitionName"]: self._prepare_parameter_ranges(
                training_job_definition["HyperParameterRanges"]
            )
            for training_job_definition in description["TrainingJobDefinitions"]
        }

    def _prepare_parameter_ranges(self, parameter_ranges):
        """Convert parameter ranges a dictionary using the parameter range names as the keys"""
        out = {}
        for _, ranges in parameter_ranges.items():
            for param in ranges:
                out[param["Name"]] = param
        return out

    def description(self, force_refresh=False):
        """Call ``DescribeHyperParameterTuningJob`` for the hyperparameter tuning job.

        Args:
            force_refresh (bool): Set to True to fetch the latest data from
                SageMaker API.

        Returns:
            dict: The Amazon SageMaker response for
            ``DescribeHyperParameterTuningJob``.
        """
        if force_refresh:
            self.clear_cache()
        if not self._tuning_job_describe_result:
            self._tuning_job_describe_result = self._sage_client.describe_hyper_parameter_tuning_job(  # noqa: E501 # pylint: disable=line-too-long
                HyperParameterTuningJobName=self.name
            )
        return self._tuning_job_describe_result

    def training_job_summaries(self, force_refresh=False):
        """A (paginated) list of everything from ``ListTrainingJobsForTuningJob``.

        Args:
            force_refresh (bool): Set to True to fetch the latest data from
                SageMaker API.

        Returns:
            dict: The Amazon SageMaker response for
            ``ListTrainingJobsForTuningJob``.
        """
        if force_refresh:
            self.clear_cache()
        if self._training_job_summaries is not None:
            return self._training_job_summaries
        output = []
        next_args = {}
        for count in range(100):
            logger.debug("Calling list_training_jobs_for_hyper_parameter_tuning_job %d", count)
            raw_result = self._sage_client.list_training_jobs_for_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=self.name, MaxResults=100, **next_args
            )
            new_output = raw_result["TrainingJobSummaries"]
            output.extend(new_output)
            logger.debug(
                "Got %d more TrainingJobs. Total so far: %d",
                len(new_output),
                len(output),
            )
            if ("NextToken" in raw_result) and (len(new_output) > 0):
                next_args["NextToken"] = raw_result["NextToken"]
            else:
                break
        self._training_job_summaries = output
        return output


class TrainingJobAnalytics(AnalyticsMetricsBase):
    """Fetch training curve data from CloudWatch Metrics for a specific training job."""

    CLOUDWATCH_NAMESPACE = "/aws/sagemaker/TrainingJobs"

    def __init__(
        self,
        training_job_name,
        metric_names=None,
        sagemaker_session=None,
        start_time=None,
        end_time=None,
        period=None,
    ):
        """Initialize a ``TrainingJobAnalytics`` instance.

        Args:
            training_job_name (str): name of the TrainingJob to analyze.
            metric_names (list, optional): string names of all the metrics to
                collect for this training job. If not specified, then it will
                use all metric names configured for this job.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is specified using
                the default AWS configuration chain.
            start_time:
            end_time:
            period:
        """
        sagemaker_session = sagemaker_session or Session()
        self._sage_client = sagemaker_session.sagemaker_client
        self._cloudwatch = sagemaker_session.boto_session.client("cloudwatch")
        self._training_job_name = training_job_name
        self._start_time = start_time
        self._end_time = end_time
        self._period = period or METRICS_PERIOD_DEFAULT

        if metric_names:
            self._metric_names = metric_names
        else:
            self._metric_names = self._metric_names_for_training_job()

        super(TrainingJobAnalytics, self).__init__()
        self.clear_cache()

    @property
    def name(self):
        """Name of the TrainingJob being analyzed"""
        return self._training_job_name

    def __repr__(self):
        """The human-readable representation override."""
        return "<sagemaker.TrainingJobAnalytics for %s>" % self.name

    def clear_cache(self):
        """Clear the object of all local caches of API methods.

        This is so that the next time any properties are accessed they will be
        refreshed from the service.
        """
        super(TrainingJobAnalytics, self).clear_cache()
        self._data = defaultdict(list)
        self._time_interval = self._determine_timeinterval()

    def _determine_timeinterval(self):
        """Return a dict with two datetime objects.

        The dict includes the `start_time` and `end_time`, covering the interval
        of the training job.

        Returns:
            a dict with the `start_time` and `end_time`.
        """
        description = self._sage_client.describe_training_job(TrainingJobName=self.name)
        start_time = self._start_time or description["TrainingStartTime"]  # datetime object
        # Incrementing end time by 1 min since CloudWatch drops seconds before finding the logs.
        # This results in logs being searched in the time range in which the correct log line was
        # not present.
        # Example - Log time - 2018-10-22 08:25:55
        #       Here calculated end time would also be 2018-10-22 08:25:55 (without 1 min addition)
        #       CW will consider end time as 2018-10-22 08:25 and will not be able to search the
        #           correct log.
        end_time = self._end_time or description.get(
            "TrainingEndTime", datetime.datetime.utcnow()
        ) + datetime.timedelta(minutes=1)

        return {"start_time": start_time, "end_time": end_time}

    def _fetch_dataframe(self):
        for metric_name in self._metric_names:
            self._fetch_metric(metric_name)
        return pd.DataFrame(self._data)

    def _fetch_metric(self, metric_name):
        """Fetch all the values of a named metric, and add them to _data

        Args:
            metric_name: The metric name to fetch.
        """
        request = {
            "Namespace": self.CLOUDWATCH_NAMESPACE,
            "MetricName": metric_name,
            "Dimensions": [{"Name": "TrainingJobName", "Value": self.name}],
            "StartTime": self._time_interval["start_time"],
            "EndTime": self._time_interval["end_time"],
            "Period": self._period,
            "Statistics": ["Average"],
        }
        raw_cwm_data = self._cloudwatch.get_metric_statistics(**request)["Datapoints"]
        if len(raw_cwm_data) == 0:
            logger.warning("Warning: No metrics called %s found", metric_name)
            return

        # Process data: normalize to starting time, and sort.
        base_time = min(raw_cwm_data, key=lambda pt: pt["Timestamp"])["Timestamp"]
        all_xy = []
        for pt in raw_cwm_data:
            y = pt["Average"]
            x = (pt["Timestamp"] - base_time).total_seconds()
            all_xy.append([x, y])
        all_xy = sorted(all_xy, key=lambda x: x[0])

        # Store everything in _data to make a dataframe from
        for elapsed_seconds, value in all_xy:
            self._add_single_metric(elapsed_seconds, metric_name, value)

    def _add_single_metric(self, timestamp, metric_name, value):
        """Store a single metric in the _data dict.

        This can be converted to a dataframe.

        Args:
            timestamp: The timestamp of the metric.
            metric_name: The name of the metric.
            value: The value of the metric.
        """
        # note that this method is built this way to make it possible to
        # support live-refreshing charts in Bokeh at some point in the future.
        self._data["timestamp"].append(timestamp)
        self._data["metric_name"].append(metric_name)
        self._data["value"].append(value)

    def _metric_names_for_training_job(self):
        """Helper method to discover the metrics defined for a training job."""
        training_description = self._sage_client.describe_training_job(
            TrainingJobName=self._training_job_name
        )

        metric_definitions = training_description["AlgorithmSpecification"]["MetricDefinitions"]
        metric_names = [md["Name"] for md in metric_definitions]

        return metric_names


class ArtifactAnalytics(AnalyticsMetricsBase):
    """Fetch artifact data and make them accessible for analytics."""

    def __init__(
        self,
        sort_by=None,
        sort_order=None,
        source_uri=None,
        artifact_type=None,
        sagemaker_session=None,
    ):
        """Initialize a ``ArtifactAnalytics`` instance.

        Args:
            sort_by (str, optional): The name of the resource property used to sort
                the set of artifacts. Currently only support for sort by Name
            sort_order(str optional): How trial components are ordered, valid values are Ascending
                and Descending. The default is Descending.
            source_uri(dict optional): The artifact source uri for filtering.
            artifact_type(dict optional): The artifact type for filtering.
            sagemaker_session (obj, optional): Sagemaker session. Defaults to None.
        """
        self._sort_by = sort_by if sort_by == "Name" else None
        self._sort_order = sort_order
        self._source_uri = source_uri
        self._artifact_type = artifact_type
        self._sagemaker_session = sagemaker_session
        super(ArtifactAnalytics, self).__init__()
        self.clear_cache()

    def __repr__(self):
        """Human-readable representation override."""
        return "<sagemaker.ArtifactAnalytics>"

    def _reshape_source_type(self, artifact_source_types):
        """Reshape artifact source type."""
        out = OrderedDict()
        for artifact_source_type in artifact_source_types:
            out["ArtifactSourceType"] = artifact_source_type
        return out

    def _reshape(self, artifact_summary):
        """Reshape artifact summary."""
        out = OrderedDict()
        out["ArtifactName"] = artifact_summary.artifact_name
        out["ArtifactArn"] = artifact_summary.artifact_arn
        out["ArtifactType"] = artifact_summary.artifact_type
        out["ArtifactSourceUri"] = artifact_summary.source.source_uri
        out["CreationTime"] = artifact_summary.creation_time
        out["LastModifiedTime"] = artifact_summary.last_modified_time
        return out

    def _fetch_dataframe(self):
        """Return a pandas dataframe with all artifacts."""
        df = pd.DataFrame([self._reshape(artifact) for artifact in self._get_list_artifacts()])
        return df

    def _get_list_artifacts(self):
        """List artifacts."""
        artifacts = artifact.Artifact.list(
            source_uri=self._source_uri,
            artifact_type=self._artifact_type,
            sort_by=self._sort_by,
            sort_order=self._sort_order,
            sagemaker_session=self._sagemaker_session,
        )
        return artifacts


class ExperimentAnalytics(AnalyticsMetricsBase):
    """Fetch trial component data and make them accessible for analytics."""

    MAX_TRIAL_COMPONENTS = 10000

    def __init__(
        self,
        experiment_name=None,
        search_expression=None,
        sort_by=None,
        sort_order=None,
        metric_names=None,
        parameter_names=None,
        sagemaker_session=None,
        input_artifact_names=None,
        output_artifact_names=None,
    ):
        """Initialize a ``ExperimentAnalytics`` instance.

        Args:
            experiment_name (str, optional): Name of the experiment if you want to constrain the
                search to only trial components belonging to an experiment.
            search_expression (dict, optional): The search query to find the set of trial components
                to use to populate the data frame.
            sort_by (str, optional): The name of the resource property used to sort
                the set of trial components.
            sort_order(str optional): How trial components are ordered, valid values are Ascending
                and Descending. The default is Descending.
            metric_names (list, optional): string names of all the metrics to be shown in the
                data frame. If not specified, all metrics will be shown of all trials.
            parameter_names (list, optional): string names of the parameters to be shown in the
                data frame. If not specified, all parameters will be shown of all trials.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed. If not specified,
                one is created using the default AWS configuration chain.
            input_artifact_names(dict optional):The input artifacts for the experiment. Examples of
                input artifacts are datasets, algorithms, hyperparameters, source code, and instance
                types.
            output_artifact_names(dict optional): The output artifacts for the experiment. Examples
                of output artifacts are metrics, snapshots, logs, and images.
        """
        sagemaker_session = sagemaker_session or Session()
        self._sage_client = sagemaker_session.sagemaker_client

        if not experiment_name and not search_expression:
            raise ValueError("Either experiment_name or search_expression must be supplied.")

        self._experiment_name = experiment_name
        self._search_expression = search_expression
        self._sort_by = sort_by
        self._sort_order = sort_order
        self._metric_names = metric_names
        self._parameter_names = parameter_names
        self._input_artifact_names = input_artifact_names
        self._output_artifact_names = output_artifact_names
        self._trial_components = None
        super(ExperimentAnalytics, self).__init__()
        self.clear_cache()

    @property
    def name(self):
        """Name of the Experiment being analyzed."""
        return self._experiment_name

    def __repr__(self):
        """The human-readable representation override."""
        return "<sagemaker.ExperimentAnalytics for %s>" % self.name

    def clear_cache(self):
        """Clear the object of all local caches of API methods."""
        super(ExperimentAnalytics, self).clear_cache()
        self._trial_components = None

    def _reshape_parameters(self, parameters):
        """Reshape trial component parameters to a pandas column.

        Args:
            parameters: trial component parameters
        Returns:
            dict: Key: Parameter name, Value: Parameter value
        """
        out = OrderedDict()
        for name, value in sorted(parameters.items()):
            if self._parameter_names and name not in self._parameter_names:
                continue
            out[name] = value.get("NumberValue", value.get("StringValue"))
        return out

    def _reshape_metrics(self, metrics):
        """Reshape trial component metrics to a pandas column.

        Args:
            metrics: trial component metrics
        Returns:
            dict: Key: Metric name, Value: Metric value
        """
        statistic_types = ["Min", "Max", "Avg", "StdDev", "Last", "Count"]
        out = OrderedDict()
        for metric_summary in metrics:
            metric_name = metric_summary["MetricName"]
            if self._metric_names and metric_name not in self._metric_names:
                continue

            for stat_type in statistic_types:
                stat_value = metric_summary.get(stat_type)
                if stat_value is not None:
                    out["{} - {}".format(metric_name, stat_type)] = stat_value
        return out

    def _reshape_artifacts(self, artifacts, _artifact_names):
        """Reshape trial component input/output artifacts to a pandas column.

        Args:
            artifacts: trial component input/output artifacts
        Returns:
            dict: Key: artifacts name, Value: artifacts value
        """
        out = OrderedDict()
        for name, value in sorted(artifacts.items()):
            if _artifact_names and (name not in _artifact_names):
                continue
            out["{} - {}".format(name, "MediaType")] = value.get("MediaType")
            out["{} - {}".format(name, "Value")] = value.get("Value")
        return out

    def _reshape_parents(self, parents):
        """Reshape trial component parents to a pandas column.

        Args:
            parents: trial component parents (trials and experiments)
        Returns:
            dict: Key: artifacts name, Value: artifacts value
        """
        out = OrderedDict()
        trials = []
        experiments = []
        for parent in parents:
            trials.append(parent["TrialName"])
            experiments.append(parent["ExperimentName"])
        out["Trials"] = trials
        out["Experiments"] = experiments
        return out

    def _reshape(self, trial_component):
        """Reshape trial component data to pandas columns.

        Args:
            trial_component: dict representing a trial component
        Returns:
            dict: Key-Value pair representing the data in the pandas dataframe
        """
        out = OrderedDict()
        for attribute in ["TrialComponentName", "DisplayName"]:
            out[attribute] = trial_component.get(attribute, "")

        source = trial_component.get("Source", "")
        if source:
            out["SourceArn"] = source["SourceArn"]

        out.update(self._reshape_parameters(trial_component.get("Parameters", [])))
        out.update(self._reshape_metrics(trial_component.get("Metrics", [])))
        out.update(
            self._reshape_artifacts(
                trial_component.get("InputArtifacts", []), self._input_artifact_names
            )
        )
        out.update(
            self._reshape_artifacts(
                trial_component.get("OutputArtifacts", []), self._output_artifact_names
            )
        )
        out.update(self._reshape_parents(trial_component.get("Parents", [])))
        return out

    def _fetch_dataframe(self):
        """Return a pandas dataframe includes all the trial_components."""

        df = pd.DataFrame([self._reshape(component) for component in self._get_trial_components()])
        return df

    def _get_trial_components(self, force_refresh=False):
        """Get all trial components matching the given search query expression.

        Args:
            force_refresh (bool): Set to True to fetch the latest data from SageMaker API.

        Returns:
            list: List of dicts representing the trial components
        """
        if force_refresh:
            self.clear_cache()
        if self._trial_components is not None:
            return self._trial_components

        if not self._search_expression:
            self._search_expression = {}

        if self._experiment_name:
            if not self._search_expression.get("Filters"):
                self._search_expression["Filters"] = []

            self._search_expression["Filters"].append(
                {
                    "Name": "Parents.ExperimentName",
                    "Operator": "Equals",
                    "Value": self._experiment_name,
                }
            )

        return self._search(self._search_expression, self._sort_by, self._sort_order)

    def _search(self, search_expression, sort_by, sort_order):
        """Perform a search query using SageMaker Search and return the matching trial components.

        Args:
            search_expression: Search expression to filter trial components.
            sort_by: The name of the resource property used to sort the trial components.
            sort_order: How trial components are ordered, valid values are Ascending
                and Descending. The default is Descending.
        Returns:
            list: List of dict representing trial components.
        """
        trial_components = []

        search_args = {
            "Resource": "ExperimentTrialComponent",
            "SearchExpression": search_expression,
        }

        if sort_by:
            search_args["SortBy"] = sort_by

        if sort_order:
            search_args["SortOrder"] = sort_order

        while len(trial_components) < self.MAX_TRIAL_COMPONENTS:
            search_response = self._sage_client.search(**search_args)
            components = [result["TrialComponent"] for result in search_response["Results"]]
            trial_components.extend(components)
            if "NextToken" in search_response and len(components) > 0:
                search_args["NextToken"] = search_response["NextToken"]
            else:
                break

        return trial_components
