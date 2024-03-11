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
"""Contains classes to manage metrics for Sagemaker Experiment"""
from __future__ import absolute_import

import datetime
import logging
import os
import time
import threading
import queue

import dateutil.tz

from sagemaker.session import Session

METRICS_DIR = os.environ.get("SAGEMAKER_METRICS_DIRECTORY", ".")
METRIC_TS_LOWER_BOUND_TO_NOW = 1209600  # on seconds
METRIC_TS_UPPER_BOUND_FROM_NOW = 7200  # on seconds

BATCH_SIZE = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _RawMetricData(object):
    """A Raw Metric Data Object"""

    MetricName = None
    Value = None
    Timestamp = None
    Step = None

    def __init__(self, metric_name, value, timestamp=None, step=None):
        """Construct a `_RawMetricData` instance.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            timestamp (datetime.datetime or float or str): Timestamp of the metric.
                If not specified, the current UTC time will be used.
            step (int):  Iteration number of the metric (default: None).
        """
        if timestamp is None:
            timestamp = time.time()
        elif isinstance(timestamp, datetime.datetime):
            # If the input is a datetime then convert it to UTC time.
            # Assume a naive datetime is in local timezone
            if not timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=dateutil.tz.tzlocal())
            timestamp = (timestamp - timestamp.utcoffset()).replace(tzinfo=datetime.timezone.utc)
            timestamp = timestamp.timestamp()
        else:
            timestamp = float(timestamp)

        if timestamp < (time.time() - METRIC_TS_LOWER_BOUND_TO_NOW) or timestamp > (
            time.time() + METRIC_TS_UPPER_BOUND_FROM_NOW
        ):
            raise ValueError(
                "Supplied timestamp %f is invalid."
                " Timestamps must be between two weeks before and two hours from now." % timestamp
            )
        value = float(value)

        self.MetricName = metric_name
        self.Value = float(value)
        self.Timestamp = timestamp
        if step is not None:
            if not isinstance(step, int):
                raise ValueError("step must be int.")
            self.Step = step

    def to_record(self):
        """Convert the `_RawMetricData` object to dict"""
        return self.__dict__

    def to_raw_metric_data(self):
        """Converts the metric data to a BatchPutMetrics RawMetricData item"""
        # Convert timestamp from float to timestamp str.
        # Otherwise will get ParamValidationError
        raw_metric_data = {
            "MetricName": self.MetricName,
            "Value": self.Value,
            "Timestamp": str(int(self.Timestamp)),
        }
        if self.Step is not None:
            raw_metric_data["Step"] = int(self.Step)
        return raw_metric_data

    def __str__(self):
        """String representation of the `_RawMetricData` object."""
        return repr(self)

    def __repr__(self):
        """Return a string representation of this _RawMetricData` object."""
        return "{}({})".format(
            type(self).__name__,
            ",".join(["{}={}".format(k, repr(v)) for k, v in vars(self).items()]),
        )


class _MetricsManager(object):
    """Collects metrics and sends them directly to SageMaker Metrics data plane APIs."""

    def __init__(self, trial_component_name: str, sagemaker_session: Session, sink=None) -> None:
        """Initialize a `_MetricsManager` instance

        Args:
            trial_component_name (str): The Name of the Trial Component to log metrics to
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
            sink (object): The metrics sink to use.
        """
        if sink is None:
            self.sink = _SyncMetricsSink(
                trial_component_name, sagemaker_session.sagemaker_metrics_client
            )
        else:
            self.sink = sink

    def log_metric(self, metric_name, value, timestamp=None, step=None):
        """Sends a metric to metrics service."""

        metric_data = _RawMetricData(metric_name, value, timestamp, step)
        self.sink.log_metric(metric_data)

    def __enter__(self):
        """Return self"""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Execute self.close()"""
        self.sink.close()

    def close(self):
        """Close the metrics object."""
        self.sink.close()


class _SyncMetricsSink(object):
    """Collects metrics and sends them directly to metrics service."""

    def __init__(self, trial_component_name, metrics_client) -> None:
        """Initialize a `_SyncMetricsSink` instance

        Args:
            trial_component_name (str): The Name of the Trial Component to log metrics.
            metrics_client (boto3.client): boto client for metrics service
        """
        self._trial_component_name = trial_component_name
        self._metrics_client = metrics_client
        self._buffer = []

    def log_metric(self, metric_data):
        """Sends a metric to metrics service."""

        # this is a simplistic solution which calls BatchPutMetrics
        # on the same thread as the client code
        self._buffer.append(metric_data)
        self._drain()

    def _drain(self, close=False):
        """Pops off all metrics in the buffer and starts sending them to metrics service."""

        if not self._buffer:
            return

        if len(self._buffer) < BATCH_SIZE and not close:
            return

        # pop all the available metrics
        available_metrics, self._buffer = self._buffer, []

        self._send_metrics(available_metrics)

    def _send_metrics(self, metrics):
        """Calls BatchPutMetrics directly on the metrics service."""
        while metrics:
            batch, metrics = (
                metrics[:BATCH_SIZE],
                metrics[BATCH_SIZE:],
            )
            request = self._construct_batch_put_metrics_request(batch)
            response = self._metrics_client.batch_put_metrics(**request)
            errors = response["Errors"] if "Errors" in response else None
            if errors:
                message = errors[0]["Message"]
                raise Exception(f'{len(errors)} errors with message "{message}"')

    def _construct_batch_put_metrics_request(self, batch):
        """Creates dictionary object used as request to metrics service."""
        return {
            "TrialComponentName": self._trial_component_name.lower(),
            "MetricData": list(map(lambda x: x.to_raw_metric_data(), batch)),
        }

    def close(self):
        """Drains any remaining metrics."""
        self._drain(close=True)


class _MetricQueue(object):
    """A thread safe queue for sending metrics to SageMaker.

    Args:
        trial_component_name (str): the ARN of the resource
        metric_name (str): the name of the metric
        metrics_client (boto_client): the boto client for SageMaker Metrics service
    """

    _CONSUMER_SLEEP_SECONDS = 5

    def __init__(self, trial_component_name, metric_name, metrics_client):
        # infinite queue size
        self._queue = queue.Queue()
        self._buffer = []
        self._thread = threading.Thread(target=self._run)
        self._started = False
        self._finished = False
        self._trial_component_name = trial_component_name
        self._metrics_client = metrics_client
        self._metric_name = metric_name
        self._logged_metrics = 0

    def log_metric(self, metric_data):
        """Adds a metric data point to the queue"""
        self._buffer.append(metric_data)

        if len(self._buffer) < BATCH_SIZE:
            return

        self._enqueue_all()

        if not self._started:
            self._thread.start()
            self._started = True

    def _run(self):
        """Starts the metric thread which sends metrics to SageMaker in batches"""

        while not self._queue.empty() or not self._finished:
            if self._queue.empty():
                time.sleep(self._CONSUMER_SLEEP_SECONDS)
            else:
                batch = self._queue.get()
                self._send_metrics(batch)

    def _send_metrics(self, metrics_batch):
        """Calls BatchPutMetrics directly on the metrics service."""
        request = self._construct_batch_put_metrics_request(metrics_batch)
        self._logged_metrics += len(metrics_batch)
        self._metrics_client.batch_put_metrics(**request)

    def _construct_batch_put_metrics_request(self, batch):
        """Creates dictionary object used as request to metrics service."""

        return {
            "TrialComponentName": self._trial_component_name,
            "MetricData": list(map(lambda x: x.to_raw_metric_data(), batch)),
        }

    def _enqueue_all(self):
        """Enqueue all buffered metrics to be sent to SageMaker"""

        available_metrics, self._buffer = self._buffer, []
        if available_metrics:
            self._queue.put(available_metrics)

    def close(self):
        """Flushes any buffered metrics"""

        self._enqueue_all()
        self._finished = True

    def is_active(self):
        """Is the thread active (still draining metrics to SageMaker)"""

        return self._thread.is_alive()


class _AsyncMetricsSink(object):
    """Collects metrics and sends them directly to metrics service."""

    _COMPLETE_SLEEP_SECONDS = 1.0

    def __init__(self, trial_component_name, metrics_client) -> None:
        """Initialize a `_AsyncMetricsSink` instance

        Args:
            trial_component_name (str): The Name of the Trial Component to log metrics to.
            metrics_client (boto3.client): boto client for metrics service
        """
        self._trial_component_name = trial_component_name
        self._metrics_client = metrics_client
        self._buffer = []
        self._is_draining = False
        self._metric_queues = {}

    def log_metric(self, metric_data):
        """Sends a metric to metrics service."""

        if metric_data.MetricName in self._metric_queues:
            self._metric_queues[metric_data.MetricName].log_metric(metric_data)
        else:
            cur_metric_queue = _MetricQueue(
                self._trial_component_name, metric_data.MetricName, self._metrics_client
            )
            self._metric_queues[metric_data.MetricName] = cur_metric_queue
            cur_metric_queue.log_metric(metric_data)

    def close(self):
        """Closes the metric file."""
        logging.debug("Closing")
        for q in self._metric_queues.values():
            q.close()

        # TODO should probably use join
        while any(map(lambda x: x.is_active(), self._metric_queues.values())):
            time.sleep(self._COMPLETE_SLEEP_SECONDS)
        logging.debug("Closed")
