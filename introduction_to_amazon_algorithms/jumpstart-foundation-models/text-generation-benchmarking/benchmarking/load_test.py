from concurrent import futures
import datetime
from copy import deepcopy
from itertools import repeat
from typing import Any, Dict, List, NamedTuple, Optional, Union
import math
import time

import boto3
import numpy as np
from sagemaker.predictor import Predictor


class PredictionResult(NamedTuple):
    """A NamedTuple responsible for the result of a single endpoint prediction."""

    time_utc_start: datetime.datetime
    time_utc_end: datetime.datetime
    payload: Dict[str, Any]
    result: Any

    def client_latency(self):
        """The client latency for this single prediction."""
        return (self.time_utc_end - self.time_utc_start).total_seconds()

    def input_sequence_num_words(self):
        """The word count of the input sequence."""
        return self._num_words(self.payload["text_inputs"])

    def output_sequence(self):
        """The output sequence with the input sequence prefix removed.

        Text generation models, by default, include the input sequence in the model response.
        """
        data = deepcopy(self.result)
        while not isinstance(data, str):
            if isinstance(data, list):
                data = data[0]
            if isinstance(data, dict):
                if "generated_text" in data:
                    data = data["generated_text"]
                elif "generated_texts" in data:
                    data = data["generated_texts"]
                else:
                    raise ValueError("Output data contains dictionary without recognized keys.")

        text_inputs = self.payload["text_inputs"]
        if data.startswith(text_inputs):
            data = data[len(text_inputs) :]

        return data

    def output_sequence_num_words(self):
        """The word count of the output sequence."""
        return self._num_words(self.output_sequence())

    @staticmethod
    def _num_words(text: str) -> int:
        return len(text.split())


class BatchInvocationStatistics(NamedTuple):
    """A NamedTuple holding start and stop times for a batch of endpoint predictions."""

    time_utc_start: datetime.datetime
    time_utc_end: datetime.datetime
    num_invocations: int
    results: List[PredictionResult]

    def duration_seconds(self) -> float:
        """Computes the time in seconds of the batch load test."""
        return (self.time_utc_end - self.time_utc_start).total_seconds()

    def throughput(self) -> float:
        """Computes the number of invocation responses per second."""
        return self.num_invocations / self.duration_seconds()

    def get_client_statistics(self) -> Dict[str, Any]:
        """Collect statistics on the number of input/output sequence words, the latency, and the latency per word."""
        return {
            "InputSequenceWords": self._collect_statistics([x.input_sequence_num_words() for x in self.results]),
            "OutputSequenceWords": self._collect_statistics([x.output_sequence_num_words() for x in self.results]),
            "Latency": self._collect_statistics([x.client_latency() for x in self.results]),
            "LatencyPerOutputWord": self._collect_statistics(
                [
                    x.client_latency() / x.output_sequence_num_words()
                    for x in self.results
                    if x.output_sequence_num_words() > 0
                ]
            ),
        }

    @staticmethod
    def _collect_statistics(data: List[Union[float, int]]) -> Dict[str, Any]:
        return {
            "Average": np.average(data).item(),
            "Minimum": np.amin(data).item(),
            "Maximum": np.amax(data).item(),
            "p50": np.quantile(data, 0.50).item(),
            "p90": np.quantile(data, 0.90).item(),
            "p95": np.quantile(data, 0.95).item(),
        }


def predict_once_and_collect_client_results(predictor: Predictor, payload: Dict[str, Any]):
    """Perform a single endpoint prediction and produce a PredictionResult."""
    time_utc_start = datetime.datetime.utcnow()
    result = predictor.predict(payload)
    time_utc_end = datetime.datetime.utcnow()
    return PredictionResult(time_utc_start, time_utc_end, payload, result)


def run_load_test(
    predictor: Predictor,
    payload: Dict[str, Any],
    num_invocations: int,
    max_workers: int,
) -> BatchInvocationStatistics:
    """Concurrently invoke an endpoint prediction multiple times and gather results in BatchInvocationStatistics."""
    time.sleep(60.0)  # wait for 1 cloudwatch period to ensure no extra queries are reported
    time_utc_start = datetime.datetime.utcnow()
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        predictors = repeat(predictor, num_invocations)
        payloads = repeat(payload, num_invocations)
        results = executor.map(predict_once_and_collect_client_results, predictors, payloads)

    results = list(results)
    time_utc_end = datetime.datetime.utcnow()
    return BatchInvocationStatistics(time_utc_start, time_utc_end, num_invocations, results)


def query_cloudwatch_get_metric_statistics(
    endpoint_name: str, start_time: datetime.datetime, end_time: datetime.datetime
) -> Dict[str, Any]:
    """Obtain load test statistics from the Amazon CloudWatch GetMetricStatistics API."""
    cloudwatch = boto3.client("cloudwatch")
    sample_count = ["SampleCount"]
    statistics = ["Average", "Minimum", "Maximum"]
    extended = ["p50", "p90", "p95"]

    period = math.ceil((end_time - start_time).total_seconds() / 60) * 60

    datapoints = {}
    for metric_name in ["ModelLatency", "OverheadLatency"]:
        metrics = cloudwatch.get_metric_statistics(
            MetricName=metric_name,
            Dimensions=[
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "VariantName", "Value": "AllTraffic"},
            ],
            Namespace="AWS/SageMaker",
            StartTime=start_time.isoformat(),
            EndTime=end_time.isoformat(),
            Period=period,
            Statistics=sample_count + statistics,
            ExtendedStatistics=extended,
        )
        _datapoints = next(iter(metrics.get("Datapoints")), {})
        datapoints[metric_name] = {
            **{x: _datapoints.get(x, 0) for x in sample_count},
            **{x: _datapoints.get(x, 0) / 1e6 for x in statistics},
            **{x: _datapoints.get("ExtendedStatistics", {}).get(x, 0) / 1e6 for x in extended},
        }
    return datapoints


def extract_cloudwatch_metrics(
    invocation_statistics: BatchInvocationStatistics,
    model_id: str,
    payload_name: str,
    endpoint_name: str,
    retry_wait_time: float,
    max_total_retry_time: float,
) -> Dict[str, Any]:
    """Iteratively query the Amazon CloudWatch GetMetricStatistics API to obtain load test statistics.

    Because endpoints emit data points to CloudWatch on periodic intervals, we must wait to obtain metrics until after
    all data points associated with the load test are emitted. This could take about 1 minute to complete.
    """
    retry_duration_cumulative = 0.0
    retry = True
    while retry is True:
        start_time = invocation_statistics.time_utc_start
        delay_time = datetime.timedelta(seconds=retry_duration_cumulative)
        end_time = invocation_statistics.time_utc_end + delay_time
        metrics = query_cloudwatch_get_metric_statistics(endpoint_name, start_time, end_time)
        sample_count_latency = metrics["ModelLatency"].get("SampleCount", 0)
        num_invocations = invocation_statistics.num_invocations
        if (sample_count_latency < num_invocations) and (retry_duration_cumulative < max_total_retry_time):
            print(
                f" - {logging_prefix(model_id, payload_name)} Sample count {sample_count_latency} < {num_invocations}. "
                f"Retrying in {retry_wait_time} seconds ..."
            )
            retry_duration_cumulative += retry_wait_time
            time.sleep(retry_wait_time)
        else:
            retry = False
    return metrics


def logging_prefix(model_id: str, payload_name: Optional[str] = None) -> str:
    """A standardized prefix for all console logs."""
    items = [f"Model '{model_id}'"]
    if payload_name is not None:
        items.append(f"Payload '{payload_name}'")
    return f"({', '.join(items)}):"


def run_benchmarking_load_tests(
    predictor: Predictor,
    payload: Dict[str, Any],
    model_id: str,
    payload_name: str,
    num_invocations: int,
    max_workers: int,
    retry_wait_time: float,
    max_total_retry_time: float,
) -> Dict[str, Any]:
    """Run all benchmarks and load tests on the provided Predictor."""
    print(f"{logging_prefix(model_id, payload_name)} Begin latency load test ...")
    statistics_latency = run_load_test(predictor, payload, num_invocations, 1)
    metrics = extract_cloudwatch_metrics(
        statistics_latency,
        model_id,
        payload_name,
        predictor.endpoint_name,
        retry_wait_time,
        max_total_retry_time,
    )
    metrics["Client"] = statistics_latency.get_client_statistics()

    print(f"{logging_prefix(model_id, payload_name)} Begin throughput load test ...")
    statistics_throughput = run_load_test(predictor, payload, num_invocations, max_workers)
    metrics["Throughput"] = statistics_throughput.throughput()
    metrics["WordThroughput"] = metrics["Throughput"] * metrics["Client"]["OutputSequenceWords"]["Average"]

    print(f"{logging_prefix(model_id, payload_name)} Finished benchmarking load tests ...")
    metrics["ModelID"] = model_id
    metrics["PayloadName"] = payload_name
    metrics["SampleOutput"] = statistics_throughput.results[0].output_sequence()

    return metrics
