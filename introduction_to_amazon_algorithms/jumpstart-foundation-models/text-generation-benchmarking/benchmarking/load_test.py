from concurrent import futures
import datetime
from copy import deepcopy
import logging
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Type, Union
import math
import time

import boto3
import numpy as np
from sagemaker.predictor import Predictor
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

from benchmarking.concurrency_probe import ConcurrentProbeIteratorBase
from benchmarking.constants import (
    CLOUDWATCH_PERIOD_SECONDS,
    SM_INVOCATION_TIMEOUT_SECONDS,
)
from benchmarking.constants import MAX_TOTAL_RETRY_TIME_SECONDS
from benchmarking.constants import RETRY_WAIT_TIME_SECONDS
from benchmarking.logging import logging_prefix
from benchmarking.custom_predictor import CustomPredictor


class PredictionResult(NamedTuple):
    """A NamedTuple responsible for the result of a single endpoint prediction."""

    time_utc_start: datetime.datetime
    time_utc_end: datetime.datetime
    payload: Dict[str, Any]
    result: Any

    def client_latency(self) -> float:
        """The client latency for this single prediction."""
        return (self.time_utc_end - self.time_utc_start).total_seconds() * 1e3

    def input_sequence_num_words(self) -> int:
        """The word count of the input sequence."""
        return self._num_words(self._text_inputs())

    def output_sequence(self) -> str:
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
                elif "generation" in data:
                    data = data["generation"]
                else:
                    raise ValueError("Output data contains dictionary without recognized keys.")

        text_inputs = self._text_inputs()
        if data.startswith(text_inputs):
            data = data[len(text_inputs) :]

        return data

    def num_words(self) -> int:
        """The word count of the output sequence."""
        return self._num_words(self.output_sequence())

    def num_tokens(self, tokenizer: PreTrainedTokenizerBase) -> int:
        """The token count of the output sequence."""
        return len(tokenizer.encode(self.output_sequence()))

    def _text_inputs(self) -> str:
        if "inputs" in self.payload:
            return self.payload["inputs"]
        elif "text_inputs" in self.payload:
            return self.payload["text_inputs"]
        else:
            raise ValueError("Expected input text keys are not detected in payload.")

    @staticmethod
    def _num_words(text: str) -> int:
        return len(text.split())


class BatchInvocationStatistics(NamedTuple):
    """A NamedTuple holding start and stop times for a batch of endpoint predictions."""

    time_utc_start: datetime.datetime
    time_utc_end: datetime.datetime
    num_invocations: int
    results: List[PredictionResult]

    def _duration_seconds(self) -> float:
        """Computes the time in seconds of the batch load test."""
        return (self.time_utc_end - self.time_utc_start).total_seconds()

    def _throughput(self, values: List[int]) -> float:
        """Computes the number of invocation responses per second."""
        return sum(values) / self._duration_seconds()

    def _throughput_robust(self, values: List[int]) -> float:
        """Computes the median throughput of result values in units values per second to eliminate outliers."""
        throughput_values = [
            sum(values[: (i + 1)]) / (result.time_utc_end - self.time_utc_start).total_seconds()
            for i, result in enumerate(self.results)
        ]
        return self._collect_statistics(throughput_values)["Maximum"]

    def get_statistics(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        price_per_endpoint: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Collect statistics on the number of input/output sequence words, the latency, and the latency per word."""
        latency_per_word = self._collect_statistics(
            [x.client_latency() / x.num_words() for x in self.results if x.num_words() > 0]
        )
        word_throughput_robust = self._throughput_robust([x.num_words() for x in self.results])
        time_to_generate_1m_words = 1e6 / word_throughput_robust / 3600
        statistics: Dict[str, Any] = {
            "InputSequenceWords": self._collect_statistics([x.input_sequence_num_words() for x in self.results]),
            "OutputSequenceWords": self._collect_statistics([x.num_words() for x in self.results]),
            "Latency": self._collect_statistics([x.client_latency() for x in self.results]),
            "LatencyPerWord": latency_per_word,
            "TestDuration": self._duration_seconds(),
            "RequestThroughputRobust": self._throughput_robust([1 for _ in self.results]),
            "RequestThroughput": self._throughput([1 for _ in self.results]),
            "WordThroughputRobust": word_throughput_robust,
            "WordThroughput": self._throughput([x.num_words() for x in self.results]),
            "TimeToGenerate1MWords": time_to_generate_1m_words,
        }
        if tokenizer is not None:
            output_sequence_tokens = [x.num_tokens(tokenizer) for x in self.results]
            latency_per_token = self._collect_statistics(
                [x.client_latency() / x.num_tokens(tokenizer) for x in self.results if x.num_tokens(tokenizer) > 0]
            )
            token_throughput = self._throughput(output_sequence_tokens)
            token_throughput_robust = self._throughput_robust(output_sequence_tokens)
            time_to_generate_1m_tokens = 1e6 / token_throughput / 3600
            statistics.update(
                {
                    "OutputSequenceTokens": self._collect_statistics([output_sequence_tokens for x in self.results]),
                    "LatencyPerToken": latency_per_token,
                    "TokenThroughputRobust": token_throughput_robust,
                    "TokenThroughput": token_throughput,
                    "TimeToGenerate1MTokens": time_to_generate_1m_tokens,
                }
            )
        if price_per_endpoint is not None:
            statistics.update({"CostToGenerate1MTokens": time_to_generate_1m_tokens * price_per_endpoint})
            if tokenizer is not None:
                statistics.update({"CostToGenerate1MTokens": time_to_generate_1m_tokens * price_per_endpoint})
        return statistics

    @staticmethod
    def _collect_statistics(data: List[Union[float, int]]) -> Dict[str, Any]:
        return {
            "Median": np.median(data).item(),
            "Average": np.average(data).item(),
            "Minimum": np.amin(data).item(),
            "Maximum": np.amax(data).item(),
            "p50": np.quantile(data, 0.50).item(),
            "p90": np.quantile(data, 0.90).item(),
            "p95": np.quantile(data, 0.95).item(),
            "p99": np.quantile(data, 0.99).item(),
        }


class LoadTester:
    def __init__(
        self,
        predictor: CustomPredictor,
        payload: Dict[str, Any],
        model_id: str,
        payload_name: str,
        tokenizer_model_id: Optional[str] = None,
        huggingface_hub_token: Optional[str] = None,
        price_per_endpoint: Optional[float] = None,
    ) -> None:
        self.predictor = predictor
        self.payload = payload
        self.model_id = model_id
        self.payload_name = payload_name
        self._logging_prefix = logging_prefix(self.model_id, self.payload_name)
        if tokenizer_model_id is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, token=huggingface_hub_token)
        else:
            self.tokenizer = None
        self.price_per_endpoint = price_per_endpoint

    def predict_once_and_collect_client_results(self) -> PredictionResult:
        """Perform a single endpoint prediction and produce a PredictionResult."""
        time_utc_start = datetime.datetime.utcnow()
        result = self.predictor.predict(self.payload)
        time_utc_end = datetime.datetime.utcnow()
        return PredictionResult(time_utc_start, time_utc_end, self.payload, result)

    def run_load_test(self, num_invocations: int, max_workers: int) -> Optional[BatchInvocationStatistics]:
        """Concurrently invoke an endpoint prediction multiple times and gather results in BatchInvocationStatistics."""
        time_utc_start = datetime.datetime.utcnow()
        timeout_seconds = SM_INVOCATION_TIMEOUT_SECONDS * num_invocations / max_workers
        _logging_prefix = logging_prefix(self.model_id, self.payload_name, max_workers)

        logging.info(f"{_logging_prefix} Begin throughput load test ...")

        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_list = [
                executor.submit(self.predict_once_and_collect_client_results) for _ in range(num_invocations)
            ]
            done, not_done = futures.wait(
                futures_list,
                timeout=timeout_seconds,
                return_when=futures.FIRST_EXCEPTION,
            )
            for future in done:
                if future._exception is not None:
                    logging.info(f"{_logging_prefix} Cancelling and awaiting future completion: {future._exception}")
                    if not_done:
                        self._cancel_futures_and_wait(not_done)
                    raise future._exception

            if not_done:
                logging.info(f"{_logging_prefix} Cancelling and awaiting future completion: Load test timeout.")
                self._cancel_futures_and_wait(not_done)
                raise TimeoutError("Load test timeout.")

            results = [future.result(timeout=0.0) for future in futures_list]

        time_utc_end = datetime.datetime.utcnow()
        return BatchInvocationStatistics(time_utc_start, time_utc_end, num_invocations, results)

    @staticmethod
    def _cancel_futures_and_wait(
        futures_list: Set[futures.Future],
        timeout: float = SM_INVOCATION_TIMEOUT_SECONDS,
    ) -> None:
        if futures_list:
            for future_to_cancel in futures_list:
                future_to_cancel.cancel()
            futures.wait(futures_list, timeout=timeout, return_when=futures.ALL_COMPLETED)

    def run_latency_load_test(
        self,
        num_invocations: int,
        retry_wait_time: float = RETRY_WAIT_TIME_SECONDS,
        max_total_retry_time: float = MAX_TOTAL_RETRY_TIME_SECONDS,
    ) -> Dict[str, Any]:
        logging.info(f"{self._logging_prefix} Begin latency load test ...")
        time.sleep(CLOUDWATCH_PERIOD_SECONDS)  # wait for 1 cloudwatch period to ensure no extra queries are reported
        statistics_latency = self.run_load_test(num_invocations, 1)
        metrics = self._extract_cloudwatch_metrics(statistics_latency, retry_wait_time, max_total_retry_time)
        metrics["Client"] = statistics_latency.get_statistics(self.tokenizer, self.price_per_endpoint)
        metrics["ModelID"] = self.model_id
        metrics["Invocations"] = num_invocations
        metrics["ConcurrentRequests"] = max_workers
        metrics["PayloadName"] = self.payload_name
        return metrics

    def run_throughput_load_test(self, num_invocations: int, max_workers: int) -> Dict[str, Any]:
        statistics_throughput = self.run_load_test(num_invocations, max_workers)
        metrics = statistics_throughput.get_statistics(self.tokenizer, self.price_per_endpoint)
        metrics.update(
            {
                "ModelID": self.model_id,
                "PayloadName": self.payload_name,
                "Invocations": num_invocations,
                "ConcurrentRequests": max_workers,
            }
        )
        return metrics

    def run_concurrency_probe(
        self,
        iterator_cls: Type[ConcurrentProbeIteratorBase],
        num_invocation_hook: Callable[[int], int],
    ) -> List[Dict[str, Any]]:
        """Probe the endpoint with a series of concurrent request payloads to obtain a set of throughput measures.

        Arguments:
            concurrent_request_iterator (ConcurrentRequestIteratorBase): An iterator that controls the number of
                concurrent requests to load the endpoint with during the probe.
            num_invocation_hook (Callable[[int], int]): A hook that controls the number of invocations use during a
                single load test as a function of the number of concurrent requests.
        """
        logging.info(f"{self._logging_prefix} Begin concurrency probe ...")
        results: List[Dict[str, Any]] = []
        concurrent_request_iterator = iterator_cls(self.model_id, self.payload_name)
        for concurrent_requests in concurrent_request_iterator:
            try:
                num_invocations = num_invocation_hook(concurrent_requests)
                result = self.run_throughput_load_test(num_invocations, concurrent_requests)
                if concurrent_request_iterator.send(result, self.predictor):
                    results.append(result)
            except Exception as e:
                concurrent_request_iterator.exception = e

        logging.info(f"{self._logging_prefix} End concurrency probe. {concurrent_request_iterator.stop_reason}")
        return results

    def _query_cloudwatch_get_metric_statistics(
        self, start_time: datetime.datetime, end_time: datetime.datetime
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
                    {"Name": "EndpointName", "Value": self.predictor.endpoint_name},
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

    def _extract_cloudwatch_metrics(
        self,
        invocation_statistics: BatchInvocationStatistics,
        retry_wait_time: float,
        max_total_retry_time: float,
    ) -> Dict[str, Any]:
        """Iteratively query the Amazon CloudWatch GetMetricStatistics API to obtain load test statistics.

        Because endpoints emit data points to CloudWatch on periodic intervals, we must wait to obtain metrics until
        after all data points associated with the load test are emitted. This could take about 1 minute to complete.
        """
        retry_duration_cumulative = 0.0
        retry = True
        while retry is True:
            start_time = invocation_statistics.time_utc_start
            delay_time = datetime.timedelta(seconds=retry_duration_cumulative)
            end_time = invocation_statistics.time_utc_end + delay_time
            metrics = self._query_cloudwatch_get_metric_statistics(start_time, end_time)
            sample_count_latency = metrics["ModelLatency"].get("SampleCount", 0)
            num_invocations = invocation_statistics.num_invocations
            if (sample_count_latency < num_invocations) and (retry_duration_cumulative < max_total_retry_time):
                logging.info(
                    f" - {self._logging_prefix} Sample count {sample_count_latency} < {num_invocations}. "
                    f"Retrying in {retry_wait_time} seconds ..."
                )
                retry_duration_cumulative += retry_wait_time
                time.sleep(retry_wait_time)
            else:
                retry = False
        return metrics
