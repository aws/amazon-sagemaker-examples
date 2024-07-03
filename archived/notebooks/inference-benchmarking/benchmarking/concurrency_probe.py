from abc import abstractmethod
from typing import Any, Dict, Optional

from sagemaker.predictor import Predictor
from benchmarking.custom_predictor import CustomPredictor


class ConcurrentProbeIteratorBase:
    def __init__(self, model_id: str, payload_name: str):
        self.model_id = model_id
        self.payload_name = payload_name
        self.exception: Optional[Exception] = None
        self.stop_reason: str = "No stop reason set."
        self.result: Dict[str, Any] = None

    def __iter__(self) -> "ConcurrentProbeIteratorBase":
        return self

    @abstractmethod
    def __next__(self) -> int:
        raise NotImplementedError

    def send(self, result: Dict[str, Any], predictor: CustomPredictor) -> bool:
        """Send load test results to the iterator and return whether to use results.

        Some iterators may make internal adjustments (e.g., scale endpoint instances and repeat load test for the same
        conccurent request setting) before using the results.
        """
        self.result = result
        return True


class ConcurrentProbeExponentialScalingIterator(ConcurrentProbeIteratorBase):
    """An iterator used during a concurrency probe to exponentially scale concurrent requests."""

    def __init__(
        self,
        model_id: str,
        payload_name: str,
        start: int = 1,
        scale_factor: float = 2.0,
    ) -> None:
        self.concurrent_requests = start
        self.scale_factor = scale_factor
        super().__init__(model_id, payload_name)

    def __next__(self) -> int:
        if self.exception is not None:
            e = self.exception
            self.stop_reason = "".join([type(e).__name__, f": {e}" if str(e) else ""])
            raise StopIteration

        if self.result is None:
            return self.concurrent_requests

        self.concurrent_requests = int(self.concurrent_requests * self.scale_factor)

        return self.concurrent_requests


def num_invocation_scaler(concurrent_requests: int, num_invocation_factor: int = 3) -> int:
    return concurrent_requests * num_invocation_factor
