
from abc import abstractmethod
from typing import Any, Dict, Optional

from sagemaker.predictor import Predictor


class ConcurrentProbeIteratorBase:
    def __init__(self):
        self.exception: Optional[Exception] = None
        self.stop_reason: str = "No stop reason set."
        self.result: Dict[str, Any] = None

    def __iter__(self) -> 'ConcurrentProbeIteratorBase':
        return self
    
    @abstractmethod
    def __next__(self) -> int:
        raise NotImplementedError

    def send(self, result: Dict[str, Any], predictor: Predictor) -> bool:
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
        start: int = 1,
        scale_factor: float = 2.0,
        max_latency_seconds: float = 25.0,
    ) -> None:
        self.concurrent_requests = start
        self.scale_factor = scale_factor
        self.max_latency_seconds = max_latency_seconds
        super().__init__()
    
    def __next__(self) -> int:
        if self.exception is not None:
            self.stop_reason = f"Error occured: {self.exception}"
            raise StopIteration

        if self.result is None:
            return self.concurrent_requests
        
        last_latency_seconds = self.result["Latency"]["p90"] / 1e3
        if (last_latency_seconds > self.max_latency_seconds):
            self.stop_reason = f"Last p90 latency = {last_latency_seconds} > {self.max_latency_seconds}."
            raise StopIteration
        
        self.concurrent_requests = int(self.concurrent_requests * self.scale_factor)

        return self.concurrent_requests


def num_invocation_scaler(concurrent_requests: int, num_invocation_factor: int = 3) -> int:
    return concurrent_requests * num_invocation_factor
