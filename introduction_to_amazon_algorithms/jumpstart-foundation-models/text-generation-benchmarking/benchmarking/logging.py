import logging
from typing import Optional
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s : %(message)s",
    stream=sys.stdout,
)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def logging_prefix(
    model_id: str,
    payload_name: Optional[str] = None,
    concurrent_requests: Optional[int] = None,
) -> str:
    """A standardized prefix for all console logs."""
    items = [f"Model '{model_id}'"]
    if payload_name is not None:
        items.append(f"Payload '{payload_name}'")
    if concurrent_requests is not None:
        items.append(f"Concurrency {concurrent_requests}")
    return f"({', '.join(items)}):"
