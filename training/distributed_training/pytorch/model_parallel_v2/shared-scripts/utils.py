"""Utils."""

import os
import re
import subprocess
from typing import Any, Dict, Optional

import numpy


def parse_nccl_test_log(log_file: str) -> Optional[numpy.ndarray]:
    """Parse NCCL test log file.

    Sample output with 2 nodes:

    # Out of bounds values : 0 OK
    # Avg bus bandwidth    : 29.8872
    #
    # Out of bounds values : 0 OK
    # Avg bus bandwidth    : 28.9057
    #
    """
    try:
        with subprocess.Popen(
            f"grep 'Avg bus bandwidth' {log_file}",
            shell=True, stdout=subprocess.PIPE, encoding="UTF-8",
        ) as pipe:
            result, _ = pipe.communicate()
    except Exception as _:
        return None

    bandwidth = []
    for line in str(result).split(os.linesep):
        line = line.strip()
        if not line:
            continue
        splits = line.split(":")
        if len(splits) == 2 and re.match(r"^\d+\.\d*$", splits[-1].strip()):
            bandwidth.append(float(splits[-1].strip()))

    return numpy.array(bandwidth)


def get_nccl_test_report(bandwidth: Optional[numpy.ndarray]) -> Optional[Dict[str, Any]]:
    """Get the complete NCCL test report."""
    if bandwidth is None:
        return None

    bandwidth = bandwidth.reshape((-1,))
    size = len(bandwidth)
    if not size:
        return None

    data_sorted = numpy.sort(bandwidth)
    report = {
        "data": bandwidth,
        "data_sorted": data_sorted,
        "len": size,
        # Stats.
        "max": numpy.max(bandwidth),
        "mean": numpy.mean(bandwidth),
        "median": numpy.median(bandwidth),
        "min": numpy.min(bandwidth),
        "std": numpy.std(bandwidth),
    }

    for index in range(2, 6):
        if size < index:
            break

        report.update({
            f"max{index}": data_sorted[-index],
            f"min{index}": data_sorted[index - 1],
        })

    return report
