# Offline process-monitor regression tests

From the repository root, with Python 3.11–3.13:

```sh
python -m pip install 'rich>=13,<16' 'black==26.10.1'
python -m unittest discover -s '   inference/sm-app_autoscaling_realtime_endpoints_inference_components/tests' -v
python -m black -l 100 --check '   inference/sm-app_autoscaling_realtime_endpoints_inference_components/utils/llmperf.py' '   inference/sm-app_autoscaling_realtime_endpoints_inference_components/tests'
```

The monitor drains stdout and stderr while waiting for completion, reports
progress every 15 seconds, then prints the completed output once. Undecodable
UTF-8 bytes are replaced for display. A nonzero exit is logged as before; the
caller still owns the process and its failure-handling policy.

Tests cover timeout retries, nonzero exits, invalid UTF-8, and a local subprocess
writing 1 MiB to each pipe. The real-child test has a five-second watchdog and
cleans up its process on failure. Tests do not launch LLMPerf, invoke endpoints,
use AWS credentials, or validate autoscaling behavior.

`communicate` buffers output in memory, as the previous implementation did once
the child exited. For unbounded log streams, redirect output to files instead.
