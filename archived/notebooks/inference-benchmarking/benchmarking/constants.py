import boto3
from botocore.config import Config
from sagemaker.session import Session
from pathlib import Path


SAVE_METRICS_FILE_PATH = Path.cwd() / "latency_benchmarking.json"
CLOUDWATCH_PERIOD_SECONDS = 60.0
MAX_CONCURRENT_INVOCATIONS_PER_MODEL = 30
MAX_CONCURRENT_BENCHMARKS = 20
RETRY_WAIT_TIME_SECONDS = 30.0
MAX_TOTAL_RETRY_TIME_SECONDS = 120.0
NUM_INVOCATIONS = 10
SM_INVOCATION_TIMEOUT_SECONDS = 60.0
SM_SESSION = Session(
    sagemaker_runtime_client=boto3.client(
        "sagemaker-runtime",
        config=Config(connect_timeout=5, retries={"mode": "standard", "total_max_attempts": 10}),
    ),
    sagemaker_client=boto3.client(
        "sagemaker",
        config=Config(connect_timeout=5, read_timeout=60, retries={"total_max_attempts": 20}),
    ),
)
