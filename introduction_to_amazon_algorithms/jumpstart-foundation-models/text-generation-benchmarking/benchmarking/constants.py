import boto3
from botocore.config import Config
from sagemaker.session import Session
from pathlib import Path


SAVE_METRICS_FILE_PATH = Path.cwd() / "latency_benchmarking.json"
CLOUDWATCH_PERIOD = 60.0
MAX_CONCURRENT_INVOCATIONS_PER_MODEL = 30
MAX_CONCURRENT_BENCHMARKS = 15
RETRY_WAIT_TIME_SECONDS = 30.0
MAX_TOTAL_RETRY_TIME_SECONDS = 120.0
NUM_INVOCATIONS = 10
SM_SESSION = Session(
    sagemaker_client=boto3.client(
        "sagemaker",
        config=Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 20}),
    )
)
MODEL_ID_TO_HF_REPO_ID = {
    "huggingface-llm-falcon-7b-bf16": "tiiuae/falcon-7b",
    "huggingface-llm-falcon-7b-instruct-bf16": "tiiuae/falcon-7b-instruct",
    "huggingface-llm-falcon-40b-bf16": "tiiuae/falcon-40b",
    "huggingface-llm-falcon-40b-instruct-bf16": "tiiuae/falcon-40b-instruct",
    "huggingface-llm-falcon-180b-bf16": "tiiuae/falcon-7b",
    "huggingface-llm-falcon-180b-chat-bf16": "tiiuae/falcon-7b",
    "meta-textgeneration-llama-2-7b": "TheBloke/Llama-2-7B-GPTQ",
    "meta-textgeneration-llama-2-13b": "TheBloke/Llama-2-13B-GPTQ",
    "meta-textgeneration-llama-2-70b": "TheBloke/Llama-2-70B-GPTQ"
}
