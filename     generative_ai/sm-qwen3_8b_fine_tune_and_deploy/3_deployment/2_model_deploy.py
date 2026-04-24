"""
Deploy fine-tuned Qwen3-VL model to SageMaker using ModelBuilder.

ModelBuilder notes:
  - Custom container passthrough: when image_uri is a non-first-party ECR image
    and no model/inference_spec is set, ModelBuilder skips model packaging.
  - No model_server needed (only for known servers like TGI, DJL, etc.).
  - No SchemaBuilder needed (the custom container handles serialization).
  - s3_model_data_url points directly to the HuggingFace model files on S3.
  - deploy() with AsyncInferenceConfig is a first-class supported path.
"""

import boto3
from sagemaker.serve import ModelBuilder
from sagemaker.session import Session
from sagemaker.core.inference_config import AsyncInferenceConfig

# ==========================================
# 1. Configuration
# ==========================================
REGION = "us-east-2"
AWS_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"

# S3 path to HuggingFace-format model (safetensors + config.json + tokenizer)
# After EasyR1 training + model_merger.py, this is typically:
#   s3://<bucket>/<training-job>/output/model/global_step_X/actor/huggingface/
S3_MODEL_DATA_URL = "s3://your-bucket/path/to/huggingface-model/"

ENDPOINT_NAME = "your-model-endpoint"
MODEL_NAME = ENDPOINT_NAME

# Custom vLLM container (build with docker/build_and_push.sh)
ECR_REPO_NAME = "qwen3-vl-sagemaker-inference"
IMAGE_TAG = "latest"
CONTAINER_IMAGE = f"{AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"

# Environment variables for vLLM container
# SM_VLLM_ prefix — parsed by sagemaker-entrypoint.sh into vLLM CLI args
VLLM_ENV = {
    "SM_VLLM_MODEL": "/opt/ml/model",
    "SM_VLLM_MAX_MODEL_LEN": "2048",
    "SM_VLLM_DTYPE": "bfloat16",
    "SM_VLLM_TRUST_REMOTE_CODE": "true",
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
    "SM_VLLM_ENFORCE_EAGER": "true",
}

# ==========================================
# 2. Initialize SageMaker session
# ==========================================
sagemaker_session = Session(boto3.Session(region_name=REGION))

# ==========================================
# 3. Build and deploy with ModelBuilder
# ==========================================
# Custom container passthrough: no model_server or SchemaBuilder needed.
# ModelBuilder detects a non-first-party ECR image and skips model packaging.
print("=" * 60)
print("Building model with ModelBuilder...")
print("=" * 60)

model_builder = ModelBuilder(
    image_uri=CONTAINER_IMAGE,
    s3_model_data_url=S3_MODEL_DATA_URL,
    role_arn=ROLE_ARN,
    env_vars=VLLM_ENV,
    sagemaker_session=sagemaker_session,
)

model = model_builder.build(model_name=MODEL_NAME)
print(f"Model built: {MODEL_NAME}")

# Deploy as async endpoint
print(f"Deploying async endpoint: {ENDPOINT_NAME}")
endpoint = model_builder.deploy(
    endpoint_name=ENDPOINT_NAME,
    instance_type="ml.g6.2xlarge",
    initial_instance_count=1,
    inference_config=AsyncInferenceConfig(
        output_path=f"s3://sagemaker-{REGION}-{AWS_ACCOUNT_ID}/async_inference_output",
        max_concurrent_invocations_per_instance=10,
    ),
)
print(f"Endpoint {ENDPOINT_NAME} is being created (10-15 min)...")

# ==========================================
# 4. Test inference (after endpoint is InService)
# ==========================================
# Async endpoints require uploading the payload to S3 first,
# then calling invoke_async with the S3 input location.
#
# import json
# import uuid
#
# s3 = boto3.client("s3", region_name=REGION)
# bucket = f"sagemaker-{REGION}-{AWS_ACCOUNT_ID}"
#
# payload = {
#     "model": "/opt/ml/model",
#     "messages": [{"role": "user", "content": "What is 2+2?"}],
#     "max_tokens": 128,
# }
# input_key = f"async_inference_input/{uuid.uuid4()}.json"
# s3.put_object(
#     Bucket=bucket,
#     Key=input_key,
#     Body=json.dumps(payload),
#     ContentType="application/json",
# )
#
# response = endpoint.invoke_async(
#     input_location=f"s3://{bucket}/{input_key}",
#     content_type="application/json",
# )
# print(f"Output location: {response.output_location}")
# # Poll response.output_location for the result once the job completes

# ==========================================
# 5. Cleanup (when done)
# ==========================================
# endpoint.delete_model()
# endpoint.delete_endpoint()
