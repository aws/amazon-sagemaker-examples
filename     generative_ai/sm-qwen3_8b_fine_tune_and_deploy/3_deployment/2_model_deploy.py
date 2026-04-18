import boto3
from sagemaker.serve import ModelBuilder
from sagemaker.session import Session
from sagemaker.core.resources import Model, EndpointConfig, Endpoint
from sagemaker.core.shapes import (
    Container,
    ModelDataSource,
    S3DataSource,
    ProductionVariant,
    AsyncInferenceConfig,
    AsyncInferenceOutputConfig,
    AsyncInferenceClientConfig,
)

# 1. Setup environment
REGION = 'us-east-2'
AWS_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
role = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'

# Initialize SageMaker session
sagemaker_session = Session(boto3.Session(region_name=REGION))

# ==========================================
# Model configuration
# ==========================================
s3_model_uri = "s3://your-bucket/path/to/model/"
endpoint_name = "your-model-endpoint"
model_name = endpoint_name
endpoint_config_name = f"{endpoint_name}-config"

# Custom vLLM container (build with docker/build_and_push.sh)
ECR_REPO_NAME = "qwen3-vl-sagemaker-inference"
IMAGE_TAG = "latest"
container_image = f"{AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"

# Environment variables for vLLM container
vllm_env = {
    # SM_VLLM_ prefix for custom container entrypoint
    "SM_VLLM_MODEL": "/opt/ml/model",
    "SM_VLLM_MAX_MODEL_LEN": "2048",
    "SM_VLLM_DTYPE": "bfloat16",
    "SM_VLLM_TRUST_REMOTE_CODE": "true",
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
    "SM_VLLM_ENFORCE_EAGER": "true",
}

# ==========================================
# Using SageMaker v3 API with Async Inference
# ==========================================
print("Creating model with SageMaker v3 API...")

# Create Model using v3 API
model = Model.create(
    model_name=model_name,
    execution_role_arn=role,
    primary_container=Container(
        image=container_image,
        model_data_source=ModelDataSource(
            s3_data_source=S3DataSource(
                s3_uri=s3_model_uri,
                s3_data_type="S3Prefix",
                compression_type="None",
            )
        ),
        environment=vllm_env,
    ),
)
print(f"Created model: {model.model_name}")

# Create Endpoint Config with Async Inference
print("Creating endpoint config with async inference...")
endpoint_config = EndpointConfig.create(
    endpoint_config_name=endpoint_config_name,
    production_variants=[
        ProductionVariant(
            variant_name="AllTraffic",
            model_name=model_name,
            initial_instance_count=1,
            instance_type="ml.g6.2xlarge",
        )
    ],
    async_inference_config=AsyncInferenceConfig(
        output_config=AsyncInferenceOutputConfig(
            s3_output_path=f"s3://sagemaker-{REGION}-{AWS_ACCOUNT_ID}/async_inference_output",
        ),
        client_config=AsyncInferenceClientConfig(
            max_concurrent_invocations_per_instance=10,
        ),
    ),
)
print(f"Created endpoint config: {endpoint_config.endpoint_config_name}")

# Create Endpoint
print(f"Creating async endpoint: {endpoint_name}")
endpoint = Endpoint.create(
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_name,
)
print(f"Endpoint {endpoint_name} is being created...")
print("This may take 10-15 minutes...")

# Optional: Wait for endpoint to be in service
# endpoint.wait_for_status(status="InService", poll_seconds=30)
# print(f"Endpoint {endpoint_name} is now InService!")
