import boto3

# 1. Setup environment
REGION =  'us-east-2'
sm_client = boto3.client("sagemaker", region_name=REGION)
role = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'

# ==========================================
# Model configuration
# ==========================================
s3_model_uri = "s3://your-bucket/path/to/model/"
endpoint_name = "your-model-endpoint"
model_name = endpoint_name
endpoint_config_name = endpoint_name

# Custom vLLM container (build with docker/build_and_push.sh)
AWS_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
ECR_REPO_NAME = "qwen3-vl-sagemaker-inference"
IMAGE_TAG = "latest"
container_image = f"{AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"

# Create Model
sm_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": container_image,
        "ModelDataSource": {
            "S3DataSource": {
                "S3Uri": s3_model_uri,
                "S3DataType": "S3Prefix",
                "CompressionType": "None"
            }
        },
        "Environment": {
            # SM_VLLM_ prefix for custom container entrypoint
            "SM_VLLM_MODEL": "/opt/ml/model",
            "SM_VLLM_MAX_MODEL_LEN": "2048",
            "SM_VLLM_DTYPE": "bfloat16",
            "SM_VLLM_TRUST_REMOTE_CODE": "true",
            "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
            "SM_VLLM_ENFORCE_EAGER": "true",
        }
    },
    ExecutionRoleArn=role
)
print(f"Created model: {model_name}")

# Create Endpoint Config (Async)
sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.g6.2xlarge",
    }],
    AsyncInferenceConfig={
        "OutputConfig": {
            "S3OutputPath": f"s3://sagemaker-{REGION}-{AWS_ACCOUNT_ID}/async_inference_output"
        },
        "ClientConfig": {
            "MaxConcurrentInvocationsPerInstance": 10
        }
    }
)
print(f"Created endpoint config: {endpoint_config_name}")

# Create Endpoint
sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
# print(f"Creating endpoint: {endpoint_name}")
print("This may take 10-15 minutes...")
