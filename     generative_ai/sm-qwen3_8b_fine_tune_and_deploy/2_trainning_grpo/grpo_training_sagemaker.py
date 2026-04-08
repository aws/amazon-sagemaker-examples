"""
SageMaker GRPO Training Job Launcher
Uses EasyR1/veRL framework with custom Docker image.
"""

import boto3
import json
from datetime import datetime
from pprint import pprint

# ==========================================
# Configuration
# ==========================================
REGION = 'us-east-2'  # Changed to us-east-2 for g6e availability
ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'

# Model configuration
# Option 1: Use HuggingFace model ID in config.yml (default)
# Option 2: Use model from S3 (set path below)
MODEL_S3 = "s3://your-bucket/models/your-model/"  # e.g., "s3://your-bucket/models/your-model/"

# Dataset configuration (S3 paths)
TRAINING_DATASET = "s3://your-bucket/train.parquet"
EVAL_DATASET = "s3://your-bucket/eval.parquet"  # Optional
S3_BUCKET = "sagemaker-us-east-2-YOUR_ACCOUNT_ID"  # us-east-2 bucket

# Output configuration
S3_OUTPUT_PATH = f"s3://{S3_BUCKET}/model-output/"
TRAINING_JOB_NAME = f"qwen-vl-grpo-easyr1-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Instance configuration - 4x GPU recommended for GRPO
INSTANCE_TYPE = "ml.g6e.48xlarge"  # 4x A10G GPUs
INSTANCE_COUNT = 1
VOLUME_SIZE_GB = 200

# Get account ID for ECR image
ACCOUNT_ID = boto3.client('sts').get_caller_identity()['Account']

# Training container - custom EasyR1 image
TRAINING_IMAGE = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/aws-sample-grpo-training:latest"

# ==========================================
# Hyperparameters
# ==========================================
# NOTE: Main config is in docker/train_config.yaml (copied from config.yml)
# These are only for S3 data paths - everything else uses the yaml config
hyperparameters = {
    'training_data_s3': TRAINING_DATASET,
    'eval_data_s3': EVAL_DATASET,
}

# Add model_s3 if using S3 model instead of HuggingFace
if MODEL_S3:
    hyperparameters['model_s3'] = MODEL_S3


# ==========================================
# Create Training Job
# ==========================================
def create_training_job():
    sm_client = boto3.client('sagemaker', region_name=REGION)
    
    print("=" * 60)
    print("EasyR1 GRPO Training Configuration")
    print("=" * 60)
    print(f"Job Name: {TRAINING_JOB_NAME}")
    print(f"Model: {MODEL_S3 or 'From config.yml (HuggingFace)'}")
    print(f"Instance: {INSTANCE_TYPE} x {INSTANCE_COUNT}")
    print(f"Training data: {TRAINING_DATASET}")
    print(f"Eval data: {EVAL_DATASET}")
    print(f"Output: {S3_OUTPUT_PATH}")
    print(f"Image: {TRAINING_IMAGE}")
    print("\nHyperparameters:")
    pprint(hyperparameters)
    print("=" * 60)
    
    # Create training job
    response = sm_client.create_training_job(
        TrainingJobName=TRAINING_JOB_NAME,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            'TrainingImage': TRAINING_IMAGE,
            'TrainingInputMode': 'File',
            'ContainerEntrypoint': ['/bin/bash', '/opt/ml/code/entrypoint.sh'],
        },
        HyperParameters=hyperparameters,
        OutputDataConfig={
            'S3OutputPath': S3_OUTPUT_PATH
        },
        ResourceConfig={
            'InstanceType': INSTANCE_TYPE,
            'InstanceCount': INSTANCE_COUNT,
            'VolumeSizeInGB': VOLUME_SIZE_GB,
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 86400  # 24 hours max
        },
        EnableManagedSpotTraining=False,
        EnableInterContainerTrafficEncryption=False,
        EnableNetworkIsolation=False,
    )
    
    print(f"\nTraining job created: {TRAINING_JOB_NAME}")
    print(f"ARN: {response['TrainingJobArn']}")
    print(f"\nMonitor at: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{TRAINING_JOB_NAME}")
    
    return response


def wait_for_training_job():
    """Wait for training job to complete."""
    sm_client = boto3.client('sagemaker', region_name=REGION)
    
    print(f"\nWaiting for training job {TRAINING_JOB_NAME} to complete...")
    
    waiter = sm_client.get_waiter('training_job_completed_or_stopped')
    waiter.wait(
        TrainingJobName=TRAINING_JOB_NAME,
        WaiterConfig={'Delay': 60, 'MaxAttempts': 1440}  # Check every 60s, max 24h
    )
    
    response = sm_client.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)
    status = response['TrainingJobStatus']
    
    print(f"\nTraining job completed with status: {status}")
    
    if status == 'Completed':
        print(f"Model artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
    elif status == 'Failed':
        print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
    
    return response


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true', help='Wait for job completion')
    args = parser.parse_args()
    
    create_training_job()
    
    if args.wait:
        wait_for_training_job()
