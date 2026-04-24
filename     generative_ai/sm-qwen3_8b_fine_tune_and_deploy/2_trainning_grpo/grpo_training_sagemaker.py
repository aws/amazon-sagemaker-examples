"""
SageMaker GRPO Training Job Launcher
Uses EasyR1/veRL framework with custom Docker image.
Refactored to use SageMaker v3 ModelTrainer API.
"""

import boto3
from datetime import datetime
from pprint import pprint
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, SourceCode
from sagemaker.core.shapes.shapes import (
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.core.helper.session_helper import Session

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
BASE_JOB_NAME = "qwen-vl-grpo-easyr1"

# Instance configuration - 4x GPU recommended for GRPO
INSTANCE_TYPE = "ml.g6e.48xlarge"  # 4x L40S GPUs
INSTANCE_COUNT = 1
VOLUME_SIZE_GB = 200

# Create boto3 and SageMaker sessions with explicit region
boto3_session = boto3.Session(region_name=REGION)
sts = boto3_session.client('sts')
ACCOUNT_ID = sts.get_caller_identity()['Account']
sagemaker_session = Session(boto_session=boto3_session)

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
    print("=" * 60)
    print("EasyR1 GRPO Training Configuration (ModelTrainer API)")
    print("=" * 60)
    print(f"Base Job Name: {BASE_JOB_NAME}")
    print(f"Model: {MODEL_S3 or 'From config.yml (HuggingFace)'}")
    print(f"Instance: {INSTANCE_TYPE} x {INSTANCE_COUNT}")
    print(f"Training data: {TRAINING_DATASET}")
    print(f"Eval data: {EVAL_DATASET}")
    print(f"Output: {S3_OUTPUT_PATH}")
    print(f"Image: {TRAINING_IMAGE}")
    print(f"Region: {REGION}")
    print("\nHyperparameters:")
    pprint(hyperparameters)
    print("=" * 60)

    # Create ModelTrainer using SDK 3.x
    # Scripts are baked into image at /opt/ml/code/
    # Override the default "train" command with our entrypoint script
    print("\nCreating ModelTrainer...")
    trainer = ModelTrainer(
        training_image=TRAINING_IMAGE,
        role=ROLE_ARN,
        base_job_name=BASE_JOB_NAME,
        sagemaker_session=sagemaker_session,  # Explicitly set region
        source_code=SourceCode(
            command="/bin/bash /opt/ml/code/entrypoint.sh",
        ),
        compute=Compute(
            instance_type=INSTANCE_TYPE,
            instance_count=INSTANCE_COUNT,
            volume_size_in_gb=VOLUME_SIZE_GB,
        ),
        hyperparameters=hyperparameters,
        output_data_config=OutputDataConfig(
            s3_output_path=S3_OUTPUT_PATH,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=86400,  # 24 hours max
        ),
        training_input_mode="File",
    )

    return trainer


def start_training(trainer, wait=False):
    """Start the training job.

    Args:
        trainer: ModelTrainer instance
        wait: If True, wait for job completion. If False, submit and return.
    """
    print("\nSubmitting training job...")

    # Start training job
    # ModelTrainer automatically generates job name with timestamp
    trainer.train(
        input_data_config=None,  # No input channels needed for GRPO
        wait=wait,  # Wait for completion if requested
        logs=wait,  # Show logs if waiting
    )

    print(f"\n{'='*60}")
    print("Training job submitted successfully!")
    print(f"Monitor at: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs")
    print(f"{'='*60}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true', help='Wait for job completion and stream logs')
    args = parser.parse_args()

    # Create trainer
    trainer = create_training_job()

    # Start training
    start_training(trainer, wait=args.wait)
