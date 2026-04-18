"""
Fine-tune Qwen3-VL-8B with QLoRA on SageMaker.
Uses SageMaker SDK 3.x ModelTrainer API.
Scripts are baked into the Docker image - no S3 upload needed.

Supports:
- HuggingFace model ID: "Qwen/Qwen3-VL-8B-Instruct"
- S3 path for continued training: "s3://bucket/path/to/model/"
"""

from datetime import datetime

import boto3
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, SourceCode
from sagemaker.core.shapes.shapes import (
    S3DataSource,
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.core.helper.session_helper import Session

# ==========================================
# Configuration
# ==========================================
# Model source - supports both HuggingFace ID and S3 path
# Examples:
#   MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"           # HuggingFace model
# MODEL_ID = "s3://your-bucket/model-unzip/qwen-grpo/"  # S3 model (continued training)
MODEL_ID = "Qwen/Qwen3-8B"
TRAINING_DATA_S3 = "s3://your-bucket/train_data_public.json"
OUTPUT_BUCKET='your-bucket'

def is_s3_path(path: str) -> bool:
    """Check if the path is an S3 URI."""
    return path.startswith("s3://")

REGION = 'us-east-2'
boto3_session = boto3.Session(region_name=REGION)
sts = boto3_session.client('sts')
ACCOUNT_ID = sts.get_caller_identity()['Account']
BUCKET_NAME = f"sagemaker-{REGION}-{ACCOUNT_ID}"
ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'

# Create SageMaker session with explicit region
sagemaker_session = Session(boto_session=boto3_session)

# Instance configuration
INSTANCE_TYPE = "ml.g6e.2xlarge"  # 1x L40S GPU (48GB VRAM)
INSTANCE_COUNT = 1

# Training image - Use custom pre-built image with all dependencies and scripts
TRAINING_IMAGE = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/aws-sample-sft-training:latest"

# Job name
BASE_JOB_NAME = "aws-sample-qwen3-sft"


def main():
    print("=" * 60)
    print("Qwen3SFT Training with QLoRA (ModelTrainer API)")
    print("=" * 60)

    # Determine model source type
    use_s3_model = is_s3_path(MODEL_ID)
    model_source = "S3 (continued training)" if use_s3_model else "HuggingFace"

    print(f"Model: {MODEL_ID}")
    print(f"Model Source: {model_source}")
    print(f"Training Data: {TRAINING_DATA_S3}")
    print(f"Instance: {INSTANCE_TYPE}")
    print(f"Image: {TRAINING_IMAGE}")
    print(f"Base Job Name: {BASE_JOB_NAME}")
    print("=" * 60)
    
    # Hyperparameters
    # If S3 model, use /opt/ml/input/data/model as path (SageMaker mounts it there)
    model_path = "/opt/ml/input/data/model" if use_s3_model else MODEL_ID
    
    hyperparameters = {
        'model_name_or_path': model_path,
        'training_data_s3': TRAINING_DATA_S3,
        'num_train_epochs': '2',
        'lora_r': '8',
        'lora_alpha': '16',
        'model_source': 's3' if use_s3_model else 'huggingface',
        'merge_weights': '1',  # '1' = merge LoRA into base model, '0' = adapter only
    }

    output_s3 = f"s3://{OUTPUT_BUCKET}/sagemaker-training/{BASE_JOB_NAME}/output"
    print(f"\nOutput: {output_s3}")
    print("\nStarting training job...")

    # Build input data config using ModelTrainer's InputData
    input_data_config = None
    if use_s3_model:
        # Add S3 model as input channel
        # Note: InputData expects S3DataSource directly, not wrapped in DataSource
        input_data_config = [
            InputData(
                channel_name="model",
                data_source=S3DataSource(
                    s3_uri=MODEL_ID,
                    s3_data_type="S3Prefix",
                    s3_data_distribution_type="FullyReplicated",
                ),
            )
        ]
        print(f"S3 Model will be mounted at: /opt/ml/input/data/model")
    
    # Create ModelTrainer using SDK 3.x
    # Scripts are baked into image at /opt/ml/code/
    # Override the default "train" command with our entrypoint script
    trainer = ModelTrainer(
        training_image=TRAINING_IMAGE,
        role=ROLE_ARN,
        base_job_name=BASE_JOB_NAME,
        sagemaker_session=sagemaker_session,  # Explicitly set region
        source_code=SourceCode(
            command="bash /opt/ml/code/entrypoint.sh",
        ),
        compute=Compute(
            instance_type=INSTANCE_TYPE,
            instance_count=INSTANCE_COUNT,
            volume_size_in_gb=100,
        ),
        hyperparameters=hyperparameters,
        output_data_config=OutputDataConfig(
            s3_output_path=output_s3,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=86400,
        ),
        training_input_mode="File",
    )

    # Start training job
    # ModelTrainer automatically generates job name with timestamp
    trainer.train(
        input_data_config=input_data_config,
        wait=False,  # Set to True if you want to wait for completion
        logs=True,   # Set to False to disable log streaming
    )

    print(f"\nTraining job submitted!")
    print(f"Monitor at: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs")


if __name__ == "__main__":
    main()
