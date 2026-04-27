# LLM Fine-tuning & Deployment on AWS SageMaker

A complete pipeline for fine-tuning and deploying Large Language Models (LLM) on AWS SageMaker, including SFT (Supervised Fine-Tuning), GRPO (Group Relative Policy Optimization), and vLLM-based inference.

---

## Table of Contents

1. [Quick Start for First-Time Users](#quick-start-for-first-time-users)
2. [Prerequisites](#prerequisites)
3. [AWS Permissions Setup](#aws-permissions-setup)
4. [Step 1: SFT Training](#step-1-sft-training-supervised-fine-tuning)
5. [Step 2: GRPO Training](#step-2-grpo-training)
6. [Step 3: Model Deployment](#step-3-model-deployment)
7. [Step 4: Model Evaluation](#step-4-model-evaluation)
8. [Project Structure](#project-structure)
9. [Tech Stack](#tech-stack)

---

## Quick Start for First-Time Users

This section provides a high-level overview of the entire pipeline for newcomers.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM Fine-tuning Pipeline                             │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │   Step 1     │      │   Step 2     │      │   Step 3     │      │   Step 4     │
    │  SFT Train   │ ───▶ │  GRPO Train  │ ───▶ │   Deploy     │ ───▶ │  Evaluate    │
    │              │      │              │      │              │      │              │
    │ LlamaFactory │      │   EasyR1     │      │    vLLM      │      │  Inference   │
    └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
          │                     │                     │                     │
          ▼                     ▼                     ▼                     ▼
    Base Model +          SFT Model +           GRPO Model +          Production
    Your Data             Reward Function       SageMaker Endpoint    Ready!
```

### What Each Step Does

| Step | Purpose | Input | Output | Time |
|------|---------|-------|--------|------|
| **Step 1: SFT** | Teach model to follow instructions | Base model + training data | Fine-tuned model | 1-4 hours |
| **Step 2: GRPO** | Improve reasoning with RL | SFT model + reward function | Enhanced model | 4-12 hours |
| **Step 3: Deploy** | Host model for inference | GRPO model | API endpoint | 10-15 min |
| **Step 4: Evaluate** | Test model quality | Endpoint + test data | Metrics & results | Minutes |

### First-Time Setup Checklist

Before running any training, complete these one-time setup tasks:

```
□ 1. Install prerequisites (Python 3.11+, AWS CLI v2, Docker Desktop)
□ 2. Create AWS IAM user with required permissions
□ 3. Create SageMaker Execution Role (copy the ARN!)
□ 4. Configure AWS CLI (aws configure)
□ 5. Request GPU instance quotas (may take 24-48 hours)
□ 6. (Optional) Setup Weights & Biases for monitoring
□ 7. Upload your training data to S3
```

### Step-by-Step Execution Order

For each step, you always follow the same pattern:

```
1. Edit configuration file (set your S3 paths, role ARN, etc.)
2. Build and push Docker image to ECR
3. Run Python script to submit job to SageMaker
4. Monitor progress in AWS Console
```

**Complete execution flow:**

```bash
# ============================================================
# STEP 1: SFT Training
# ============================================================
# 1.1 Edit config
vim 1_trainning_sft/sft_training_fsdp.py  # Set MODEL_ID, TRAINING_DATA_S3, ROLE_ARN

# 1.2 Build & push Docker image
cd 1_trainning_sft/docker
./build_and_push.sh

# 1.3 Submit training job
cd ../..
python 1_trainning_sft/sft_training_fsdp.py

# 1.4 Wait for completion (check SageMaker Console)
# Output: s3://sagemaker-REGION-ACCOUNT_ID/sagemaker-training/JOB_NAME/output/

# ============================================================
# STEP 2: GRPO Training
# ============================================================
# 2.1 Edit config
vim 2_trainning_grpo/grpo_training_sagemaker.py  # Set MODEL_S3 (from Step 1), TRAINING_DATASET, ROLE_ARN

# 2.2 Build & push Docker image
cd 2_trainning_grpo/docker
./build_and_push.sh

# 2.3 Submit training job
cd ../..
python 2_trainning_grpo/grpo_training_sagemaker.py

# 2.4 Wait for completion
# Output: s3://sagemaker-REGION-ACCOUNT_ID/model-output/JOB_NAME/output/

# ============================================================
# STEP 3: Model Deployment
# ============================================================
# 3.1 Edit config
vim 3_deployment/2_model_deploy.py  # Set s3_model_uri (from Step 2), endpoint_name, role

# 3.2 Build & push Docker image
cd 3_deployment/docker
./build_and_push.sh

# 3.3 Deploy model
cd ../..
python 3_deployment/2_model_deploy.py

# 3.4 Wait for endpoint to be InService (check SageMaker Console)

# ============================================================
# STEP 4: Model Evaluation
# ============================================================
# 4.1 Edit config
vim 4_evaluation/4_trigger_endpoint.py  # Set ENDPOINT_NAME

# 4.2 Run evaluation
python 4_evaluation/4_trigger_endpoint.py
```

### Common Issues for First-Time Users

| Issue | Solution |
|-------|----------|
| `ResourceLimitExceeded` | Request GPU quota increase in Service Quotas Console |
| `AccessDenied` on S3 | Check IAM policies, ensure SageMaker role has S3 access |
| `ImageNotFound` | Run `build_and_push.sh` first to create ECR image |
| Docker build fails | Ensure Docker Desktop is running |
| Training job stuck | Check CloudWatch Logs for errors |

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Running scripts |
| AWS CLI | v2 | AWS authentication & ECR operations |
| Docker Desktop | Latest | Building container images |
| Git | Latest | Version control |

### AWS Account Requirements

- An active AWS account with billing enabled
- Access to GPU instances (may require service quota increase)
- Sufficient S3 storage for models and datasets

---

## AWS Permissions Setup

Before starting, you need to set up proper IAM permissions. This section is critical for AWS beginners.

### Step 1: Create IAM User (If Not Exists)

1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Click **Users** → **Create user**
3. Enter a username (e.g., `sagemaker-developer`)
4. Select **Provide user access to the AWS Management Console** (optional)
5. Click **Next**

### Step 2: Required IAM Policies

Your IAM user needs the following permissions. You can either:
- **Option A**: Attach AWS managed policies (easier, broader permissions)
- **Option B**: Create a custom policy (more secure, minimal permissions)

Attach these AWS managed policies to your IAM user:

| Policy Name | Purpose |
|-------------|---------|
| `AmazonSageMakerFullAccess` | SageMaker training & deployment |
| `AmazonS3FullAccess` | S3 bucket operations |
| `AmazonEC2ContainerRegistryFullAccess` | ECR image push/pull |
| `IAMReadOnlyAccess` | Read IAM roles |

### Step 3: Create SageMaker Execution Role

SageMaker needs an IAM role to access AWS resources during training and inference.

1. Go to [IAM Console](https://console.aws.amazon.com/iam/) → **Roles** → **Create role**
2. Select **AWS service** → **SageMaker**
3. Click **Next**
4. Attach these policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
5. Name the role (e.g., `SageMakerExecutionRole`)
6. Click **Create role**
7. **Copy the Role ARN** - you'll need this later (format: `arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole`)

### Step 4: Configure AWS CLI

```bash
# Install AWS CLI (if not installed)
# macOS
brew install awscli

# Or download from: https://aws.amazon.com/cli/

# Configure credentials
aws configure
```

Enter when prompted:
- **AWS Access Key ID**: Your IAM user access key
- **AWS Secret Access Key**: Your IAM user secret key
- **Default region name**: `us-east-2` (recommended - better GPU availability for g6e instances)
- **Default output format**: `json`

### Step 5: Verify Setup

```bash
# Verify AWS CLI is configured
aws sts get-caller-identity

# Expected output:
# {
#     "UserId": "",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/your-username"
# }
```

### Step 6: Request GPU Instance Quota (If Needed)

By default, AWS accounts have 0 quota for GPU instances. You need to request an increase:

1. Go to [Service Quotas Console](https://console.aws.amazon.com/servicequotas/)
2. Search for **Amazon SageMaker**
3. Find and request increase for:
   - `ml.g6e.xlarge for training job usage` (SFT training)
   - `ml.g6e.24xlarge for training job usage` (GRPO training)
   - `ml.g6.2xlarge for endpoint usage` (Inference)
4. Request quota of at least `1` for each
5. Wait for approval (usually 24-48 hours)

### Step 7: Setup Weights & Biases (Optional)

[Weights & Biases](https://wandb.ai/) provides real-time training monitoring and visualization.

1. Create a free account at https://wandb.ai/
2. Get your API key from https://wandb.ai/authorize
3. Create `.env.local` file for training:

```bash
cp 1_trainning_sft/docker/.env.local.example 1_trainning_sft/docker/.env.local
```

4. Edit `1_trainning_sft/docker/.env.local`:

```bash
WANDB_PROJECT=your-project-name
WANDB_API_KEY=your-api-key
WANDB_DISABLED=false  # Set to true to disable
```

---

## Step 1: SFT Training (Supervised Fine-Tuning)

SFT is the first stage of fine-tuning, teaching the model to follow instructions.

### File Structure & Relationships

```
1_trainning_sft/
├── sft_training_fsdp.py          # [Step 2] Submit training job to SageMaker
└── docker/
    ├── Dockerfile                # Container definition (base: LlamaFactory)
    ├── build_and_push.sh         # [Step 1] Build & push container to ECR
    ├── requirements.txt          # Additional Python dependencies
    ├── .env.local                # W&B credentials (create from .env.local.example)
    ├── train_config.yaml         # LlamaFactory training configuration
    ├── train_script.py           # Main training logic (download data, run training, merge weights)
    └── entrypoint.sh             # Container startup script
```

**How it works:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  build_and_push.sh                                                          │
│  ├── Builds Docker image with Dockerfile                                    │
│  ├── Copies train_script.py, train_config.yaml, entrypoint.sh into image   │
│  └── Pushes image to Amazon ECR                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  sft_training_fsdp.py                                                       │
│  ├── References the ECR image                                               │
│  ├── Passes hyperparameters (model path, S3 data, LoRA settings)           │
│  └── Submits SageMaker Training Job                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SageMaker Training Job (runs inside container)                             │
│  ├── entrypoint.sh: Loads .env.local, reads hyperparameters, calls train   │
│  ├── train_script.py: Downloads S3 data, updates config, runs LlamaFactory │
│  └── train_config.yaml: LlamaFactory settings (LoRA, batch size, epochs)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**File purposes:**

| File | Purpose |
|------|---------|
| `build_and_push.sh` | Builds Docker image and pushes to ECR (run this first) |
| `sft_training_fsdp.py` | Submits training job to SageMaker using the ECR image |
| `Dockerfile` | Defines container with LlamaFactory base + custom scripts |
| `entrypoint.sh` | Container startup: loads env vars, parses hyperparameters |
| `train_script.py` | Downloads training data from S3, runs LlamaFactory, merges LoRA weights |
| `train_config.yaml` | LlamaFactory config: LoRA rank, learning rate, batch size, etc. |
| `.env.local` | Weights & Biases credentials for training monitoring |

### Customizing Training Parameters

To modify training behavior, edit `1_trainning_sft/docker/train_config.yaml`:

```yaml
### LoRA Settings
lora_rank: 8              # Higher = more parameters, better quality, slower
lora_alpha: 16            # Scaling factor (typically 2x lora_rank)
lora_target: all          # Which layers to apply LoRA

### Training Settings
per_device_train_batch_size: 1    # Batch size per GPU
gradient_accumulation_steps: 10   # Effective batch = batch_size × accumulation
learning_rate: 5.0e-5             # Learning rate
num_train_epochs: 2               # Number of training epochs
cutoff_len: 2048                  # Max sequence length

### Evaluation
val_size: 0.05            # 5% of data for validation
eval_steps: 100           # Evaluate every N steps
```

> ⚠️ After modifying `train_config.yaml`, you must rebuild and push the Docker image:
> ```bash
> cd 1_trainning_sft/docker
> ./build_and_push.sh
> ```

### 1.1 Prepare Training Data

SFT training uses **ShareGPT format** (conversation-style JSON). Upload your data to S3.

**Data Format (train.json):**
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Your prompt here"},
      {"from": "gpt", "value": "Expected response here"}
    ],
    "images": ["image1.jpg", "image2.jpg"]  // Optional, for vision models
  }
]
```

**Dataset Info (dataset_info.json):**
```json
{
  "train": {
    "file_name": "train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
}
```

Upload to S3:
```bash
aws s3 cp train.json s3://your-bucket/sft-data/train.json
aws s3 cp dataset_info.json s3://your-bucket/sft-data/dataset_info.json
# If using images:
aws s3 cp --recursive images/ s3://your-bucket/sft-data/images/
```

### 1.2 Configure Training Parameters

Edit `1_trainning_sft/sft_training_fsdp.py`:

```python
# ==========================================
# ⚠️ MODIFY THESE VALUES
# ==========================================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"  # Or your S3 model path
TRAINING_DATA_S3 = "s3://your-bucket/your-training-data.json"
ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
INSTANCE_TYPE = "ml.g6e.xlarge"  # 1x L40S GPU (48GB)
```

### 1.3 Build and Push Training Container

```bash
cd 1_trainning_sft/docker
./build_and_push.sh
```

> ⚠️ **Before running `build_and_push.sh`**, double-check these values in the script:
> - `AWS_REGION` - Must match your deployment region (e.g., `us-east-2`)
> - `AWS_ACCOUNT_ID` - Your 12-digit AWS account ID
> - `ECR_URI` - The full ECR repository URI (auto-generated from above values)
>
> Open the script and verify:
> ```bash
> AWS_REGION="us-east-2"           # ← Verify this matches your region
> AWS_ACCOUNT_ID="YOUR_ACCOUNT_ID"    # ← Replace with YOUR account ID
> ```

This script will:
1. Create ECR repository (if not exists)
2. Build Docker image with LlamaFactory
3. Push to Amazon ECR

Expected output:
```
Done! Image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/llamafactory-sft-training:latest
```

### 1.4 Submit Training Job

> ⚠️ **Before running**, double-check these values in `sft_training_fsdp.py`:
> - `REGION` - Must match your deployment region
> - `ROLE_ARN` - Your SageMaker execution role ARN
> - `ECR_IMAGE` - The ECR image URI (must match region and account ID)

```bash
cd ../..  # Back to project root
python 1_trainning_sft/sft_training_fsdp.py
```

Expected output:
```
============================================================
Qwen3-VL-8B SFT Training with QLoRA
============================================================
Model: Qwen/Qwen3-VL-8B-Instruct
Training Data: s3://your-bucket/data.json
Instance: ml.g6e.xlarge
Job Name: qwen3-vl-sft-20260122-143052
============================================================

Training job submitted: qwen3-vl-sft-20260122-143052
Monitor at: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/qwen3-vl-sft-20260122-143052
```

### 1.5 Monitor Training Progress

1. Click the console link from the output
2. Or go to [SageMaker Console](https://console.aws.amazon.com/sagemaker/) → **Training** → **Training jobs**
3. View logs in **CloudWatch Logs**

Training typically takes 1-4 hours depending on dataset size.

### 1.6 Locate Output Model

After training completes, find your model at:
```
s3://sagemaker-REGION-ACCOUNT_ID/sagemaker-training/JOB_NAME/output/model.tar.gz
```

---

## Step 2: GRPO Training

GRPO (Group Relative Policy Optimization) is an advanced training technique for improving model reasoning using reinforcement learning with custom reward functions.

### File Structure & Relationships

```
2_trainning_grpo/
├── grpo_training_sagemaker.py    # [Step 2] Submit training job to SageMaker
├── config.yml                    # GRPO configuration (reference for train_config.yaml)
└── docker/
    ├── Dockerfile                # Container definition (base: NVIDIA PyTorch + EasyR1)
    ├── build_and_push.sh         # [Step 1] Build & push container to ECR
    ├── requirements.txt          # Additional Python dependencies
    ├── .env.local                # W&B credentials (create from .env.local.example)
    ├── train_config.yaml         # EasyR1/veRL training configuration
    ├── train_script.py           # Main training logic
    ├── entrypoint.sh             # Container startup script
    └── reward_function/          # Custom reward functions
        ├── __init__.py
        └── math.py               # Tagging reward function with multi-metric scoring
```

**How it works:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  build_and_push.sh                                                          │
│  ├── Builds Docker image with Dockerfile (NVIDIA PyTorch + EasyR1)         │
│  ├── Copies train_script.py, train_config.yaml, entrypoint.sh into image   │
│  ├── Copies reward_function/ directory for custom reward computation       │
│  └── Pushes image to Amazon ECR                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  grpo_training_sagemaker.py                                                 │
│  ├── References the ECR image                                               │
│  ├── Passes hyperparameters (model S3 path, training/eval data S3 paths)   │
│  └── Submits SageMaker Training Job                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SageMaker Training Job (runs inside container)                             │
│  ├── entrypoint.sh: Loads .env.local, reads hyperparameters, calls train   │
│  ├── train_script.py: Downloads S3 data, updates config, runs EasyR1       │
│  ├── train_config.yaml: EasyR1/veRL settings (GRPO, rollout, FSDP)         │
│  └── reward_function/math.py: Custom reward scoring (recall, precision)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**File purposes:**

| File | Purpose |
|------|---------|
| `build_and_push.sh` | Builds Docker image and pushes to ECR (run this first) |
| `grpo_training_sagemaker.py` | Submits training job to SageMaker using the ECR image |
| `config.yml` | Reference GRPO configuration (copied to docker/train_config.yaml) |
| `Dockerfile` | Defines container with NVIDIA PyTorch base + EasyR1 + NCCL fixes |
| `entrypoint.sh` | Container startup: loads env vars, parses hyperparameters |
| `train_script.py` | Downloads training data from S3, runs EasyR1 GRPO training |
| `train_config.yaml` | EasyR1/veRL config: GRPO algorithm, rollout, FSDP settings |
| `reward_function/math.py` | Custom reward function with multi-metric scoring |
| `.env.local` | Weights & Biases credentials for training monitoring |

### Customizing GRPO Training Parameters

To modify training behavior, edit `2_trainning_grpo/config.yml` (then copy to `docker/train_config.yaml`):

```yaml
### Data Settings
data:
  max_prompt_length: 1800         # Max input prompt length
  max_response_length: 300        # Max generated response length
  rollout_batch_size: 8           # Batch size for rollout generation
  val_batch_size: 16              # Validation batch size

### Algorithm Settings
algorithm:
  adv_estimator: grpo             # Use GRPO algorithm
  kl_coef: 1.0e-2                 # KL divergence coefficient
  use_kl_loss: true               # Enable KL loss

### Actor (Policy Model) Settings
worker:
  actor:
    global_batch_size: 4          # Mini-batch size for PPO updates
    micro_batch_size_per_device_for_update: 1
    model:
      # NOTE: This is the DEFAULT model path (HuggingFace ID)
      # If MODEL_S3 is set in grpo_training_sagemaker.py, it will OVERRIDE this value
      model_path: Qwen/Qwen3-VL-8B-Instruct
      enable_gradient_checkpointing: true
    optim:
      lr: 1.0e-6                  # Learning rate
      weight_decay: 1.0e-2
    fsdp:
      enable_full_shard: true     # Enable FSDP for multi-GPU
      enable_cpu_offload: false
    offload:
      offload_params: true        # Offload to CPU to save GPU memory
      offload_optimizer: true

### Rollout (Generation) Settings
  rollout:
    n: 5                          # Number of samples per prompt
    temperature: 1.0              # Sampling temperature
    tensor_parallel_size: 4       # Use all 4 GPUs for generation
    gpu_memory_utilization: 0.8   # vLLM GPU memory usage

### Reward Function
  reward:
    reward_function: /opt/ml/code/reward_function/math.py:compute_score

### Trainer Settings
trainer:
  total_epochs: 1                 # Number of training epochs
  n_gpus_per_node: 4              # Use all 4 GPUs
  val_freq: 30                    # Validate every N steps
  logger: ["file", "wandb"]       # Enable W&B logging
```

**Important:** The `model_path` in `config.yml` is only used when `MODEL_S3 = None` in `grpo_training_sagemaker.py`. When `MODEL_S3` is set, the training script downloads the model from S3 and overrides this value at runtime.

> ⚠️ After modifying `config.yml`, rebuild the Docker image (the build script automatically copies `config.yml` to `docker/train_config.yaml`):
> ```bash
> cd 2_trainning_grpo/docker
> ./build_and_push.sh
> ```

### Custom Reward Function

The reward function in `reward_function/math.py` computes a multi-metric score for tagging tasks:

| Metric | Weight | Description |
|--------|--------|-------------|
| `recall` | 35% | Proportion of ground truth tags found |
| `accuracy` | 35% | Classification accuracy of matched tags |
| `precision` | 20% | Penalizes extra predicted tags |
| `match_quality` | 5% | Fuzzy matching quality score |
| `formatting` | 5% | Correct output format (9 categories) |

To customize the reward function, edit `2_trainning_grpo/docker/reward_function/math.py`:

```python
def compute_score(
    reward_inputs: list[dict[str, Any]],
    recall_weight: float = 0.35,
    precision_weight: float = 0.2,
    accuracy_weight: float = 0.35,
    match_quality_weight: float = 0.05,
    formatting_weight: float = 0.05
) -> list[dict[str, float]]:
    """
    Compute reward scores for GRPO training.
    
    Args:
        reward_inputs: [{"response": "...", "ground_truth": "..."}, ...]
    
    Returns:
        [{"overall": 0.8, "recall": 0.9, "precision": 0.7, ...}, ...]
    """
```

### Docker Image Details

The GRPO Docker image is built from `nvcr.io/nvidia/pytorch:25.05-py3` with:

- **EasyR1/veRL**: GRPO training framework from HiyoGa
- **vLLM 0.11.0**: Fast inference for rollout generation
- **PyTorch 2.8.0**: Latest PyTorch with CUDA support
- **Flash Attention 2.8.3**: Efficient attention computation
- **NCCL Fixes**: Pre-configured for GPUs without NVLink (g6e L40S, g5 A10G)

Key NCCL environment variables (already set in Dockerfile):
```bash
NCCL_P2P_DISABLE=1      # Disable P2P (no NVLink)
NCCL_SHM_DISABLE=1      # Disable shared memory
NCCL_NET=Socket         # Use socket communication
```

### 2.1 Prepare Training Data

GRPO training uses **Parquet format** with prompt-answer pairs. Upload your data to S3.

**Data Format (train.parquet):**

| Column | Type | Description |
|--------|------|-------------|
| `problem` | string | The full prompt including system instructions and user input |
| `answer` | string | The expected ground truth response (used by reward function) |

**Example row:**
| Column | Content |
|--------|---------|
| `problem` | You are a professional and rigorous product tagging expert, responsible for automatically generating and classifying tags based on the provided product information...<br><br>Product Name: MI Amazon Usb Type-C Cable Smartphone Charging (Black) \|Connectivity: Usb 2.0 (Sync And Charging)\| Universal For All Type-C Devices (Grey)... |
| `answer` | 1, Product Name: USB Type-C Cable; Charging Cable<br>2, Brand / Product Line: MI<br>3, Function & Usage: Fast Charging; Data Sync<br>4, Ingredients & Materials: TPE + Nylon<br>5, Specifications / Model: 1m; 3A; 480Mbps<br>6, Color: Black; Grey<br>7, Style & Features: Durable<br>8, Use Occasion: Smartphones; Tablets; Laptops<br>9, Corresponding Holiday: |

**Create parquet from pandas:**
```python
import pandas as pd

data = [
    {"problem": "Your prompt here...", "answer": "Expected response..."},
    {"problem": "Another prompt...", "answer": "Another response..."},
]
df = pd.DataFrame(data)
df.to_parquet("train.parquet", index=False)
df.to_parquet("eval.parquet", index=False)  # Can be same or subset
```

Upload to S3:
```bash
aws s3 cp train.parquet s3://your-bucket/grpo-data/train.parquet
aws s3 cp eval.parquet s3://your-bucket/grpo-data/eval.parquet
```

### 2.2 Configure GRPO Parameters

Edit `2_trainning_grpo/grpo_training_sagemaker.py`:

```python
# ==========================================
# ⚠️ MODIFY THESE VALUES
# ==========================================
REGION = 'us-east-2'  # Region with g6e availability
ROLE_ARN = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'

# Model configuration - TWO OPTIONS:
# Option 1: Use S3 model (from SFT training output) - RECOMMENDED
MODEL_S3 = "s3://your-bucket/model-from-sft/"  # This OVERRIDES config.yml model_path

# Option 2: Use HuggingFace model (set MODEL_S3 = None)
# MODEL_S3 = None  # Will use model_path from config.yml (e.g., Qwen/Qwen3-VL-8B-Instruct)

TRAINING_DATASET = "s3://your-bucket/train.parquet"
EVAL_DATASET = "s3://your-bucket/eval.parquet"
S3_BUCKET = "sagemaker-us-east-2-YOUR_ACCOUNT_ID"
INSTANCE_TYPE = "ml.g6e.48xlarge"  # 4x L40S GPUs (recommended)
```

**Model Path Priority:**
1. If `MODEL_S3` is set → Downloads from S3 and overrides `config.yml`
2. If `MODEL_S3 = None` → Uses `worker.actor.model.model_path` from `config.yml`

This allows you to either:
- Use your SFT-trained model from S3 (typical workflow)
- Use a HuggingFace model directly (for testing or starting fresh)

### 2.3 Build GRPO Training Container

```bash
cd 2_trainning_grpo/docker
./build_and_push.sh
```

> ⚠️ **Before running `build_and_push.sh`**, double-check these values in the script:
> - `AWS_REGION` - Must match your deployment region (e.g., `us-east-2`)
> - `AWS_ACCOUNT_ID` - Your 12-digit AWS account ID
> - `ECR_URI` - The full ECR repository URI (auto-generated from above values)
>
> Open the script and verify:
> ```bash
> AWS_REGION="us-east-2"           # ← Verify this matches your region
> AWS_ACCOUNT_ID="YOUR_ACCOUNT_ID"    # ← Replace with YOUR account ID
> ```

This script will:
1. Create ECR repository (if not exists)
2. Build Docker image with EasyR1 + NCCL fixes
3. Push to Amazon ECR

Expected output:
```
Done! Image: 123456789012.dkr.ecr.us-east-2.amazonaws.com/easyr1-grpo-training:latest
```

### 2.4 Submit GRPO Training Job

> ⚠️ **Before running**, double-check these values in `grpo_training_sagemaker.py`:
> - `REGION` - Must match your deployment region
> - `ROLE_ARN` - Your SageMaker execution role ARN
> - `ECR_IMAGE` - The ECR image URI (must match region and account ID)

```bash
cd ../..
python 2_trainning_grpo/grpo_training_sagemaker.py
```

Expected output:
```
============================================================
EasyR1 GRPO Training Configuration
============================================================
Job Name: qwen-vl-grpo-easyr1-20260126-143052
Model: s3://your-bucket/model-from-sft/
Instance: ml.g6e.48xlarge x 1
Training data: s3://your-bucket/train.parquet
Eval data: s3://your-bucket/eval.parquet
Output: s3://sagemaker-us-east-2-YOUR_ACCOUNT_ID/model-output/
============================================================

Training job created: qwen-vl-grpo-easyr1-20260126-143052
Monitor at: https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/qwen-vl-grpo-easyr1-20260126-143052
```

### 2.5 Monitor Training Progress

1. Click the console link from the output
2. Or go to [SageMaker Console](https://console.aws.amazon.com/sagemaker/) → **Training** → **Training jobs**
3. View logs in **CloudWatch Logs**
4. If W&B is enabled, monitor at https://wandb.ai/

GRPO training typically takes 4-12 hours depending on dataset size and number of epochs.

---

## Step 3: Model Deployment

Deploy your fine-tuned model to a SageMaker async endpoint with vLLM inference engine.

### File Structure & Relationships

```
3_deployment/
├── 2_model_deploy.py             # [Step 2] Deploy model to SageMaker endpoint
├── 3_autoscale_setting.py        # [Step 3] Configure auto-scaling (scale to zero)
└── docker/
    ├── Dockerfile                # Container definition (base: vLLM)
    ├── build_and_push.sh         # [Step 1] Build & push container to ECR
    └── sagemaker-entrypoint.sh   # vLLM startup script with SM_VLLM_ env parsing
```

**How it works:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  build_and_push.sh                                                          │
│  ├── Pulls vllm/vllm-openai:latest base image                              │
│  ├── Copies sagemaker-entrypoint.sh into image                             │
│  └── Pushes image to Amazon ECR                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2_model_deploy.py                                                          │
│  ├── References the ECR image                                               │
│  ├── Points to trained model in S3 (from SFT training output)              │
│  ├── Sets vLLM environment variables (SM_VLLM_*)                           │
│  └── Creates SageMaker Model → EndpointConfig → Endpoint                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SageMaker Endpoint (runs inside container)                                 │
│  ├── sagemaker-entrypoint.sh: Parses SM_VLLM_* env vars → vLLM CLI args    │
│  └── Starts vLLM OpenAI-compatible API server on port 8080                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3_autoscale_setting.py (Optional)                                          │
│  └── Configures auto-scaling to scale down to 0 when idle                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**File purposes:**

| File | Purpose |
|------|---------|
| `build_and_push.sh` | Builds vLLM Docker image and pushes to ECR (run this first) |
| `2_model_deploy.py` | Creates SageMaker Model, EndpointConfig, and Endpoint |
| `3_autoscale_setting.py` | Enables scale-to-zero to save costs when idle |
| `Dockerfile` | Defines container with vLLM base + custom entrypoint |
| `sagemaker-entrypoint.sh` | Converts SM_VLLM_* environment variables to vLLM CLI arguments |

### Customizing vLLM Settings

vLLM settings are configured via environment variables in `2_model_deploy.py`:

```python
"Environment": {
    "SM_VLLM_MODEL": "/opt/ml/model",           # Model path (don't change)
    "SM_VLLM_MAX_MODEL_LEN": "2048",            # Max sequence length
    "SM_VLLM_DTYPE": "bfloat16",                # Data type (bfloat16/float16)
    "SM_VLLM_TRUST_REMOTE_CODE": "true",        # Required for Qwen models
    "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",    # GPU memory usage (0.0-1.0)
    "SM_VLLM_ENFORCE_EAGER": "true",            # Disable CUDA graphs
}
```

The `sagemaker-entrypoint.sh` automatically converts these to vLLM CLI arguments:
- `SM_VLLM_MAX_MODEL_LEN=2048` → `--max-model-len 2048`
- `SM_VLLM_TRUST_REMOTE_CODE=true` → `--trust-remote-code`

### 3.1 Configure Deployment Parameters

Edit `3_deployment/2_model_deploy.py`:

```python
# ==========================================
# ⚠️ MODIFY THESE VALUES
# ==========================================
s3_model_uri = "s3://your-bucket/path/to/model/"  # Your trained model
endpoint_name = "your-model-endpoint"
role = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'

# Container settings
AWS_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
```

### 3.2 Build Inference Container

```bash
cd 3_deployment/docker
./build_and_push.sh
```

> ⚠️ **Before running `build_and_push.sh`**, double-check these values in the script:
> - `AWS_REGION` - Must match your deployment region (e.g., `us-east-2`)
> - `AWS_ACCOUNT_ID` - Your 12-digit AWS account ID
> - `ECR_URI` - The full ECR repository URI (auto-generated from above values)
>
> Open the script and verify:
> ```bash
> AWS_REGION="us-east-2"           # ← Verify this matches your region
> AWS_ACCOUNT_ID="YOUR_ACCOUNT_ID"    # ← Replace with YOUR account ID
> ```

### 3.3 Deploy Model

> ⚠️ **Before running**, double-check these values in `2_model_deploy.py`:
> - `AWS_REGION` - Must match your deployment region
> - `AWS_ACCOUNT_ID` - Your 12-digit AWS account ID
> - `role` - Your SageMaker execution role ARN
> - `ECR_IMAGE` - The ECR image URI (must match region and account ID)

```bash
cd ../..
python 3_deployment/2_model_deploy.py
```

Expected output:
```
Created model: your-model-endpoint
Created endpoint config: your-model-endpoint
Creating endpoint... This may take 10-15 minutes...
```

### 3.4 Monitor Deployment

1. Go to [SageMaker Console](https://console.aws.amazon.com/sagemaker/)
2. Navigate to **Inference** → **Endpoints**
3. Wait for status to change from `Creating` to `InService`

### 3.5 Configure Auto-Scaling (Optional)

Enable scale-to-zero to save costs when endpoint is idle:

```bash
# Edit endpoint_name in the script first
python 3_deployment/3_autoscale_setting.py
```

This configures:
- **Min capacity**: 0 (allows shutdown)
- **Max capacity**: 2
- **Scale-in cooldown**: 10 minutes of idle time before shutdown

---

## Step 4: Model Evaluation

Test your deployed model with text and image inputs.

### 4.1 Configure Endpoint

Edit `4_evaluation/4_trigger_endpoint.py`:

```python
# ==========================================
# ⚠️ MODIFY THESE VALUES
# ==========================================
ENDPOINT_NAME = "your-model-endpoint"
INPUT_BUCKET = "sagemaker-us-east-1-YOUR_ACCOUNT_ID"
REGION = "us-east-1"
```

### 4.2 Run Evaluation

```bash
python 4_evaluation/4_trigger_endpoint.py
```

### 4.3 Custom Inference

```python
from 4_evaluation.4_trigger_endpoint import invoke_sagemaker_async, load_image

# Text-only inference
result, elapsed = invoke_sagemaker_async("Describe this product in detail")
print(f"Result ({elapsed:.2f}s): {result}")

# Image + Text inference
image = load_image("product.jpg")  # Local file
# Or from S3:
# image = load_image("s3://your-bucket/images/product.jpg")

result, elapsed = invoke_sagemaker_async(
    "What is shown in this image?",
    images=[image]
)
print(f"Result ({elapsed:.2f}s): {result}")
```

---

## Project Structure

```
├── 1_trainning_sft/              # SFT Training
│   ├── sft_training_fsdp.py          # Submit training job
│   └── docker/
│       ├── Dockerfile                # LlamaFactory container
│       ├── build_and_push.sh         # Build & push to ECR
│       ├── train_script.py           # Training logic
│       ├── train_config.yaml         # LlamaFactory config
│       └── .env.local                # W&B credentials (create from .env.local.example)
│
├── 2_trainning_grpo/             # GRPO Training
│   ├── grpo_training_sagemaker.py    # Submit GRPO job
│   ├── config.yml                    # GRPO configuration (reference)
│   └── docker/
│       ├── Dockerfile                # NVIDIA PyTorch + EasyR1 container
│       ├── build_and_push.sh         # Build & push to ECR
│       ├── train_script.py           # Training logic
│       ├── train_config.yaml         # EasyR1/veRL config
│       ├── entrypoint.sh             # Container startup
│       ├── .env.local                # W&B credentials
│       └── reward_function/          # Custom reward functions
│           └── math.py               # Multi-metric tagging reward
│
├── 3_deployment/                 # Model Deployment
│   ├── 2_model_deploy.py             # Deploy to endpoint
│   ├── 3_autoscale_setting.py        # Auto-scaling config
│   └── docker/
│       ├── Dockerfile                # vLLM inference container
│       └── build_and_push.sh         # Build & push to ECR
│
├── 4_evaluation/                 # Model Evaluation
│   └── 4_trigger_endpoint.py         # Test endpoint
│
├── data_parquet 2/               # Sample data
│   ├── train.parquet
│   └── eval.parquet
│
└── requirements.txt              # Python dependencies
```

---

## Tech Stack

- **Training Framework**: LlamaFactory (SFT), EasyR1/veRL (GRPO)
- **Inference Engine**: vLLM with OpenAI-compatible API
- **Model**: Qwen3-VL-8B (Vision-Language Model)
- **Cloud Services**: AWS SageMaker, ECR, S3, CloudWatch
- **Monitoring**: Weights & Biases (optional)
- **Deep Learning**: PyTorch 2.8.0, Flash Attention 2.8.3, FSDP
- **GRPO Components**: Custom reward functions, multi-GPU rollout with vLLM
