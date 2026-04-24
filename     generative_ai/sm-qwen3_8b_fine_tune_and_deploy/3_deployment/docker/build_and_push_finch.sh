#!/bin/bash

# Configuration - ⚠️ MODIFY THESE FOR YOUR ENVIRONMENT
AWS_REGION="us-east-2"
AWS_ACCOUNT_ID="YOUR_ACCOUNT_ID"
ECR_REPO_NAME="qwen3-vl-sagemaker-inference"
IMAGE_TAG="latest"

# Full ECR image URI
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

echo "=== Building Custom vLLM Container for Qwen3-VL (Finch) ==="
echo "ECR URI: ${ECR_URI}"

# Step 1: Login to ECR
echo ""
echo "Step 1: Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | finch login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 2: Create ECR repository (if not exists)
echo ""
echo "Step 2: Creating ECR repository (if not exists)..."
aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION} 2>/dev/null || echo "Repository already exists"

# Step 3: Pull base image
echo ""
echo "Step 3: Pulling base image..."
finch pull --platform linux/amd64 vllm/vllm-openai:latest

# Step 4: Build image
echo ""
echo "Step 4: Building image..."
cd "$(dirname "$0")"
finch build --platform linux/amd64 -t ${ECR_REPO_NAME}:${IMAGE_TAG} .

# Step 5: Tag and push to ECR
echo ""
echo "Step 5: Tagging and pushing to ECR..."
finch tag ${ECR_REPO_NAME}:${IMAGE_TAG} ${ECR_URI}
finch push ${ECR_URI}

echo ""
echo "=== Done! ==="
echo "Image available at: ${ECR_URI}"
