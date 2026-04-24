#!/bin/bash
# Build and push LlamaFactory training image to ECR

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REGION=${AWS_REGION:-us-east-2}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="llamafactory-sft-training"
IMAGE_TAG="latest"

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "=============================================="
echo "Building LlamaFactory SFT Training Image"
echo "=============================================="
echo "ECR URI: ${ECR_URI}"
echo "Build context: ${SCRIPT_DIR}"
echo "=============================================="

# Login to Docker Hub (for hiyouga/llamafactory base image)
echo "Logging in to Docker Hub..."

# Create ECR repository if not exists
echo "Creating ECR repository (if needed)..."
aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION} 2>/dev/null || true

# Login to your ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build from script directory
echo "Building image..."
docker build --platform linux/amd64 -t ${REPO_NAME}:${IMAGE_TAG} "${SCRIPT_DIR}/"

# Tag and push
docker tag ${REPO_NAME}:${IMAGE_TAG} ${ECR_URI}
echo "Pushing to ECR..."
docker push ${ECR_URI}

echo "=============================================="
echo "Done! Image: ${ECR_URI}"
echo "=============================================="
