#!/usr/bin/env bash
# This script shows how to build the Docker image locally

if [ "$#" -eq 3 ]; then
    region=$1
    dlc_account=$2
    base_image=$3
else
    echo "usage: $0 <aws-region> $1 <dlc_account> $3 <base-image>"
    exit 1
fi

# Build the docker image locally
# login ECR for the DLC account
if [[ "$region" == *"cn"* ]]; then
    aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${dlc_account}.dkr.ecr.${region}.amazonaws.com.cn
else
    aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${dlc_account}.dkr.ecr.${region}.amazonaws.com
fi

docker build -t mrcnn_base_image . -f Dockerfile.mrcnn_sm_base --build-arg FROM_IMAGE_NAME=${base_image}
if [ $? -eq 0 ]; then
    echo "base image tag: mrcnn_base_image"
else
    echo "Error: Image build failed"
    exit 1
fi