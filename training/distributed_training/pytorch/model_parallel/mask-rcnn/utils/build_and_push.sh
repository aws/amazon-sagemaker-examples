#!/usr/bin/env bash
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.
# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
# set region
if [ "$#" -eq 4 ]; then
    region=$1
    image=$2
    tag=$3
    base_image=$4
else
    echo "usage: $0 <aws-region> $1 <image-repo> $2 <image-tag> $3 <base-image>"
    exit 1
fi
# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)
if [ $? -ne 0 ]
then
    exit 255
fi
if [[ "$region" == *"cn"* ]]; then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com.cn/${image}:${tag}"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"
fi
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region ${region} --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    aws ecr create-repository --region ${region} --repository-name "${image}" > /dev/null
fi
# Build the docker image locally with the image name and then push it to ECR
# with the full name.
# login ECR for the current account
if [[ "$region" == *"cn"* ]]; then
    aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com.cn
else
    aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
fi
# login ECR for the rubik image account
if [[ "$region" == *"cn"* ]]; then
    aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin 039281049316.dkr.ecr.${region}.amazonaws.com.cn
else
    aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin 578276202366.dkr.ecr.${region}.amazonaws.com
fi
docker build -t ${image} . --build-arg BASE_IMAGE=${base_image}
docker tag ${image} ${fullname}
docker push ${fullname}
if [ $? -eq 0 ]; then
	echo "Amazon ECR URI: ${fullname}"
else
	echo "Error: Image build and push failed"
	exit 1
fi