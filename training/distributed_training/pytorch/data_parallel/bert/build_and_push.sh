#!/usr/bin/env bash
# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.
# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
# set region

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$#" -eq 3 ]; then
    region=$1
    image=$2
    tag=$3
else
    echo "usage: $0 <aws-region> $1 <image-repo> $2 <image-tag>"
    exit 1
fi
# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)
if [ $? -ne 0 ]
then
    exit 255
fi

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region ${region} --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    aws ecr create-repository --region ${region} --repository-name "${image}" > /dev/null
fi
# Build the docker image locally with the image name and then push it to ECR
# with the full name.
# login ECR for the current account

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

docker build ${DIR}/ -t ${image} -f ${DIR}/Dockerfile  --build-arg region=${region}
docker tag ${image} ${fullname}
docker push ${fullname}
if [ $? -eq 0 ]; then
	echo "Amazon ECR URI: ${fullname}"
else
	echo "Error: Image build and push failed"
	exit 1
fi
