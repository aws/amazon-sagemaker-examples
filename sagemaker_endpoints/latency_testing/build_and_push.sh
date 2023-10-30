#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
model=$1

# parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# cd "$parent_path"


if [ "$model" == "" ]
then
    echo "Usage: $0 <model-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${model}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${model}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${model}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

cd ${model}

chmod +x decision_trees/train
chmod +x decision_trees/serve

docker build  -t ${model} .
docker tag ${model} ${fullname}

docker push ${fullname}
