#!/bin/bash

# The name of our algorithm
algorithm_name=${1:-sagemaker-autogluon-training}

# Get account
account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration
region=$(aws configure get region)
region=${region:-us-east-2}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)
$(aws ecr get-login --registry-ids 763104351884 --region ${region} --no-include-email)

# Build the docker image, tag with full name and then push it to ECR
docker build  -t ${algorithm_name} -f container-training/Dockerfile.training .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}