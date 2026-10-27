#!/bin/bash

# Get account
account=${1}

# Get the region defined in the current configuration
region=${2:-us-east-1}

# The name of our algorithm
algorithm_name=${3:-sagemaker-autogluon-inference}

uri_prefix=${4}
fullname="${uri_prefix}/${algorithm_name}:latest"

# Get the registry id
registry_id=${5}
registry_uri=${6}

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region ${region} --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --region ${region} --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)
$(aws ecr get-login --registry-ids ${registry_id} --region ${region} --no-include-email)
    
# Build the docker image, tag with full name and then push it to ECR
docker build -t ${algorithm_name} -f container-inference/Dockerfile.inference . --build-arg REGISTRY_URI=${registry_uri}
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
