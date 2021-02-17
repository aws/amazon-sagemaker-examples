#!/bin/bash

# The name of our algorithm
algorithm_name=sagemaker-rmars

set -e # stop if anything fails

account=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID $account"

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}
echo "AWS Region $region"

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

echo "Target image URI $fullname"

# If the repository doesn't exist in ECR, create it.

echo "Checking for existing repository..."
set +e
aws ecr describe-repositories --repository-names "${algorithm_name}"
if [ $? -ne 0 ]
then
    set -e
    echo "Creating repository"
    aws ecr create-repository --repository-name "${algorithm_name}"
else
    set -e
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}

docker push ${fullname}
