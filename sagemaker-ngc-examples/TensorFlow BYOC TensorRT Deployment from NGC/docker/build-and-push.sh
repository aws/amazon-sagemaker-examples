#!/usr/bin/env bash
if [ $# -ne 1 ]; then
    echo "Usage: $0 <algorithm_version>"
    exit 255
fi
set -x
algorithm_name=ngc-tf-bert-sagemaker-demo
algorithm_version=$1

# Configure your account first, based on this:
# https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html
account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${algorithm_version}"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  --tag ${fullname} -f Dockerfile.sagemaker.gpu ./
docker push ${fullname}
