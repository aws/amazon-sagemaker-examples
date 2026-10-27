#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
reponame=$1
versiontag=$2
baseimage=$3
regionname=$4
account=$5

if [ "$reponame" == "" ] || [ "$versiontag" == "" ]  || [ "$baseimage" == "" ] || [ "$regionname" == "" ] || [ "$account" == "" ]
then
    echo "Usage: $0 <repo-name> <version-tag> <base-image> <region> <account>"
    exit 1
fi

if [ $? -ne 0 ]
then
    exit 255
fi

fullname="${account}.dkr.ecr.${regionname}.amazonaws.com/${reponame}:${versiontag}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${reponame}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${reponame}" > /dev/null
fi

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
aws ecr get-login-password --region $regionname | docker login --username AWS --password-stdin ${baseimage} 

aws ecr get-login-password --region $regionname | docker login --username AWS --password-stdin ${account}.dkr.ecr."${regionname}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build  -t ${reponame} . --build-arg BASE_IMAGE=${baseimage}
docker tag ${reponame} ${fullname}

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
aws ecr get-login-password --region $regionname | docker login --username AWS --password-stdin ${account}.dkr.ecr."${regionname}".amazonaws.com

docker push ${fullname}
echo "${fullname}"

