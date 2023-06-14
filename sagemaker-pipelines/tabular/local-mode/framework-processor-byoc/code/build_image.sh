#!/bin/sh

# The name of our algorithm
repo=$1
registry_name=$2
image_tag=$3
docker_file=$4
platforms=$5

echo "[INFO]: registry_name=${registry_name}"
echo "[INFO]: image_tag=${image_tag}"
echo "[INFO]: docker_file=${docker_file}"
echo "[INFO]: platforms=${platforms}"

cd $repo

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

echo "[INFO]: Region ${region}"

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${registry_name}:${image_tag}"

echo "[INFO]: Image name: ${fullname}"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${registry_name}" > /dev/null 2>&1

aws ecr create-repository --repository-name "${registry_name}" > /dev/null

## If you are extending Amazon SageMaker Images, you need to login to the account
# Get the login command from ECR and execute it directly
password=$(aws ecr --region ${region} get-login-password)

docker login -u AWS -p ${password} "${account}.dkr.ecr.${region}.amazonaws.com"

if [ -z ${platforms} ]
then
  docker build -t ${fullname} -f ${docker_file} .
else
  echo "Provided platform = ${platforms}"
  docker build -t ${fullname} -f ${docker_file} . --platform=${platforms}
fi

docker push ${fullname}