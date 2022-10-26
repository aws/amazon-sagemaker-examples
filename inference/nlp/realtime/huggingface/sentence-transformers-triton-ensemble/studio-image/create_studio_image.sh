#!/bin/sh

IMAGE_NAME=$1
REGISTRY_NAME=$2
SM_IMAGE_NAME=$3
ROLE_ARN=$4

if [ -z ${IMAGE_NAME} ]
then
  echo "[INFO]: IMAGE_NAME not passed"
  exit 1
fi

if [ -z ${REGISTRY_NAME} ]
then
  echo "[INFO]: REGISTRY_NAME not passed"
  exit 1
fi

if [ -z ${SM_IMAGE_NAME} ]
then
  echo "[INFO]: SM_IMAGE_NAME not passed"
  exit 1
fi

if [ -z ${ROLE_ARN} ]
then
  echo "[INFO]: ROLE_ARN not passed"
  exit 1
fi

echo "[INFO]: IMAGE_NAME=${IMAGE_NAME}"
echo "[INFO]: REGISTRY_NAME=${REGISTRY_NAME}"
echo "[INFO]: SM_IMAGE_NAME=${SM_IMAGE_NAME}"
echo "[INFO]: ROLE_ARN=${ROLE_ARN}"

aws sagemaker create-image \
    --image-name ${SM_IMAGE_NAME} \
    --role-arn ${ROLE_ARN} \
    || exit 1

account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)

aws sagemaker create-image-version \
    --image-name ${SM_IMAGE_NAME} \
    --base-image "${account}.dkr.ecr.${region}.amazonaws.com/${REGISTRY_NAME}:latest" \
    || exit 1

aws sagemaker delete-app-image-config --app-image-config-name ${SM_IMAGE_NAME}-config

aws sagemaker describe-image-version --image-name ${SM_IMAGE_NAME}

aws sagemaker create-app-image-config --cli-input-json file://${IMAGE_NAME}/app-image-config.json