#!/bin/bash
# COLLABORATORS: Jeremy Feltracco(jpfelt@), Vidya Sagar Ravipati (ravividy@)

# Immediately stop if any intermediate step fails.
set -e

REPO_ROOT_PATH="$(dirname $0)/.."

PREFIX="smgt-workflow"

echo "[START] LAUNCHING SMGT-WORKFLOW"

if [[ -z "${AWS_PROFILE}" ]];
then
  read -p "Enter your AWS CLI IAM AWS_PROFILE : " AWS_PROFILE
fi

if [[ -z "${COGNITO_POOL_ID}" ]];
then
  read -p "Enter your SageMaker Workforce cognito user pool id: " COGNITO_POOL_ID
fi

if [[ -z "${COGNITO_POOL_CLIENT_ID}" ]];
then
  read -p "Enter your SageMaker Workforce cognito client id: " COGNITO_POOL_CLIENT_ID
fi

stack_name="${PREFIX}-v0"
account=$(aws sts get-caller-identity --query Account --output text --profile ${AWS_PROFILE})
region=$(aws configure get region --profile ${AWS_PROFILE})
echo "We will be launching resources in ${region} in account ${account} with AWS CLI Profile ${AWS_PROFILE}."

DEPLOY_BUCKET_NAME="${PREFIX}-${account}-${region}"
aws s3 mb s3://${DEPLOY_BUCKET_NAME} --profile ${AWS_PROFILE}

 sam build \
     -t main-merged.yml \
     --use-container \
     --skip-pull-image

 sam package \
     --template-file .aws-sam/build/template.yaml \
     --output-template-file main-packaged.yml \
     --s3-bucket ${DEPLOY_BUCKET_NAME} \
     --profile ${AWS_PROFILE}

 sam deploy \
     --template-file main-packaged.yml \
     --stack-name ${stack_name} \
     --region ${region} \
     --tags Project=${stack_name} \
     --s3-bucket ${DEPLOY_BUCKET_NAME} \
     --parameter-overrides \
       ParameterKey=Prefix,ParameterValue=${PREFIX} \
       ParameterKey=CognitoUserPoolId,ParameterValue=${COGNITO_POOL_ID} \
       ParameterKey=CognitoUserPoolClientId,ParameterValue=${COGNITO_POOL_CLIENT_ID} \
     --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND CAPABILITY_NAMED_IAM \
     --profile ${AWS_PROFILE}

echo "[SUCCESS] DONE BUILDING SMGT-WORKFLOW"
