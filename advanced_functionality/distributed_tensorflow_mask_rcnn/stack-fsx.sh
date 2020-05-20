#!/bin/bash

if [ $# -lt 5 ]; then
    echo "usage: $0 <aws-region> <s3-import-path> <fsx-capacity> <subnet-id> <security-group-id>"
    exit 1
fi

# AWS Region
AWS_REGION=$1

# S3 import path
S3_IMPORT_PATH=$2

# FSx storage capacity in multiples of 3600
STORAGE_CAPACITY=$3

SUBNET_ID=$4

SG_ID=$5

DATE=`date +%s`

#Customize stack name as needed
STACK_NAME="fsx-stack-$DATE"

# cfn template name
CFN_TEMPLATE='cfn-fsx.yaml'

aws cloudformation create-stack --region $AWS_REGION  --stack-name $STACK_NAME \
--template-body file://$CFN_TEMPLATE \
--capabilities CAPABILITY_NAMED_IAM \
--parameters \
ParameterKey=S3ImportPath,ParameterValue=$S3_IMPORT_PATH \
ParameterKey=StorageCapacityGiB,ParameterValue=$STORAGE_CAPACITY \
ParameterKey=SecurityGroupId,ParameterValue=$SG_ID \
ParameterKey=SubnetId,ParameterValue=$SUBNET_ID 

echo "Creating FSx Luster file-system [eta 600 seconds]"

sleep 30

progress=$(aws cloudformation list-stacks --stack-status-filter 'CREATE_IN_PROGRESS' | grep $STACK_NAME | wc -l)
while [ $progress -ne 0 ]; do
let elapsed="`date +%s` - $DATE"
echo "Stack $STACK_NAME status: CREATE_IN_PROGRESS: [ $elapsed secs elapsed ]"
sleep 30 
progress=$(aws cloudformation list-stacks --stack-status-filter 'CREATE_IN_PROGRESS' | grep $STACK_NAME | wc -l)
done
sleep 5 
aws cloudformation describe-stacks --stack-name $STACK_NAME
