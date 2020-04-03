#!/bin/bash

# AWS Region; customize as needed 
AWS_REGION='us-east-2'
[[ -z "${AWS_REGION}" ]] && echo "AWS_REGION is required" && exit 1 

# Cutomize bucket name
S3_BUCKET='stvwhite-coco'
[[ -z "${S3_BUCKET}" ]] && echo "S3_BUCKET is required" && exit 1 

DATE=`date +%s`

#Customize stack name as needed
STACK_NAME="sm-stack-$DATE"

# customzie note book instance name as needed
NOTEBOOK_INSTANCE_NAME="ngc-tf-mask-rcnn-$DATE"

# cfn template name
CFN_TEMPLATE='cfn-sm.yaml'

# Notebook instance type
# ml.m5.2xlarge or ml.p3.2xlarge
NOTEBOOK_INSTANCE_TYPE='ml.m5.2xlarge'

# EBS volume size 100 - 500 GB
EBS_VOLUME_SIZE=100

aws cloudformation create-stack --region $AWS_REGION  --stack-name $STACK_NAME \
--template-body file://$CFN_TEMPLATE \
--capabilities CAPABILITY_NAMED_IAM \
--parameters \
ParameterKey=NotebookInstanceType,ParameterValue=$NOTEBOOK_INSTANCE_TYPE \
ParameterKey=NotebookInstanceName,ParameterValue=$NOTEBOOK_INSTANCE_NAME \
ParameterKey=S3BucketName,ParameterValue=$S3_BUCKET \
ParameterKey=EbsVolumeSize,ParameterValue=$EBS_VOLUME_SIZE

echo "Creating stack [ eta 600 seconds ]"
sleep 30

progress=$(aws --region $AWS_REGION cloudformation list-stacks --stack-status-filter 'CREATE_IN_PROGRESS' | grep $STACK_NAME | wc -l)
while [ $progress -ne 0 ]; do
let elapsed="`date +%s` - $DATE"
echo "Stack $STACK_NAME status: Create in progress: [ $elapsed secs elapsed ]"
sleep 30
progress=$(aws --region $AWS_REGION  cloudformation  list-stacks --stack-status-filter 'CREATE_IN_PROGRESS' | grep $STACK_NAME | wc -l)
done
sleep 5
aws --region $AWS_REGION  cloudformation describe-stacks --stack-name $STACK_NAME
