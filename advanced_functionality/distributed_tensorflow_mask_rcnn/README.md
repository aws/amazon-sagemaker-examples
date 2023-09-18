# Distributed TensorFlow training using Amazon SageMaker

## Prerequisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. [Manage your SageMaker service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/). You will need a minimum limit of 2 ```ml.p3.16xlarge``` and 2 ```ml.p3dn.24xlarge``` instance types, but a service limit of 4 for each instance type is recommended. Keep in mind that the service limit is specific to each AWS region. We recommend using ```us-west-2``` region for this tutorial.

3. Create or use an existing [Amazon S3 bucket](https://docs.aws.amazon.com/en_pv/AmazonS3/latest/gsg/CreatingABucket.html) in the AWS region where you would like to execute this tutorial. Save the S3 bucket name. You will need it later.

## Mask-RCNN training

In this tutorial, our focus is distributed [TensorFlow](https://github.com/tensorflow/tensorflow) training using [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

Concretely, we will discuss distributed TensorFlow training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) and [AWS Samples Mask R-CNN](https://github.com/aws-samples/mask-rcnn-tensorflow) models using [COCO 2017 dataset](http://cocodataset.org/#home).

This tutorial has two key steps:

1. We use [Amazon CloudFormation](https://aws.amazon.com/cloudformation/) to create a new [Sagemaker notebook instance](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/nbi.html) in an [Amazon Virtual Private Network (VPC)](https://aws.amazon.com/vpc/). We also automatically create an [Amazon EFS](https://aws.amazon.com/efs/) file-system, and an [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) file-system, and mount both the file-systems on the notebook instance.

2. We use the SageMaker notebook instance to launch distributed training jobs in the VPC using [Amazon S3](https://aws.amazon.com/s3/), [Amazon EFS](https://aws.amazon.com/efs/), or [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) as data source for input training data.

### Create SageMaker notebook instance in a VPC
Our objective in this step is to create a SageMaker notebook instance in a VPC. We have two options. We can create a SageMaker notebook instance in a new VPC, or we can create the notebook instance in an existing VPC. We cover both options below.

#### Create SageMaker notebook instance in a new VPC

The [AWS IAM User](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/id_users.html) or [AWS IAM Role](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/id_roles.html) executing this step requires [AWS IAM permissions](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/access_policies_job-functions.html) consistent with [Network Administrator](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/access_policies_job-functions.html) job function.

The CloudFormation template [cfn-sm.yaml](cfn-sm.yaml) can be used to create a [CloudFormation stack](https://docs.aws.amazon.com/en_pv/AWSCloudFormation/latest/UserGuide/stacks.html) that creates a SageMaker notebook instance in a new VPC. 

You can [create the CloudFormation stack](https://docs.aws.amazon.com/en_pv/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html) using [cfn-sm.yaml](cfn-sm.yaml) directly in CloudFormation service console. 

Alternatively, you can customize variables in [stack-sm.sh](stack-sm.sh) script and execute the script anywhere you have [AWS Command Line Interface (CLI)](https://docs.aws.amazon.com/en_pv/cli/latest/userguide/cli-chap-welcome.html) installed. The CLI option is detailed below:

   * [Install AWS CLI](https://docs.aws.amazon.com/en_pv/cli/latest/userguide/cli-chap-install.html) 
   * In ```stack-sm.sh```, set ```AWS_REGION``` to your AWS region and ```S3_BUCKET``` to your S3 bucket . These two variables are required. 
   * Optionally, you can set ```EFS_ID``` variable if you want to use an existing EFS file-system. If you leave ```EFS_ID``` blank, a new EFS file-system is created. If you chose to use an existing EFS file-system, make sure the existing file-system does not have any existing [mount targets](https://docs.aws.amazon.com/en_pv/efs/latest/ug/managing.html). 
   * Optionally, you can specify ```GIT_URL``` to add a Git-hub repository to the SageMaker notebook instance. If the Git-hub repository is private, you can specify ```GIT_USER``` and ```GIT_TOKEN``` variables.
   * Execute the customized ```stack-sm.sh``` script to create a CloudFormation stack using AWS CLI. 

The estimated time for creating this CloudFormation stack is 30 minutes. The stack will create following AWS resources:
   
   1. A [SageMaker execution role](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/sagemaker-roles.html)
   2. A [Virtual Private Network (VPC)](https://aws.amazon.com/vpc/) with Internet Gateway (IGW), 1 public subnet, 3 private subnets, a NAT gateway, a [Security Group](https://docs.aws.amazon.com/en_pv/vpc/latest/userguide/VPC_SecurityGroups.html), and a [VPC Gateway Endpoint to S3](https://docs.aws.amazon.com/en_pv/vpc/latest/userguide/vpc-endpoints-s3.html)
   3. [Amazon EFS](https://aws.amazon.com/efs/) file system with mount targets in each private subnet in the VPC.
   4. [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) file system in the VPC.
   5. A [SageMaker Notebook instance](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/nbi.html) in the VPC:
      * The EFS file-system is mounted on the SageMaker notebook instance
      * The FSx for Lustre file-system is mounted on the SageMaker notebook instance
      * The SageMaker execution role attached to the notebook instance provides appropriate IAM access to AWS resources

#### Create SageMaker notebook instance in an existing VPC

This option is only recommended for **advanced AWS users**. Make sure your existing VPC has following:
  * One or more security groups
  * One or more private subnets with NAT Gateway access and existing EFS file-system mount targets
  * Endpoint gateway to S3
  
 [Create a SageMaker notebook instance](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/howitworks-create-ws.html) in a VPC using AWS SageMaker console. When you are creating the SageMaker notebook instance, add at least 200 GB of local EBS volume under advanced configuration options. You will also need to [mount your EFS file-system on the SageMaker notebook instance](https://aws.amazon.com/blogs/machine-learning/mount-an-efs-file-system-to-an-amazon-sagemaker-notebook-with-lifecycle-configurations/), [mount your FSx for Lustre file-system on the SageMaker notebook instance](https://docs.aws.amazon.com/fsx/latest/LustreGuide/mount-fs-auto-mount-onreboot.html).

### Launch SageMaker training jobs

Jupyter notebooks for training Mask R-CNN are listed below:

- Mask R-CNN notebook that uses S3 bucket, or EFS, as data source: [mask-rcnn-scriptmode-experiment-trials.ipynb](mask-rcnn-scriptmode-experiment-trials.ipynb)
- Mask R-CNN notebook that uses S3 bucker, or FSx for Lustre file-system as data source: [```mask-rcnn-scriptmode-fsx.ipynb```](mask-rcnn-scriptmode-fsx.ipynb)
   
