# Distributed TensorFlow training using Amazon SageMaker

## Prerequisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. [Manage your SageMaker service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/). You will need a minimum limit of 2 ```ml.p3.16xlarge``` and 2 ```ml.p3dn.24xlarge``` instance types, but a service limit of 4 for each instance type is recommended. Keep in mind that the service limit is specific to each AWS region. We recommend using ```us-west-2``` region for this tutorial.

3. Create an [Amazon S3 bucket](https://docs.aws.amazon.com/en_pv/AmazonS3/latest/gsg/CreatingABucket.html) in the AWS region where you would like to execute this tutorial. Save the S3 bucket name. You will need it later.

## Mask-RCNN training

In this tutorial, our focus is distributed [TensorFlow](https://github.com/tensorflow/tensorflow) training using [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

Concretely, we will discuss distributed TensorFlow training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) and [AWS Samples Mask R-CNN](https://github.com/aws-samples/mask-rcnn-tensorflow) algorithms using [COCO 2017 dataset](http://cocodataset.org/#home).

This tutorial has two key steps:

1. We use [Amazon CloudFormation](https://aws.amazon.com/cloudformation/) to create a new [Sagemaker notebook instance](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/nbi.html) in an [Amazon Virtual Private Network (VPC)](https://aws.amazon.com/vpc/).

2. We use the SageMaker notebook instance to launch distributed training jobs in the VPC using [Amazon S3](https://aws.amazon.com/s3/), [Amazon EFS](https://aws.amazon.com/efs/), or [Amazon FSx Lustre](https://aws.amazon.com/fsx/) as data source for training data pipeline.

If you are viewing this page from a SageMaker notebook instance and wondering why we need a new SageMaker notebook instance. the reason is that your current SageMaker notebook instance may not be running in a VPC, may not have an [IAM Role](https://docs.aws.amazon.com/en_pv/IAM/latest/UserGuide/id_roles.html) attached that provides access to required AWS resources, or may not have access to [EFS mount targets](https://docs.aws.amazon.com/en_pv/efs/latest/ug/accessing-fs.html) that we need for this tutorial.

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

The estimated time for creating this CloudFormation stack is 9 minutes. The stack will create following AWS resources:
   
   1. A [SageMaker execution role](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/sagemaker-roles.html)
   2. A [Virtual Private Network (VPC)](https://aws.amazon.com/vpc/) with Internet Gateway (IGW), 1 public subnet, 3 private subnets, a NAT gateway, a [Security Group](https://docs.aws.amazon.com/en_pv/vpc/latest/userguide/VPC_SecurityGroups.html), and a [VPC Gateway Endpoint to S3](https://docs.aws.amazon.com/en_pv/vpc/latest/userguide/vpc-endpoints-s3.html)
   3. An optional [Amazon EFS](https://aws.amazon.com/efs/) file system with mount targets in each private subnet in the VPC.
   4. A [SageMaker Notebook instance](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/nbi.html) in the VPC:
      * The EFS file-system is mounted on the SageMaker notebook instance
      * The SageMaker execution role attached to the notebook instance provides appropriate IAM access to AWS resources
      
*Save the summary output of the script. You will need it later. You can also view the output under CloudFormation Stack Outputs tab in AWS Management Console.*

#### Create SageMaker notebook instance in an existing VPC

This option is only recommended for advanced AWS users. Make sure your existing VPC has following:
  * One or more security groups
  * One or more private subnets with NAT Gateway access and existing EFS file-system mount targets
  * Endpoint gateway to S3
  
 [Create a SageMaker notebook instance](https://docs.aws.amazon.com/en_pv/sagemaker/latest/dg/howitworks-create-ws.html) in a VPC using AWS SageMaker console. When you are creating the SageMaker notebook instance, add at least 100 GB of local EBS volume under advanced configuration options. You will also need to [mount your EFS file-system on the SageMaker notebook instanxce](https://aws.amazon.com/blogs/machine-learning/mount-an-efs-file-system-to-an-amazon-sagemaker-notebook-with-lifecycle-configurations/).

### Launch SageMaker tranining jobs

In SageMaker console, open  the Juypter Lab notebook server you created in the previous step. In this Juypter Lab instance, there are three Jupyter notebooks for training Mask R-CNN. All three notebooks use [SageMaker TensorFlow Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/sagemaker.tensorflow.html) in Script Mode, whereby we can keep the SageMaker entry point script outside the Docker container, and pass it as a parameter to SageMaker TensorFlow Estimator. The SageMaker TensorFlow Estimator also allows us to specify the ```distribution``` type, which means we don't have to write code in the entry point script for managing SageMaker distributed training, which greatly simplifies the entry point script. 

The three SageMaker Script Mode notebooks for training Mask R-CNN are listed below:

- Mask R-CNN notebook that uses S3 bucket as data source: [```mask-rcnn-scriptmode-s3.ipynb```](mask-rcnn-scriptmode-s3.ipynb)
- Mask R-CNN notebook that uses EFS file-system as data source: [```mask-rcnn-scriptmode-efs.ipynb```](mask-rcnn-scriptmode-efs.ipynb)
- Mask R-CNN notebook that uses FSx Lustre file-system as data source: [```mask-rcnn-scriptmode-fsx.ipynb```](mask-rcnn-scriptmode-fsx.ipynb)

Following notebooks use [SageMaker Estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html), and are **deprecated** in favor of the notebooks listed above:

- [```mask-rcnn-s3.ipynb```](mask-rcnn-s3.ipynb)
- [```mask-rcnn-efs.ipynb```](mask-rcnn-efs.ipynb)
- [```mask-rcnn-fsx.ipynb```](mask-rcnn-fsx.ipynb)


Below, we compare the three options, [Amazon S3](https://aws.amazon.com/s3/), [Amazon EFS](https://aws.amazon.com/efs/) and [Amazon FSx Lustre](https://aws.amazon.com/fsx/):

<table>
   <tr>
      <th>Data Source</th>
      <th>Description</th>
   </tr>
   <tr>
      <td>Amazon S3</td>
      <td>Each time the SageMaker training job is launched, it takes approximately 20 minutes to download COCO 2017 dataset from your S3 bucket to the <i>Amazon EBS volume</i> attached to each training instance. During training, data is input to the training data pipeline from the EBS volume attached to each training instance. 
      </td>
   </tr>
    <tr>
      <td>Amazon EFS</td>
      <td>It takes approximately 46 minutes to copy COCO 2017 dataset from your S3 bucket to your EFS file-system. You only need to copy this data once. During tranining, data is input from the shared <i>Amazon EFS file-system</i> mounted on all the training instances. 
      </td>
   </tr>
    <tr>
      <td>Amazon FSx Lustre</td>
      <td>It takes approximately 10 minutes to create a new FSx Lustre file-system and import COCO 2017 dataset from your S3 bucket to the new FSx Lustre file-system. You only need to do this once. During training, data is input from the shared <i>Amazon FSx Lustre file-system</i> mounted on all the training instances. 
      </td>
   </tr>
<table>

In all three cases, the logs and model checkpoints output during training are written to the EBS volume attached to each training instance, and uploaded to your S3 bucket when training completes. The logs are also fed into CloudWatch as training progresses and can be reviewed during training.  

System and model training metrics are fed into Amazon CloudWatch metrics during training and can be visualized in SageMaker console.

### Training samples results

Below are sample experiment results for the two algorithms, after training for 24 epochs on COCO 2017 dataset:

1. [TensorPack Mask/Faster-RCNN algorithm](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)

    * coco_val2017-mAP(bbox)/IoU=0.5: 0.59231
    * coco_val2017-mAP(bbox)/IoU=0.5:0.95: 0.3844
    * coco_val2017-mAP(bbox)/IoU=0.75: 0.41564
    * coco_val2017-mAP(bbox)/large: 0.51084
    * coco_val2017-mAP(bbox)/medium: 0.41643
    * coco_val2017-mAP(bbox)/small: 0.21634
    * coco_val2017-mAP(segm)/IoU=0.5: 0.56011
    * coco_val2017-mAP(segm)/IoU=0.5:0.95: 0.34917
    * coco_val2017-mAP(segm)/IoU=0.75: 0.37312
    * coco_val2017-mAP(segm)/large: 0.48118
    * coco_val2017-mAP(segm)/medium: 0.37815
    * coco_val2017-mAP(segm)/small: 0.18192
    
2. [AWS Samples Mask R-CNN algorithm](https://github.com/aws-samples/mask-rcnn-tensorflow)

    * mAP(bbox)/IoU=0.5: 0.5983
    * mAP(bbox)/IoU=0.5:0.95: 0.38296
    * mAP(bbox)/IoU=0.75: 0.41296
    * mAP(bbox)/large: 0.50688
    * mAP(bbox)/medium: 0.41901
    * mAP(bbox)/small: 0.21421
    * mAP(segm)/IoU=0.5: 0.56733
    * mAP(segm)/IoU=0.5:0.95: 0.35262
    * mAP(segm)/IoU=0.75: 0.37365
    * mAP(segm)/large: 0.48337
    * mAP(segm)/medium: 0.38459
    * mAP(segm)/small: 0.18244
   
