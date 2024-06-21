# Bring Your Own Active Learning

This repository contains the code that you will need to complete the tutorial in [Bring your own model for Amazon SageMaker labeling workflows with active learning](https://aws.amazon.com/blogs/machine-learning/bring-your-own-model-for-amazon-sagemaker-labeling-workflows-with-active-learning/).

With Amazon SageMaker Ground Truth, you can easily and inexpensively build more accurately labeled machine learning datasets. To decrease labeling costs, Ground Truth uses active learning to choose “difficult” images that require human annotation and “easy” images that can be automatically labeled with machine learning (automated labeling or auto-labeling).

Use Bring your own model for Amazon SageMaker labeling workflows with active learning to learn how to create an active learning workflow with your own algorithm to run training and inference in that workflow. This example can be used as a starting point to perform active learning and auto annotation with a custom labeling job.

#### Pre-req to build:
1) Install [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
2) Install [Python 3.7](https://www.python.org/downloads/)

#### How to build:

1) Clone the repository using `git clone` command 
2) go to folder `<github-root-dir>/ground_truth_labeling_jobs/bring_your_own_model_for_sagemaker_labeling_workflows_with_active_learning/src`
3) To build the code in this repository run: `sam build`

#### Pre-req to deploy:
1) Export the temporary AWS security token. Follow the instructions listed [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_use-resources.html#using-temp-creds-sdk-cli)
2) Create the S3 bucket which will be used when deploying the model.

#### How to deploy:
To create AWS resources required for creation of an active learning workflow, we use a [CloudFormation Stack](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/stacks.html). 

Run command: `sam deploy -t <template_path> --region <region> --stack-name <stack_name> --s3-bucket <your_bucket_name> --capabilities <capabilities>` where options are as follows
  * `--region` specifies the AWS Region where you would like to launch this Stack.
  * `--stack-name` name of the CloudFormation stack. 
  * `--s3-bucket` name of S3 bucket in the region specified by `--region` parameter value.
  * `capabilities` capabilities used in stack

For example:
`sam deploy -t .aws-sam/build/template.yaml --region us-west-2 --stack-name example-stack-name --s3-bucket your-bucket-name-us-west-2 --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND`

#### Unit test instructions:

1. Install test modules
`pip3 install pytest`
`pip3 install moto`
`pip3 install boto3`

2. Add lambda layer modules to the python path.
`export PYTHONPATH="<github-root-dir>/src/dependency/python"`

3. Run all tests
`python3 -m pytest`
