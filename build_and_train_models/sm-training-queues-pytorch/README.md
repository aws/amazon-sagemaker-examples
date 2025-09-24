# Amazon SageMaker Job Queuing, powered by AWS Batch

AWS Batch enables efficient queuing and resource management for your SageMaker Training Jobs.

## Getting Started

The instructions below are designed to get you going with this feature quickly.

### Setup IAM permissions

The example notebooks require permissions to invoke AWS Batch APIs.  
Below is a sample IAM policy granting these permissions - this should be added to the role being used to execute the notebooks (which the notebooks use as both the role to invoke AWS Batch and the role passed to SageMaker for training execution).

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["batch:*"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["iam:CreateServiceLinkedRole"],
      "Resource": "arn:aws:iam::*:role/*AWSServiceRoleForAWSBatchWithSagemaker",
      "Condition": {
        "StringEquals": {
          "iam:AWSServiceName": "sagemaker-queuing.batch.amazonaws.com"
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": ["sagemaker.amazonaws.com"]
        }
      }
    }
  ]
}
```

### Python setup

In order to use the feature, the python `boto3` library needs to be installed.

```
pip install -U boto3 sagemaker
```

### Create AWS Batch queues

To run the [examples](./examples) provided, the Batch queue need to be created. Refer to [smtj_batch_utils](./smtj_batch_utils/README.md) for additional information.
