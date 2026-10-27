# AWS Batch for SageMaker Training jobs

These example notebooks demonstrate how to use AWS Batch for SageMaker Training jobs. SageMaker training queues allow you to efficiently manage and prioritize your training jobs, optimizing resource utilization and reducing wait times.

## IAM Permissions

The example notebooks require permissions to invoke AWS Batch APIs.  Below is a sample IAM policy granting these permissions - this should be added to the role being used to execute the notebooks (which the notebooks use as both the role to invoke AWS Batch and the role passed to SageMaker for training execution).

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "batch:*"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:CreateServiceLinkedRole"
            ],
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
                    "iam:PassedToService": [
                        "sagemaker.amazonaws.com"
                    ]
                }
            }
        }
    ]
}
```

**Note**: if you would like to create separate IAM roles for AWS Batch invocation and SageMaker training execution, there is sample code in the [utils/aws_batch_resource_management.py](utils/aws_batch_resource_management.py) module.

## Examples

The following notebooks provide hands-on examples for getting started with SageMaker training queues:

* [Getting Started with Training Queues using Estimator](training_queue_getting_started_with_estimator.ipynb) - Learn how to use SageMaker training queues with the an Estimator to queue and manage your training jobs.

* [Getting Started with Training Queues using ModelTrainer](training_queue_getting_started_with_model_trainer.ipynb) - Learn how to use SageMaker training queues with the a ModelTrainer to queue and manage your training jobs.
