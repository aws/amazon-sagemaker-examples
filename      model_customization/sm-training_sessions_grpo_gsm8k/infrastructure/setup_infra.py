"""
Set up the single AWS IAM role needed for this SageMaker Training Sessions
example: one training + inference execution role, trusted by both the
Training Sessions service and SageMaker Inference, with permissions for the
S3 default bucket, SageMaker Hub content, Model Package operations, Training
Session runtime operations, and pulling the DJL LMI inference container from
ECR.

There's only one role -- it's passed as both the training session's role_arn
(sm-training_sessions_grpo_gsm8k.ipynb) and the inference ExecutionRoleArn (sm-training_sessions_deploy_grpo_checkpoint.ipynb).

Nothing else needs to be provisioned ahead of time: the notebooks discover
their own region/account/role/bucket at runtime via
`sagemaker.core.helper.session_helper.Session()` and `get_execution_role()`
(the same pattern SageMaker Studio notebooks use), and the S3 default bucket
is created automatically on first use. There's no config file to manage.

Prerequisites:
    - AWS credentials configured (aws configure), or run from an IAM-capable
      environment (admin CLI, CloudShell) -- creating IAM roles requires
      iam:CreateRole / iam:PutRolePolicy
    - boto3 installed (pip install boto3)

Usage:
    python setup_infra.py
    python setup_infra.py --region us-west-2
"""

import argparse
import json

import boto3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--region",
    default="us-east-1",
    choices=["us-east-1", "us-west-2"],
    help="Region supported by SageMaker Training Sessions (default: us-east-1)",
)
args = parser.parse_args()

REGION = args.region
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

ROLE_NAME = "SageMakerTrainingSessionRole"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}"

# The conventional SageMaker default-bucket name -- what Session().default_bucket()
# resolves to when no custom default bucket is configured for the account. Not
# created here; the notebooks create it on first use via the SDK. This is only
# used to scope the S3 permissions below. If your account uses a custom default
# bucket, update this.
BUCKET_NAME = f"sagemaker-{REGION}-{ACCOUNT_ID}"

iam = boto3.client("iam")

print(f"Account: {ACCOUNT_ID}")
print(f"Region:  {REGION}")
print(f"Bucket (conventional default): {BUCKET_NAME}")
print("---")


# Step 1: Create the IAM role
print("\n[1/2] Creating IAM role...")
trust_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            # Trusted by both the Training Sessions service (sm-training_sessions_grpo_gsm8k.ipynb) and
            # SageMaker Inference (sm-training_sessions_deploy_grpo_checkpoint.ipynb) -- one role covers both.
            "Principal": {
                "Service": [
                    "training-session.sagemaker.amazonaws.com",
                    "sagemaker.amazonaws.com",
                ]
            },
            "Action": ["sts:AssumeRole", "sts:TagSession"],
        }
    ],
}
try:
    iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Training + inference execution role for the SageMaker Training Sessions example",
    )
    print(f"  Created role: {ROLE_ARN}")
except iam.exceptions.EntityAlreadyExistsException:
    print(f"  Role already exists: {ROLE_ARN}")


# Step 2: Attach permissions policy
print("\n[2/2] Attaching permissions policy...")
permissions_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3DefaultBucketAccess",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:CreateBucket"],
            "Resource": [f"arn:aws:s3:::{BUCKET_NAME}", f"arn:aws:s3:::{BUCKET_NAME}/*"],
        },
        {
            "Sid": "JumpStartCacheAccess",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                f"arn:aws:s3:::jumpstart-cache-prod-{REGION}",
                f"arn:aws:s3:::jumpstart-cache-prod-{REGION}/*",
            ],
        },
        {
            "Sid": "HubContentAccess",
            "Effect": "Allow",
            "Action": ["sagemaker:DescribeHubContent"],
            "Resource": "*",
        },
        {
            "Sid": "ModelPackageAccess",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateModelPackageGroup",
                "sagemaker:DescribeModelPackageGroup",
                "sagemaker:DeleteModelPackageGroup",
                "sagemaker:CreateModelPackage",
                "sagemaker:DescribeModelPackage",
                "sagemaker:DeleteModelPackage",
                "sagemaker:ListModelPackages",
            ],
            "Resource": "*",
        },
        {
            "Sid": "TrainingSessionRuntimeAccess",
            "Effect": "Allow",
            "Action": [
                "sagemaker:ExecuteSampleOperation",
                "sagemaker:GetSampleOperationResult",
                "sagemaker:ExecuteTrainingOperation",
                "sagemaker:GetTrainingOperationResult",
            ],
            "Resource": "*",
        },
        {
            "Sid": "ECRAccess",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
                "ecr:GetAuthorizationToken",
            ],
            "Resource": "*",
        },
        {
            # Lets the hosting container (sm-training_sessions_deploy_grpo_checkpoint.ipynb) publish its startup/inference
            # logs to CloudWatch. Without this the endpoint still runs, but no log group
            # is ever created, so failures like a failed ping health check are undebuggable.
            "Sid": "CloudWatchLogsForHosting",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
            ],
            "Resource": f"arn:aws:logs:*:{ACCOUNT_ID}:log-group:/aws/sagemaker/*",
        },
        {
            "Sid": "PassRole",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": ROLE_ARN,
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": [
                        "training-session.sagemaker.amazonaws.com",
                        "sagemaker.amazonaws.com",
                    ]
                }
            },
        },
    ],
}
iam.put_role_policy(
    RoleName=ROLE_NAME,
    PolicyName="TrainingSessionAndInferencePermissions",
    PolicyDocument=json.dumps(permissions_policy),
)
print("  Permissions policy attached")

print("\n" + "=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print(f"  role_arn: {ROLE_ARN}")
print(f"  region:   {REGION}")
print(
    "\nNothing else to configure -- sm-training_sessions_grpo_gsm8k.ipynb, and sm-training_sessions_deploy_grpo_checkpoint.ipynb\n"
    "all discover region/account/role/bucket automatically at runtime."
)
