"""
Set up the AWS IAM execution role for this SageMaker AI training session APIs example.

The role is named SageMakerTrainingSessionRole and is created under the /service-role/ path,
which the AmazonSageMakerTrainingSessionAccess managed policy allows callers to pass to the
training sessions service. It is trusted by the training sessions service and by SageMaker AI
Inference, so the same role serves the training notebook (as the session role_arn) and the deploy
notebook (as the model and endpoint ExecutionRoleArn). It gets:

    - the AmazonSageMakerTrainingSessionFullAccess managed policy (training sessions)
    - an inline policy for hosting: JumpStart cache read, pulling the DJL LMI container from
      Amazon ECR, and writing endpoint logs to CloudWatch

The identity that runs the training notebook needs the AmazonSageMakerTrainingSessionAccess
managed policy. The deploy notebook additionally needs SageMaker AI hosting permissions and
iam:PassRole on this role for sagemaker.amazonaws.com.

Prerequisites:
    - AWS credentials that can call iam:CreateRole, iam:AttachRolePolicy, and iam:PutRolePolicy
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
    choices=["us-east-1", "us-west-2", "ap-south-1"],
    help="Region supported by SageMaker AI training session APIs (default: us-east-1)",
)
args = parser.parse_args()

REGION = args.region
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

ROLE_NAME = "SageMakerTrainingSessionRole"
ROLE_PATH = "/service-role/"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role{ROLE_PATH}{ROLE_NAME}"
MANAGED_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSageMakerTrainingSessionFullAccess"


iam = boto3.client("iam")

print(f"Account: {ACCOUNT_ID}")
print(f"Region:  {REGION}")
print("---")


# Step 1: Create the IAM role
print("\n[1/2] Creating IAM role...")
trust_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            # Trusted by both the Training Sessions service (sm-training_sessions_1_train_grpo_gsm8k.ipynb) and
            # SageMaker Inference (sm-training_sessions_2_deploy_grpo_checkpoint.ipynb) -- one role covers both.
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
        Path=ROLE_PATH,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Training + inference execution role for the SageMaker AI training session APIs example",
    )
    print(f"  Created role: {ROLE_ARN}")
except iam.exceptions.EntityAlreadyExistsException:
    existing = iam.get_role(RoleName=ROLE_NAME)["Role"]
    if existing["Path"] != ROLE_PATH:
        raise SystemExit(
            f"  {existing['Arn']} exists outside {ROLE_PATH}. The AmazonSageMakerTrainingSessionAccess "
            f"managed policy only allows passing roles under {ROLE_PATH}. Delete it with "
            "`python infrastructure/cleanup_infra.py --delete-roles`, then run this script again."
        )
    iam.update_assume_role_policy(RoleName=ROLE_NAME, PolicyDocument=json.dumps(trust_policy))
    print(f"  Role already exists: {ROLE_ARN}")


# Step 2: Attach permissions
print("\n[2/2] Attaching permissions...")
iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=MANAGED_POLICY_ARN)
print(f"  Attached managed policy: {MANAGED_POLICY_ARN}")

hosting_policy = {
    "Version": "2012-10-17",
    "Statement": [
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
            # Endpoint startup and inference logs from the deploy notebook's hosting container.
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
    ],
}
for stale in iam.list_role_policies(RoleName=ROLE_NAME)["PolicyNames"]:
    if stale != "InferenceHostingPermissions":
        iam.delete_role_policy(RoleName=ROLE_NAME, PolicyName=stale)
        print(f"  Removed inline policy from an earlier setup: {stale}")
iam.put_role_policy(
    RoleName=ROLE_NAME,
    PolicyName="InferenceHostingPermissions",
    PolicyDocument=json.dumps(hosting_policy),
)
print("  Attached inline policy: InferenceHostingPermissions")

print("\n" + "=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print(f"  role_arn: {ROLE_ARN}")
print(f"  region:   {REGION}")
print(
    "\nThe identity that runs the notebooks needs the AmazonSageMakerTrainingSessionAccess managed\n"
    "policy. The notebooks discover the Region, account, role, and bucket at runtime."
)
