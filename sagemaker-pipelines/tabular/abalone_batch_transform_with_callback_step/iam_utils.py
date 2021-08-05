import json
import boto3
iam = boto3.client('iam')


def create_lambda_role(role_name):
    try:
        response = iam.create_role(
            RoleName = role_name,
            AssumeRolePolicyDocument = json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }),
            Description='Role for Lambda to call ECS Fargate task'
        )

        role_arn = response['Role']['Arn']

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )

        role_policy_document = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:DescribeModelPackage",
                        "sagemaker:ListModelPackageGroups",
                        "sagemaker:ListModelPackages",
                        "sagemaker:SendPipelineExecutionStepSuccess",
                        "sagemaker:SendPipelineExecutionStepFailure"
                    ],
                    "Resource": "*"
                }
            ]
        })

        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName='lambda_sagemaker',
            PolicyDocument=role_policy_document
        )
        return role_arn
    except Exception as e:
        raise e