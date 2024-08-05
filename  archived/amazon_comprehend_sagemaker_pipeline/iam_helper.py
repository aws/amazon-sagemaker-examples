import boto3
import json

iam = boto3.client("iam")


def create_lambda_role(role_name):
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for Lambda to call SageMaker functions",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.attach_role_policy(
            RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/AWSLambda_FullAccess"
        )

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        )

        response = iam.attach_role_policy(
            RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/ComprehendFullAccess"
        )

        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]


def delete_lambda_role(role_name):
    response = iam.detach_role_policy(
        RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/AWSLambda_FullAccess"
    )

    response = iam.detach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    )

    response = iam.detach_role_policy(
        RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/ComprehendFullAccess"
    )

    response = iam.delete_role(RoleName=role_name)

    return response
