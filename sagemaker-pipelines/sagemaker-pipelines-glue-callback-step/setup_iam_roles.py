import json
import boto3

iam = boto3.client("iam")


def create_ecs_task_role(role_name):
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for ECS task execution",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
        )

        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName="create_log_group",
            PolicyDocument='{"Version":"2012-10-17","Statement":{"Effect":"Allow","Action":"logs:CreateLogGroup","Resource":"*"}}',
        )

        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]


def create_task_runner_role(role_name):
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for ECS tasks",
        )

        role_arn = response["Role"]["Arn"]

        role_policy_document = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": "sagemaker:*", "Resource": "*"},
                    {
                        "Effect": "Allow",
                        "Action": ["glue:StartJobRun", "glue:GetJobRun"],
                        "Resource": "*",
                    },
                    {"Effect": "Allow", "Action": "logs:CreateLogGroup", "Resource": "*"},
                ],
            }
        )

        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName="glue_logs_sagemaker",
            PolicyDocument=role_policy_document,
        )

        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName="create_log_group",
            PolicyDocument='{"Version":"2012-10-17","Statement":{"Effect":"Allow","Action":"logs:CreateLogGroup","Resource":"*"}}',
        )

        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]


def create_glue_pipeline_role(role_name, bucket):
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "glue.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for Glue ETL job",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.attach_role_policy(
            RoleName=role_name, PolicyArn="arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
        )

        role_policy_document = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": "s3:*", "Resource": f"arn:aws:s3:::{bucket}"}
                ],
            }
        )

        response = iam.put_role_policy(
            RoleName=role_name, PolicyName="glue_s3_bucket", PolicyDocument=role_policy_document
        )

        role_policy_document = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": "s3:*", "Resource": f"arn:aws:s3:::{bucket}/*"}
                ],
            }
        )

        response = iam.put_role_policy(
            RoleName=role_name, PolicyName="glue_s3_objects", PolicyDocument=role_policy_document
        )

        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]


def create_lambda_sm_pipeline_role(role_name, ecs_role_arn, task_role_arn):
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
            Description="Role for Lambda to call ECS Fargate task",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        role_policy_document = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": "ecs:RunTask", "Resource": ["*"]},
                    {"Effect": "Allow", "Action": "sqs:*", "Resource": ["*"]},
                    {"Effect": "Allow", "Action": "sagemaker:*", "Resource": ["*"]},
                    {
                        "Effect": "Allow",
                        "Action": "iam:PassRole",
                        "Resource": [ecs_role_arn, task_role_arn],
                    },
                ],
            }
        )

        response = iam.put_role_policy(
            RoleName=role_name, PolicyName="ecs_sqs_sagemaker", PolicyDocument=role_policy_document
        )
        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]
