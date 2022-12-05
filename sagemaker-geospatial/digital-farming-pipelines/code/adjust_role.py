#!/usr/bin/python
#Script for adding policy and trust relationship to the role
import boto3
import sys
import json
import os

print('Argument List:', str(sys.argv))

policy_doc = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:*",
                "sagemaker-geospatial:*",
                "lambda:*",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "sqs:ReceiveMessage",
                "sqs:DeleteMessage",
                "sqs:GetQueueAttributes",
                "s3:*",
                "iam:PassRole",
            ],
            "Resource": "*",
        }
    ],
}

trust_relationship = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": [
                    "sagemaker.amazonaws.com",
                    "sagemaker-geospatial.amazonaws.com",
                    "lambda.amazonaws.com",
                    "s3.amazonaws.com"
                ]
            },
            "Action": "sts:AssumeRole",
        }
    ],
}



def create_policy(policy_doc):
    # Create IAM policy...
    try:
        print('Will create policy...')
        client = boto3.client('iam')
        policy = client.create_policy(
            PolicyName='geospatial-policy',
            PolicyDocument=json.dumps(policy_doc),
            Description='Policy for Digital Farming demo',
        )
        policy_arn = policy['Policy']['Arn']
        print("Created policy %s.", policy_arn)
    except:
        print("Couldn't create policy geospatial-policy.")
        raise
    return policy_arn

def attach_policy(role_name, policy_arn):
    # Attach policy to role...
    try:
        print(f'Will attach policy {policy_arn} to role {role_name}...')
        client = boto3.client('iam')
        response = client.attach_role_policy(
            RoleName=role_name,
            PolicyArn=policy_arn
        )
        print("Attached policy %s to role %s.", policy_arn, role_name)
    except:
        print("Couldn't attach policy %s to role %s.", policy_arn, role_name)
        raise

def edit_trust(role_name, trust_relationship):
    # Edit trust relationship...
    try:
        print('Will edit Trust Relationship...')
        client = boto3.client('iam')
        response = client.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=trust_relationship
        )
        print('Modified trust relationship for role')
    except:
        print("Couldn't edit the trust policy.")
        
def main():
    role = sys.argv[1]
    print(f'role_arn: {role}')
    role_name = role[role.rindex('/')+1:]
    print(f'role_name: {role_name}')

    policy_arn = create_policy(policy_doc)

    attach_policy(role_name, policy_arn)

    edit_trust(role_name, trust_relationship)

if __name__ == "__main__":
    main()