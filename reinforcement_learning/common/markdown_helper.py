# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

def generate_s3_write_permission_for_sagemaker_role(role):
    role_name = role.split("/")[-1]
    url = "https://console.aws.amazon.com/iam/home#/roles/%s" % role_name
    text = "1. Go to IAM console to edit current SageMaker role: [%s](%s).\n" % (role_name, url)
    text += "2. Next, go to the `Permissions tab` and click on `Attach Policy.` \n"
    text += "3. Search and select `AmazonKinesisVideoStreamsFullAccess` policy\n"
    return text

def generate_kinesis_create_permission_for_sagemaker_role(role):
    role_name = role.split("/")[-1]
    url = "https://console.aws.amazon.com/iam/home#/roles/%s" % role_name
    text = "1. Go to IAM console to edit current SageMaker role: [%s](%s).\n" % (role_name, url)
    text += "2. Next, go to the `Permissions tab` and click on `Attach Policy.` \n"
    text += "3. Search and select `AmazonS3FullAccess` policy\n"
    return text

def generate_help_for_s3_endpoint_permissions(role):
    role_name = role.split("/")[-1]
    url = "https://console.aws.amazon.com/iam/home#/roles/%s" % role_name
    text = ">It looks like your SageMaker role has insufficient premissions. Please do the following:\n"
    text += "1. Go to IAM console to edit current SageMaker role: [%s](%s).\n" % (role_name, url)
    text += "2. Select %s and then click on `Edit Policy`\n" % role_name
    text += "3. Select the JSON tab and add the following JSON blob to the `Statement` list:\n"
    text += """```json
            {
            "Action": [
                "ec2:DescribeRouteTables",
                "ec2:CreateVpcEndpoint"
            ],
            "Effect": "Allow",
            "Resource": "*"
            },```\n"""
    text += "4. Now wait for a few minutes before executing this cell again!"
    return text


def generate_help_for_robomaker_trust_relationship(role):
    role_name = role.split("/")[-1]
    url = "https://console.aws.amazon.com/iam/home#/roles/%s" % role_name
    text = "1. Go to IAM console to edit current SageMaker role: [%s](%s).\n" % (role_name, url)
    text += "2. Next, go to the `Trust relationships tab` and click on `Edit Trust Relationship.` \n"
    text += "3. Replace the JSON blob with the following:\n"
    text += """```json
            {
              "Version": "2012-10-17",
              "Statement": [
                {
                  "Effect": "Allow",
                  "Principal": {
                    "Service": [
                      "sagemaker.amazonaws.com",
                      "robomaker.amazonaws.com"
                    ]
                  },
                  "Action": "sts:AssumeRole"
                }
              ]
            }```\n"""
    text += "4. Once this is complete, click on Update Trust Policy and you are done."
    return text


def generate_help_for_robomaker_all_permissions(role):
    role_name = role.split("/")[-1]
    url = "https://console.aws.amazon.com/iam/home#/roles/%s" % role_name
    text = ">It looks like your SageMaker role has insufficient premissions. Please do the following:\n"
    text += "1. Go to IAM console to edit current SageMaker role: [%s](%s).\n" % (role_name, url)
    text += "2. Click on policy starting with `AmazonSageMaker-ExecutionPolicy` and then edit policy.\n"
    text += "3. Go to JSON tab, add the following JSON blob to the `Statement` list and save policy:\n"
    text += """```json
        {
            "Effect": "Allow",
            "Action": [
                "robomaker:CreateSimulationApplication",
                "robomaker:DescribeSimulationApplication",
                "robomaker:DeleteSimulationApplication",
                "robomaker:CreateSimulationJob",
                "robomaker:DescribeSimulationJob",
                "robomaker:CancelSimulationJob",
                "robomaker:ListSimulationApplications"
            ],
            "Resource": [
                "*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": "iam:CreateServiceLinkedRole",
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "iam:AWSServiceName": "robomaker.amazonaws.com"
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": [
                        "robomaker.amazonaws.com"
                    ]
                }
            }
        },```\n"""
    text += "4. Next, go to the `Trust relationships tab` and click on `Edit Trust Relationship.` \n"
    text += "5. Add the following JSON blob to the `Statement` list:\n"
    text += """```json
            {
              "Effect": "Allow",
              "Principal": {
                "Service": "robomaker.amazonaws.com"
              },
              "Action": "sts:AssumeRole"
            },```\n"""
    text += "6. Now wait for a few minutes before executing this cell again!"
    return text


def generate_robomaker_links(job_arns, aws_region):
    simulation_ids = [job_arn.split("/")[-1] for job_arn in job_arns]
    robomaker_links = []
    for simulation_id in simulation_ids:
        robomaker_link = "https://%s.console.aws.amazon.com/robomaker/home?region=%s#simulationJobs/%s" % (aws_region,
                                                                                                           aws_region,
                                                                                                           simulation_id)
        robomaker_links.append(robomaker_link)

    markdown_content = '> Click on the following links for visualization of simulation jobs on RoboMaker Console\n'
    for i in range(len(robomaker_links)):
        markdown_content += "- [Simulation %s](%s)  \n" % (i + 1, robomaker_links[i])

    markdown_content += "\nYou can click on Gazebo after you open the above link to start the simulator."
    return markdown_content


def create_s3_endpoint_manually(aws_region, default_vpc):
    url = "https://%s.console.aws.amazon.com/vpc/home?region=%s#Endpoints:sort=vpcEndpointId" % (aws_region, aws_region)
    text = ">VPC S3 endpoint creation failed. Please do the following to create an endpoint manually:\n"
    text += "1. Go to [VPC console | Endpoints](%s)\n" % url
    text += "2. Click on `Create Endpoint`. Select Service Name as `com.amazonaws.%s.s3`.\n" % (aws_region)
    text += "3. Next, select your Default VPC: `%s` and click the checkbox against the main Route Table ID\n" % (
    default_vpc)
    text += "4. Select `Full Access` in policy and click on `Create Endpoint`\n"
    text += "5. That should be it! Now wait for a few seconds before proceeding to the next cell."
    return text


def generate_help_for_administrator_policy(role):
    role_name = role.split("/")[-1]
    url = "https://console.aws.amazon.com/iam/home#/roles/%s" % role_name
    text = "1. Go to IAM console to edit current SageMaker role: [%s](%s).\n" % (role_name, url)
    text += "2. Next, go to the `Permissions tab` and click on `Attach policies`. \n"
    text += "3. Check the box for `AdministratorAccess`\n"
    text += "4. Click on `Attach policy` at the bottom.\n"
    text += "5. You'll see message `Policy AdministratorAccess has been attached for the %s`. \n" % (role)
    text += "6. Once this is complete, you are all set."
    return text

def generate_help_for_experiment_manager_permissions(role):
    role_name = role.split("/")[-1]
    url = "https://console.aws.amazon.com/iam/home#/roles/%s" % role_name
    text = ">It looks like your SageMaker role has insufficient premissions. Please do the following:\n"
    text += "1. Go to IAM console to edit current SageMaker role: [%s](%s).\n" % (role_name, url)
    text += "2. Click on policy starting with `AmazonSageMaker-ExecutionPolicy` and then edit policy.\n"
    text += "3. Go to JSON tab, add the following JSON blob to the `Statement` list and save policy:\n"
    text += """```json
        {
            "Effect": "Allow",
            "Action": [
                "cloudformation:DescribeStacks",
                "cloudformation:ValidateTemplate",
                "cloudformation:CreateStack",
                "dynamodb:DescribeTable",
                "dynamodb:CreateTable",
                "dynamodb:DeleteTable",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Query",
                "dynamodb:BatchWriteItem",
                "iam:CreateRole",
                "iam:GetRole",
                "iam:PutRolePolicy",
                "iam:DeleteRolePolicy",
                "iam:DeleteRole",
                "iam:PassRole",
                "cloudwatch:PutDashboard",
                "firehose:ListDeliveryStreams",
                "firehose:DeleteDeliveryStream",
                "firehose:DescribeDeliveryStream",
                "firehose:CreateDeliveryStream",
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "glue:GetTable",
                "glue:DeleteTable",
                "glue:GetPartitions",
                "glue:UpdateTable",
                "glue:CreateTable",
                "glue:GetDatabase"
            ],
            "Resource": [
                "*"
            ]
        },```\n"""
    text += "4. Now wait for a few minutes before executing this cell again!"
    return text

