import boto3
import logging
from time import strftime, gmtime


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def deploy_ml_gateway_pattern(
    sagemaker_endpoint_name: str, region: str, s3_bucket_name: str
) -> str:
    """
    Create an API Gateway HTTP endpoint that points to a Lambda function
    which points to a SageMaker endpoint.

    :param sagemaker_endpoint_name: str
    :param s3_bucket_name: str
    :return: str
    """

    cloudformation = boto3.client("cloudformation", region_name=region)
    timestamp: int = strftime("%d%H%M%S", gmtime())
    stack_name: str = f"ml-gateway-{timestamp}"
    lambda_name: str = f"serverless-artillery-{timestamp}-dev-loadGenerator"
    cloudformation.create_stack(
        StackName=stack_name,
        TemplateBody=ml_gateway_cf_body(region),
        Parameters=[
            {"ParameterKey": "SageMakerEndPointName", "ParameterValue": sagemaker_endpoint_name},
            {
                "ParameterKey": "LambdaName",
                "ParameterValue": f"invoke-sagemaker-endpoint-{timestamp}",
            },
            {"ParameterKey": "S3BucketName", "ParameterValue": s3_bucket_name},
        ],
        Capabilities=["CAPABILITY_IAM"],
    )
    waiter = cloudformation.get_waiter("stack_create_complete")
    logger.info("Creating ML Gateway...")
    waiter.wait(StackName=stack_name)
    logger.info("ML Gateway created!")
    response = cloudformation.describe_stacks(StackName=stack_name)
    api_gateway_endpoint_url = response["Stacks"][0]["Outputs"][0]["OutputValue"]
    return f"{api_gateway_endpoint_url}/TestStage/Model"


def ml_gateway_cf_body(region: str) -> str:
    """
    Return a JSON CloudFormation template represented as a string
    that will create an API Gateway HTTP endpoint with a Lambda
    function behind it which calls SageMaker Feature Store
    and a SageMaker Production Variant.
    """

    template_body = """
    {
        "AWSTemplateFormatVersion":"2010-09-09",
        "Description":"Call SageMaker Endpoint with API Gateway and Lambda",
        "Parameters":{
            "SageMakerEndPointName": {
                "Type" : "String",
                "Description" : "Name of your SageMaker Endpoint"
            },
            "LambdaName": {
                "Type" : "String",
                "Description" : "Name of your Lambda function"
            },
            "S3BucketName": {
                "Type" : "String",
                "Description" : "Name of your S3 bucket"
            }
        },
        "Resources":{
            "lambdafunctionRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": ["lambda.amazonaws.com"]
                                },
                                "Action": ["sts:AssumeRole"]
                            }
                        ]
                    },
                    "Path": "/"
                }
                },
            "lambdafunctionRolePolicy": {
                "Type": "AWS::IAM::Policy",
                "Properties": {
                    "PolicyName": "lambda_sm_Function_Policy",
                    "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Action": [
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                                "sagemaker:InvokeEndpoint",
                                "sagemaker:GetRecord",
                                "sagemaker:PutRecord"
                            ],
                            "Effect": "Allow",
                            "Resource": ["arn:aws:logs:*:*:*",
                            "arn:aws:sagemaker:*:*:*" ]
                        }
                    ]
                    },
                "Roles": [{ "Ref": "lambdafunctionRole"}]
                }
            },
            "invokeSMEndpoint": {
            "Type": "AWS::Lambda::Function",
            "Properties": {
                "Code": {
                  "S3Bucket": {"Ref": "S3BucketName"},
                  "S3Key": "function.zip"
                },
                "Handler": "lambda_function.lambda_handler",
                "FunctionName": {"Ref": "LambdaName"},
                "Layers": [
                    "arn:aws:lambda:us-west-1:770693421928:layer:Klayers-python38-pandas:37"
                ],
                "Runtime": "python3.8",
                "Timeout": 30,
                "Role": {"Fn::GetAtt": ["lambdafunctionRole", "Arn"]}
            }
        },
        "ModelAPI": {
            "Type": "AWS::ApiGateway::RestApi",
            "Properties": {
                "Name": "ModelAPI",
                "Description": "API fronting Lambda function calling SageMaker endpoint",
                "FailOnWarnings" : true
            }
            },
            "LambdaPermission": {
                "Type": "AWS::Lambda::Permission",
                "Properties": {
                "Action": "lambda:invokeFunction",
                "FunctionName": {"Fn::GetAtt": ["invokeSMEndpoint", "Arn"]},
                "Principal": "apigateway.amazonaws.com",
                "SourceArn": {"Fn::Join": ["",
                    ["arn:aws:execute-api:", {"Ref": "AWS::Region"}, ":", {"Ref": "AWS::AccountId"}, ":", {"Ref": "ModelAPI"}, "/*"]
                ]}
                }
            },
            "ModelApiStage": {
            "DependsOn" : ["ApiGatewayAccount"],
            "Type": "AWS::ApiGateway::Stage",
            "Properties": {
                "DeploymentId": {"Ref": "ApiDeployment"},
                "MethodSettings": [{
                "DataTraceEnabled": true,
                "HttpMethod": "*",
                "LoggingLevel": "INFO",
                "ResourcePath": "/*"
                }],
                "RestApiId": {"Ref": "ModelAPI"},
                "StageName": "LATEST"
            }
            },
            "ApiGatewayCloudWatchLogsRole": {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": { "Service": ["apigateway.amazonaws.com"] },
                    "Action": ["sts:AssumeRole"]
                }]
                },
                "Policies": [{
                "PolicyName": "ApiGatewayLogsPolicy",
                "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:DescribeLogGroups",
                        "logs:DescribeLogStreams",
                        "logs:PutLogEvents",
                        "logs:GetLogEvents",
                        "logs:FilterLogEvents"
                    ],
                    "Resource": "*"
                    }]
                }
                }]
            }
            },
            "ApiGatewayAccount": {
            "Type" : "AWS::ApiGateway::Account",
            "Properties" : {
                "CloudWatchRoleArn" : {"Fn::GetAtt" : ["ApiGatewayCloudWatchLogsRole", "Arn"] }
            }
            },

            "ApiDeployment": {
            "Type": "AWS::ApiGateway::Deployment",
            "DependsOn": ["ModelRequest"],
            "Properties": {
                "RestApiId": {"Ref": "ModelAPI"},
                "StageName": "TestStage"
            }
            },
            "Model": {
            "Type": "AWS::ApiGateway::Resource",
            "Properties": {
            "RestApiId": {"Ref": "ModelAPI"},
            "ParentId": {"Fn::GetAtt": ["ModelAPI", "RootResourceId"]},
            "PathPart": "Model"
            }
        },
        "ModelRequest": {
            "DependsOn": "LambdaPermission",
            "Type": "AWS::ApiGateway::Method",
            "Properties": {
            "AuthorizationType": "NONE",
            "HttpMethod": "POST",
            "Integration": {
                "Type": "AWS_PROXY",
                "IntegrationHttpMethod": "POST",
                "Uri": {"Fn::Join" : ["",
                ["arn:aws:apigateway:", {"Ref": "AWS::Region"}, ":lambda:path/2015-03-31/functions/", {"Fn::GetAtt": ["invokeSMEndpoint", "Arn"]}, "/invocations"]
                ]}
            },
            "MethodResponses": [{
                "StatusCode": "200"
                }],
            "ResourceId": {"Ref": "Model"},
            "RestApiId": {"Ref": "ModelAPI"}
            }
        }
    },
    "Outputs":{
        "APIGatewayEndPointURL":{
        "Value": {"Fn::Join": ["", ["https://", {"Ref": "ModelAPI"}, ".execute-api.", {"Ref": "AWS::Region"}, ".amazonaws.com"]]}
        }
    }
}
    """

    return template_body.replace("lambda:us-west-1", f"lambda:{region}")
