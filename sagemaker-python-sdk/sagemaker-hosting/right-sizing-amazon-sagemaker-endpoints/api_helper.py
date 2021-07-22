import json
import time
import boto3
import sagemaker
from zipfile import ZipFile

iam_client = boto3.client("iam")
lambda_client = boto3.client("lambda")
api_client = boto3.client("apigateway")
iam_resource = boto3.resource("iam")
sm_client = boto3.client("sagemaker")


def create_infra(project_name, account_id, region):
    """
    Creates AWS Lambda and API Gateway for load testing.

    This function is called by the notebook to create an AWS Lambda function
    and an API gateway endpoint that we use to perform the load testing.

    Inputs:
    project_name - unique identifier that's tagged to all resources
    account_id - notebook users's AWS account ID
    region - AWS region to deploy the resources in.

    Output:
    API Gateway endpoint URL.
    """
    ## SETUP THE ROLES AND POLICIES

    # Create role that can be assumed by Lambda
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    role = iam_client.create_role(
        RoleName=f"LambdaRoleToInvokeSageMakerEndpoint{project_name}",
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description="Role for Lambda function to invoke SageMaker endpoints",
    )

    # Create policy for the role, allowing Lambda to invoke SageMaker endpoint
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "sagemaker:InvokeEndpoint",
                ],
                "Resource": ["arn:aws:logs:*:*:*", "arn:aws:sagemaker:*:*:*"],
                "Effect": "Allow",
            }
        ],
    }

    policy = iam_client.create_policy(
        PolicyName=f"LambdaSageMakerAccessPolicy{project_name}",
        PolicyDocument=json.dumps(policy_doc),
    )

    # Attach policy to role
    response = iam_resource.Role(role["Role"]["RoleName"]).attach_policy(
        PolicyArn=policy["Policy"]["Arn"]
    )

    LambdaRoleArn = role["Role"]["Arn"]

    time.sleep(10)

    ## SETUP AND DEPLOY LAMBDA FUNCTION

    # Zip the request prediction code
    with ZipFile("lambda_function.zip", "w") as zip:
        zip.write("lambda_index.py")

    # Create the lambda
    response = lambda_client.create_function(
        FunctionName=f"request-predictions-{project_name}",
        Runtime="python3.8",
        Role=LambdaRoleArn,
        Handler="lambda_index.lambda_handler",
        Code={"ZipFile": open("./lambda_function.zip", "rb").read()},
        Description="Function to invoke endpoint and return predictions",
        Timeout=60,
        MemorySize=128,
    )

    lambda_arn = response["FunctionArn"]
    lambda_name = response["FunctionName"]

    # Create rest api
    rest_api = api_client.create_rest_api(name=f"ImageClassifier-{project_name}")
    rest_api_id = rest_api["id"]

    # Get the rest api's root id
    root_resource_id = api_client.get_resources(restApiId=rest_api_id)["items"][0]["id"]

    # Create an api resource
    api_resource = api_client.create_resource(
        restApiId=rest_api_id, parentId=root_resource_id, pathPart="ImageClassifier"
    )

    api_resource_id = api_resource["id"]

    # Add a post method to the rest api resource
    api_method = api_client.put_method(
        restApiId=rest_api_id,
        resourceId=api_resource_id,
        httpMethod="POST",
        authorizationType="NONE",
    )

    lambda_uri = f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"

    # Add integrations for mapping SageMaker and Lambda HTTP response codes to API gateway HTTP response codes
    put_integration = api_client.put_integration(
        restApiId=rest_api_id,
        resourceId=api_resource_id,
        httpMethod="POST",
        type="AWS",
        integrationHttpMethod="POST",
        uri=lambda_uri,
    )

    put_method_res = api_client.put_method_response(
        restApiId=rest_api_id,
        resourceId=api_resource_id,
        httpMethod="POST",
        statusCode="200",
        responseModels={"application/json": "Empty"},
    )

    put_integration_response = api_client.put_integration_response(
        restApiId=rest_api_id,
        resourceId=api_resource_id,
        httpMethod="POST",
        statusCode="200",
        responseTemplates={"application/json": ""},
    )

    put_integration_response = api_client.put_integration_response(
        restApiId=rest_api_id,
        resourceId=api_resource_id,
        httpMethod="POST",
        statusCode="400",
        selectionPattern="Invalid*",
        responseTemplates={"application/json": ""},
    )

    put_integration_response = api_client.put_integration_response(
        restApiId=rest_api_id,
        resourceId=api_resource_id,
        httpMethod="POST",
        statusCode="500",
        selectionPattern="Internal*",
        responseTemplates={"application/json": ""},
    )

    # Deploy to a stage
    deployment = api_client.create_deployment(
        restApiId=rest_api_id,
        stageName="dev",
    )

    # API gateway needs permissions to invoke the lambda function
    lambda_api_response = lambda_client.add_permission(
        FunctionName=lambda_name,
        StatementId="api-gateway-invoke",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{rest_api_id}/*/POST/ImageClassifier",
    )

    api_url = (
        f"https://{rest_api_id}.execute-api.{region}.amazonaws.com/dev/ImageClassifier"
    )

    print("API GATEWAY URL: url = " + f"{api_url}")

    return api_url


def delete_infra(project_name, account_id, rest_api_id, models_list):
    """
    Deletes infrastructure created for load testing
    """

    try:
        iam_client.detach_role_policy(
            RoleName=f"LambdaRoleToInvokeSageMakerEndpoint{project_name}",
            PolicyArn=f"arn:aws:iam::{account_id}:policy/LambdaSageMakerAccessPolicy{project_name}",
        )

        iam_client.delete_policy(
            PolicyArn=f"arn:aws:iam::{account_id}:policy/LambdaSageMakerAccessPolicy{project_name}"
        )
        print("IAM Policy deleted.")

        iam_client.delete_role(
            RoleName=f"LambdaRoleToInvokeSageMakerEndpoint{project_name}"
        )
        print("IAM Role deleted.")

        lambda_client.delete_function(
            FunctionName=f"request-predictions-{project_name}"
        )
        print(f"Lambda Function request-predictions-{project_name} deleted.")

        api_client.delete_rest_api(restApiId=rest_api_id)
        print(f"Rest API with id {rest_api_id} deleted.")

        for model_name in models_list:
            sm_client.delete_model(ModelName=model_name)
            print(f"Model {model_name} deleted.")

        print("All resources removed.")

    except Exception as e:
        print(f"Error deleting resources: {e}")
        print("Please delete the resources manually to avoid costs.")
