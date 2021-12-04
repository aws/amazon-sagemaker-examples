
"""
This Lambda function creates an Endpoint Configuration and deploys a model to an Endpoint. 
The Lambda will update the endpoint if it exists already with the latest approved version of the model from the model registry.
The name of the model to deploy is provided via the `event` argument
"""

import json
import boto3
import traceback
import logging
from botocore.exceptions import ClientError
sagemaker_boto_client = boto3.client("sagemaker")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def describe_model_package(model_package_arn):
    """
    Describe the model version details
    """
    try:
        model_package = sagemaker_boto_client.describe_model_package(
            ModelPackageName=model_package_arn
        )

        LOGGER.info("{}".format(model_package))

        if len(model_package) == 0:
            error_message = ("No ModelPackage found for: {}".format(model_package_arn))
            LOGGER.error("{}".format(error_message))

            raise Exception(error_message)

        return model_package
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        LOGGER.error("{}".format(stacktrace))

        raise Exception(error_message)
    
def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sagemaker_boto_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            LOGGER.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sagemaker_boto_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = ("No approved ModelPackage found for ModelPackageGroup: {}".format(model_package_group_name))
            LOGGER.error("{}".format(error_message))

            raise Exception(error_message)

        model_package = approved_packages[0]
        LOGGER.info("Identified the latest approved model package: {}".format(model_package))

        return model_package
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        LOGGER.error("{}".format(stacktrace))

        raise Exception(error_message)


def lambda_handler(event, context):

    # Parameters passed in the event 
    deploy_instance = event["deploy_model_instance_type"]
    deploy_instance_count = event["deploy_model_instance_count"]
    endpoint_name = event["endpoint_name"]
    model_package_group_name = event["model_package_group_name"]
    role = event["role"]
    
    #Get latest approved version of model from model registry
    model_package_approved = get_approved_package(model_package_group_name)
    model_package_version = model_package_approved["ModelPackageVersion"]
    model_package = describe_model_package(model_package_approved["ModelPackageArn"])

    model_name = f'{endpoint_name}-model-v{model_package_version}'
    ep_config_name = f'{endpoint_name}-epc-v{model_package_version}'
    
    # Create a model using the new approved version from registry
    new_model = sagemaker_boto_client.create_model(ModelName=model_name,
                                    PrimaryContainer={
                                                       'Image': model_package["InferenceSpecification"]["Containers"][0]['Image'],
                                                       'Environment': model_package["InferenceSpecification"]["Containers"][0]['Environment']
                                                     },
                                    ExecutionRoleArn=role)
    
    # Create a new Endpoint Config
    create_endpoint_config_api_response = sagemaker_boto_client.create_endpoint_config(
                                EndpointConfigName=ep_config_name,
                                ProductionVariants=[
                                    {
                                        'VariantName': f'AllTraffic-v{model_package_version}',
                                        'ModelName': model_name,
                                        'InitialInstanceCount': deploy_instance_count,
                                        'InstanceType': deploy_instance
                                    },
                                ]
                           )

    try:
        sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
        
        # Update the existing Endpoint
        create_endpoint_api_response = sagemaker_boto_client.update_endpoint(
                            EndpointName=endpoint_name,
                            EndpointConfigName=ep_config_name
                        )
        
        return {
                    "statusCode": 200,
                    "body": json.dumps(f"Endpoint {endpoint_name} Created!"),
                    "endpoint_created": "Y"
                }
    except ClientError as error:
        if "Could not find endpoint" in error.response['Error']['Message']:             
            try:
                create_endpoint_response = sagemaker_boto_client.create_endpoint(
                                                EndpointName=endpoint_name, 
                                                EndpointConfigName=ep_config_name
                                            )
                return {
                    "statusCode": 200,
                    "body": json.dumps(f"Endpoint {endpoint_name} Created!"),
                    "endpoint_created": "Y"
                }
            except ClientError as error:
                print(error.response['Error']['Message'])
                create_config('N')
                error_message = error.response["Error"]["Message"]
                LOGGER.error("{}".format(stacktrace))
                return {
                    "statusCode": 500,
                    "body": json.dumps("Endpoint creation failed. Check logs!"),
                    "endpoint_created": "N"
                }
        else:
            print(error.response['Error']['Message'])
            create_config('N')
            error_message = error.response["Error"]["Message"]
            LOGGER.error("{}".format(stacktrace))
            return {
                    "statusCode": 500,
                    "body": json.dumps("Endpoint update failed. Check logs!"),
                    "endpoint_created": "Y"
                }
            
