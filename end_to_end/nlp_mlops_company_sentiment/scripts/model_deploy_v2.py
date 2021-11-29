#!/usr/bin/env python
import argparse
import subprocess
import sys
import os
import json
import boto3
import botocore
import sagemaker
from botocore.exceptions import ClientError
from sagemaker import ModelPackage
from pathlib import Path
import logging
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# parameters sent by the client are passed as command-line arguments to the script.
parser.add_argument("--initial-instance-count", type=int, default=1)
parser.add_argument("--endpoint-instance-type", type=str, default="ml.m5.xlarge")
parser.add_argument("--endpoint-name", type=str)
parser.add_argument("--model-package-group-name", type=str)
parser.add_argument("--role", type=str)
parser.add_argument("--region", type=str)

args, _ = parser.parse_known_args()

sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=args.region))
sagemaker_boto_client = boto3.client("sagemaker", region_name=args.region)

def create_config(flag):
    model_created = { 'model_created': flag }
    out_path = Path(f'/opt/ml/processing/output/success.json')
    out_str = json.dumps(model_created, indent=4)
    out_path.write_text(out_str, encoding='utf-8')

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

if __name__=='__main__':
    
    try:
        sagemaker_boto_client.describe_endpoint(EndpointName=args.endpoint_name)
        print(f'Endpoint {args.endpoint_name} already exists...updating with new Model Version')
        
        model_package_approved = get_approved_package(args.model_package_group_name)
        model_package_version = model_package_approved["ModelPackageVersion"]
        model_package = describe_model_package(model_package_approved["ModelPackageArn"])
                
        model_name = f'{args.endpoint_name}-model-v{model_package_version}'
        ep_config_name = f'{args.endpoint_name}-epc-v{model_package_version}'

        # Create a model
        new_model = sagemaker_boto_client.create_model(ModelName=model_name,
                                        PrimaryContainer={
                                                           'Image': model_package["InferenceSpecification"]["Containers"][0]['Image'],
                                                           'Environment': model_package["InferenceSpecification"]["Containers"][0]['Environment']
                                                         },
                                        ExecutionRoleArn=args.role)
        # Create a new Endpoint Config
        create_endpoint_config_api_response = sagemaker_boto_client.create_endpoint_config(
                                    EndpointConfigName=ep_config_name,
                                    ProductionVariants=[
                                        {
                                            'VariantName': f'AllTraffic-v{model_package_version}',
                                            'ModelName': model_name,
                                            'InitialInstanceCount': args.initial_instance_count,
                                            'InstanceType': args.endpoint_instance_type
                                        },
                                    ]
                               )
        # Update the existing Endpoint
        create_endpoint_api_response = sagemaker_boto_client.update_endpoint(
                            EndpointName=args.endpoint_name,
                            EndpointConfigName=ep_config_name
                        )

        create_config('Y')
    except ClientError as error: 
        # endpoint does not exist
        if "Could not find endpoint" in error.response['Error']['Message']: 
            model_package_approved = get_approved_package(args.model_package_group_name)
            model_package_arn = model_package_approved["ModelPackageArn"]

            model = ModelPackage(role=args.role, 
                                 model_package_arn=model_package_arn, 
                                 sagemaker_session=sagemaker_session)
            try:
                model.deploy(initial_instance_count=args.initial_instance_count, 
                             instance_type=args.endpoint_instance_type,
                             endpoint_name=args.endpoint_name)
                create_config('Y')
            except ClientError as error:
                print(error.response['Error']['Message'])
                create_config('N')
                error_message = error.response["Error"]["Message"]
                LOGGER.error("{}".format(stacktrace))
                raise Exception(error_message)
        else:
            print(error.response['Error']['Message'])
            create_config('N')
            error_message = error.response["Error"]["Message"]
            LOGGER.error("{}".format(stacktrace))
            raise Exception(error_message)