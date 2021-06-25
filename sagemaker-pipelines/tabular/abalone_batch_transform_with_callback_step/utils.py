import argparse
import logging
import os
import tarfile
import zipfile
import json
import subprocess

if os.getenv("AWS_EXECUTION_ENV") == "AWS_Lambda_python3.8":
    import sys

    sys.path.insert(0, "/opt/latestboto3/")
import boto3


logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def handler(event, context):
    """
    Gets the latest model from the model registry and returns the s3 location of the model data
    as well as the ECR location of the container to be used for inference.
    """
    for record in event["Records"]:
        # Retrieve token from queue message
        payload = json.loads(record["body"])
        token = payload["token"]

        # Retrieve leatest approved model
        model_package_group_name = os.getenv("ModelPackageGroupName")
        pck = get_approved_package(model_package_group_name)
        model_description = sm_client.describe_model_package(
            ModelPackageName=pck["ModelPackageArn"]
        )
        model_url = model_description["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        image_uri = model_description["InferenceSpecification"]["Containers"][0]["Image"]

        # Call SageMaker to complete the step
        sm_client.send_pipeline_execution_step_success(
            CallbackToken=token,
            OutputParameters=[
                {"Name": "ModelUrl", "Value": model_url},
                {"Name": "ImageUri", "Value": image_uri},
            ],
        )


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return approved_packages[0]
        # return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
