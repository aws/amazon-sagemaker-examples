import argparse
import logging
import os
import json
import pathlib
import requests
import tempfile

import boto3

import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_package_group_name', type=str, default="")

    args, _ = parser.parse_known_args()
    
    model_package_group_name = args.model_package_group_name

    is_approved = True
    reasons = []
    with open('/opt/ml/processing/input/checks.json') as checks:
        checks = json.load(checks)
        print(f"checks: {checks}")
        for key, value in checks.items():
            print(f"{key} is {value}")
            
            if not value:
                is_approved = False
                reasons.append(key)
            
    client = boto3.client(service_name="sagemaker", region_name="us-east-1")

    model_package_arn = client.list_model_packages(ModelPackageGroupName=model_package_group_name)[
    "ModelPackageSummaryList"
    ][0]["ModelPackageArn"]
    

    if is_approved:
        approval_description = "Model package meets organisational guidelines"
    else:
        approval_description = "Model values for the following checks does not meet threshold: "
        for reason in reasons:
            approval_description+= f"{reason} "
        
    model_package_update_input_dict = {
        "ModelPackageArn" : model_package_arn,
        "ApprovalDescription": approval_description,
        "ModelApprovalStatus" : "Approved" if is_approved else "Rejected"
    }
    
    model_package_update_response = client.update_model_package(**model_package_update_input_dict)
    
    assert is_approved
    print("Finished updating model status!")