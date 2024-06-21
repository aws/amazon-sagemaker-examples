import json
import os
import boto3
import sagemaker
from pathlib import Path


def get_current_folder(global_variables):
    # if calling from a file
    if "__file__" in global_variables:
        current_file = Path(global_variables["__file__"])
        current_folder = current_file.parent.resolve()
    # if calling from a notebook
    else:
        current_folder = Path(os.getcwd())
    return current_folder

region = boto3.session.Session().region_name
aws_account = boto3.client('sts').get_caller_identity().get('Account')
default_bucket = sagemaker.session.Session(boto3.session.Session()).default_bucket()
default_role = sagemaker.get_execution_role()

cfn_stack_outputs = {}
current_folder = get_current_folder(globals())

boto3_session = boto3.session.Session()
region_name = boto3_session.region_name

solution_name = "0.2.0/Fraud-detection-in-financial-networks/3.0.1"
solution_upstream_bucket = "sagemaker-solutions-prod-us-west-2"


solution_prefix = "sagemaker-soln-fdfn-js-"
solution_bucket = default_bucket

s3_data_prefix = "raw-data"
s3_processing_output = "preprocessed-data"
s3_train_output = "training-output"

role = default_role