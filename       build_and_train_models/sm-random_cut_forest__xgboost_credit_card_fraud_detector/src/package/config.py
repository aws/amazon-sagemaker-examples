import json
import sagemaker
import boto3
from pathlib import Path

from package import utils

SOLUTION_PREFIX = "sagemaker-soln-fdml-js-"
AWS_ACCOUNT_ID = aws_account = boto3.client('sts').get_caller_identity().get('Account')
AWS_REGION = boto3.session.Session().region_name
SAGEMAKER_IAM_ROLE = sagemaker.get_execution_role()
MODEL_DATA_S3_BUCKET = sagemaker.session.Session(boto3.session.Session()).default_bucket()
SOLUTIONS_S3_BUCKET = "sagemaker-solutions-prod"
SOLUTION_NAME = "Fraud-detection-using-machine-learning/3.4.1"
