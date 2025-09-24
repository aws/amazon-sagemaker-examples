import json
import os
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

current_folder = get_current_folder(globals())

SOLUTION_PREFIX = "sagemaker-soln-documents-"

DATASETS_S3_PREFIX = "datasets"
OUTPUTS_S3_PREFIX = "outputs"

SOURCE_S3_PREFIX = "0.2.0/Document-understanding/3.0.1"
SOURCE_S3_BUCKET = "sagemaker-solutions-prod-us-east-2"
SOURCE_S3_PATH = f"s3://{SOURCE_S3_BUCKET}/{SOURCE_S3_PREFIX}"

TRAINING_INSTANCE_TYPE = "ml.p3.2xlarge"
HOSTING_INSTANCE_TYPE = "ml.g4dn.2xlarge"

TAG_KEY = "sagemaker-soln"
