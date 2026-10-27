import argparse
import boto3
import json
import os
import pandas as pd
import random
import string
import time
from sklearn.metrics import f1_score, precision_score, recall_score
from urllib.parse import urlparse

RANDOM_SUFFIX = "".join(random.choices(string.ascii_lowercase, k=8))

parser = argparse.ArgumentParser()
parser.add_argument("--autopilot-job-name", type=str)
parser.add_argument("--aws-region", type=str)
parser.add_argument("--x-test-s3-path", type=str)
parser.add_argument("--y-test-file-name", type=str)
parser.add_argument("--batch-transform-output-s3-path", type=str)
parser.add_argument("--instance-type", type=str)
parser.add_argument("--instance-count", type=int)
parser.add_argument("--local-base-path", type=str)
parser.add_argument("--sagemaker-execution-role-arn", type=str)
args = parser.parse_args()

boto_session = boto3.session.Session(region_name=args.aws_region)
s3_client = boto_session.client("s3")
sagemaker_client = boto_session.client("sagemaker")

# Create model
model_name = args.autopilot_job_name + RANDOM_SUFFIX
response = sagemaker_client.create_model(
    ModelName=model_name,
    Containers=sagemaker_client.describe_auto_ml_job(
        AutoMLJobName=args.autopilot_job_name
    )["BestCandidate"]["InferenceContainers"],
    ExecutionRoleArn=args.sagemaker_execution_role_arn,
)

# Create batch transform job
batch_transform_job_name = args.autopilot_job_name + RANDOM_SUFFIX
response = sagemaker_client.create_transform_job(
    TransformJobName=batch_transform_job_name,
    ModelName=model_name,
    TransformInput={
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": args.x_test_s3_path,
            }
        },
        "ContentType": "text/csv",
        "SplitType": "Line",
    },
    TransformOutput={
        "S3OutputPath": args.batch_transform_output_s3_path,
        "AssembleWith": "Line",
    },
    TransformResources={
        "InstanceType": args.instance_type,
        "InstanceCount": args.instance_count,
    },
)

# Wait for the batch transform job to finish
while (
    sagemaker_client.describe_transform_job(TransformJobName=batch_transform_job_name)[
        "TransformJobStatus"
    ]
    == "InProgress"
):
    time.sleep(10)

# Download batch transform results
x_test_file_name = args.x_test_s3_path.split("/")[-1]
predictions_s3_path = os.path.join(
    args.batch_transform_output_s3_path, x_test_file_name + ".out"
)
o = urlparse(predictions_s3_path)
s3_client.download_file(
    Bucket=o.netloc, Key=o.path.strip("/"), Filename="predictions.csv"
)

# Create best model evaluation report
y_pred = pd.read_csv("predictions.csv", header=0).iloc[:, 0]
y_true = pd.read_csv(
    os.path.join(args.local_base_path, "data", args.y_test_file_name), header=1
)
evaluation_report = {
    "multiclass_classification_metrics": {
        "weighted_f1": {
            "value": f1_score(y_pred, y_true, average="weighted"),
            "standard_deviation": "NaN",
        },
        "weighted_precision": {
            "value": precision_score(y_pred, y_true, average="weighted"),
            "standard_deviation": "NaN",
        },
        "weighted_recall": {
            "value": recall_score(y_pred, y_true, average="weighted"),
            "standard_deviation": "NaN",
        },
    },
}
evaluation_report_path = os.path.join(
    args.local_base_path, "evaluation_report", "evaluation_report.json"
)
os.makedirs(os.path.dirname(evaluation_report_path), exist_ok=True)
with open(evaluation_report_path, "w") as f:
    f.write(json.dumps(evaluation_report))
