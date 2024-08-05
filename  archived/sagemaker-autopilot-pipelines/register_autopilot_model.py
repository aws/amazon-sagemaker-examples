import boto3
import os
from botocore.exceptions import ClientError
from urllib.parse import urlparse

s3_client = boto3.client("s3")
sagemaker_client = boto3.client("sagemaker")


def get_explainability_report_json_s3_path(s3_path):
    o = urlparse(s3_path)
    bucket_name = o.netloc
    s3_prefix = o.path.strip("/")
    paginator = s3_client.get_paginator("list_objects_v2")
    response = paginator.paginate(
        Bucket=bucket_name, Prefix=s3_prefix, PaginationConfig={"PageSize": 1}
    )
    for page in response:
        files = page.get("Contents")
        for file in files:
            if "analysis.json" in file["Key"]:
                return os.path.join("s3://", bucket_name, file["Key"])


def lambda_handler(event, context):
    # Get the explainability results from the Autopilot job
    autopilot_job = sagemaker_client.describe_auto_ml_job(
        AutoMLJobName=event["AutopilotJobName"]
    )
    explainability_report_s3_path = autopilot_job["BestCandidate"][
        "CandidateProperties"
    ]["CandidateArtifactLocations"]["Explainability"]
    autopilot_job["BestCandidate"]["InferenceContainers"][0].pop("Environment")
    sagemaker_client.create_model_package(
        ModelPackageName=event["ModelPackageName"],
        InferenceSpecification={
            "Containers": autopilot_job["BestCandidate"]["InferenceContainers"],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedTransformInstanceTypes": [event["InstanceType"]],
            "SupportedRealtimeInferenceInstanceTypes": [event["InstanceType"]],
        },
        ModelApprovalStatus=event["ModelApprovalStatus"],
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": ".json",
                    "S3Uri": os.path.join(
                        event["EvaluationReportS3Path"], "evaluation_report.json"
                    ),
                },
            },
            "Explainability": {
                "Report": {
                    "ContentType": ".json",
                    "S3Uri": get_explainability_report_json_s3_path(
                        explainability_report_s3_path
                    ),
                }
            },
        },
    )
