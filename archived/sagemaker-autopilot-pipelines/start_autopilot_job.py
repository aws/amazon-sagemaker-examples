import sys
from pip._internal import main

# Upgrading boto3 to the newest release to be able to use the latest SageMaker features
main(
    [
        "install",
        "-I",
        "-q",
        "boto3",
        "--target",
        "/tmp/",
        "--no-cache-dir",
        "--disable-pip-version-check",
    ]
)
sys.path.insert(0, "/tmp/")
import boto3

sagemaker_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    sagemaker_client.create_auto_ml_job(
        AutoMLJobName=event["AutopilotJobName"],
        InputDataConfig=[
            {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": event["TrainValDatasetS3Path"],
                    }
                },
                "TargetAttributeName": event["TargetAttributeName"],
            }
        ],
        OutputDataConfig={"S3OutputPath": event["TrainingOutputS3Path"]},
        ProblemType=event["ProblemType"],
        AutoMLJobObjective={"MetricName": event["AutopilotObjectiveMetricName"]},
        AutoMLJobConfig={
            "CompletionCriteria": {
                "MaxCandidates": event["MaxCandidates"],
                "MaxRuntimePerTrainingJobInSeconds": event[
                    "MaxRuntimePerTrainingJobInSeconds"
                ],
                "MaxAutoMLJobRuntimeInSeconds": event["MaxAutoMLJobRuntimeInSeconds"],
            },
            "Mode": event["AutopilotMode"],
        },
        RoleArn=event["AutopilotExecutionRoleArn"],
    )
