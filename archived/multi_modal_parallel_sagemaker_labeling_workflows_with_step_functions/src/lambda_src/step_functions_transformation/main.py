"""
Step functions transformation validates and transforms
input batch for later use.

It is currently a no-op.
"""

import boto3
from shared import db, log
from shared.constants import BatchCurrentStep, BatchMetadataType, BatchStatus


def lambda_handler(event, context):
    """
    Lambda function that transforms input data and stores inital DB entry

    Parameters
    ----------
    event: dict, required
    context: object, required Lambda Context runtime methods and attributes
    Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    Lambda Output Format: dict
    """

    log.log_request_and_context(event, context)
    labeling_jobs = event["labelingJobs"]
    batch_id = event["batchId"]

    error_message = ""

    """
    Example database entry input for batch
    {
      "BatchCurrentStep": "INPUT",
      "BatchId": "notebook-test-08f874a7",
      "BatchMetadataType": "INPUT",
      "BatchStatus": "INTERNAL_ERROR",
      "LabelingJobs": [
        {
          "inputConfig": {
            "inputManifestS3Uri": "s3://smgt-qa-batch-input-468814823616-us-east-1/two-frame-manifest.manifest"
          },
          "jobLevel": 1,
          "jobModality": "PointCloudObjectDetectionAudit",
          "jobName": "notebook-test-08f874a7-first-level",
          "jobType": "BATCH",
          "labelCategoryConfigS3Uri": "s3://smgt-qa-batch-input-468814823616-us-east-1/first-level-label-category-file.json",
          "maxConcurrentTaskCount": 1,
          "taskAvailabilityLifetimeInSeconds": 864000,
          "taskTimeLimitInSeconds": 604800,
          "workteamArn": "arn:aws:sagemaker:us-east-1:468814823616:workteam/private-crowd/first-level"
        },
        {
          "inputConfig": {
            "chainFromJobName": "notebook-test-08f874a7-first-level"
          },
          "jobLevel": 2,
          "jobModality": "PointCloudObjectDetectionAudit",
          "jobName": "notebook-test-08f874a7-second-level",
          "jobType": "BATCH",
          "maxConcurrentTaskCount": 1,
          "taskAvailabilityLifetimeInSeconds": 864000,
          "taskTimeLimitInSeconds": 604800,
          "workteamArn": "arn:aws:sagemaker:us-east-1:468814823616:workteam/private-crowd/first-level"
        }
      ]
    }
    """
    db.insert_transformed_input_batch_metadata(
        batch_id=batch_id,
        batch_current_step=BatchCurrentStep.INPUT,
        batch_status=BatchStatus.IN_PROGRESS,
        batch_metadata_type=BatchMetadataType.INPUT,
        error_message=error_message,
        labeling_jobs=labeling_jobs,
    )

    return {
        "batch_id": batch_id,
    }
