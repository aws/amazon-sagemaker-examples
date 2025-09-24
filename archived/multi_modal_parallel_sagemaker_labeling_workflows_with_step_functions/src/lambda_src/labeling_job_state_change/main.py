"""
Listens for GroundTruth job state changes and updates DB.

This is useful to track job state changes that happen due to external
forces, like if a job suddenly fails due to invalid input we didn't
validate fully, this will still write the job state back to the DB.
"""
import json
import os
from re import search

import boto3
from shared import db, log
from shared.constants import BatchMetadataTableAttributes, BatchMetadataType, BatchStatus
from shared.db import get_batch_metadata_by_labeling_job_name

sfn_client = boto3.client("stepfunctions")
s3_client = boto3.client("s3")
s3 = boto3.resource("s3")

glue_bucket_name = bucket_path = os.environ.get("GLUE_WM_BUCKET_NAME", None)


def extract_labeling_job(job_arn):
    return search(r":labeling-job/([^/]+)", job_arn).group(1)


def mark_job_batch_complete(job_level_batch):
    """Mark the job batch as complete, trigger sideeffects if parent is complete"""
    log.logger.info(f"Signaling batch_meta to resume execution {job_level_batch}")

    batch_id = job_level_batch[BatchMetadataTableAttributes.BATCH_ID]
    if (
        job_level_batch[BatchMetadataTableAttributes.BATCH_STATUS]
        != BatchStatus.WAIT_FOR_SMGT_RESPONSE
    ):
        log.logger.error("Invalid batch status, ignoring request")
        return
    db.update_batch_status(batch_id, BatchStatus.COMPLETE)

    # Copy worker metrics from groundtruth bucket to raw_worker_metrics
    # folder in the glue bucket
    jobOutputLocation = job_level_batch["JobOutputLocation"]
    bucketName = jobOutputLocation.split("/")[2]
    groundtruth_bucket = s3.Bucket(bucketName)

    for obj in groundtruth_bucket.objects.filter(
        Prefix="/".join(jobOutputLocation.split("/")[3:6])
    ):
        if obj.key.endswith(".json") and "worker-response" in obj.key:
            if not obj.key.endswith(".jpg.json"):
                new_key = f"raw_worker_metrics/{'/'.join(obj.key.split('/')[1:])}"
                s3_client.copy_object(
                    Bucket=glue_bucket_name, CopySource=f"{bucketName}/{obj.key}", Key=new_key
                )

    parent_batch_id = job_level_batch[BatchMetadataTableAttributes.PARENT_BATCH_ID]
    if not db.update_batch_child_count(parent_batch_id, 1):
        # Incomplete, return
        return

    parent_batch = db.get_batch_metadata(parent_batch_id)
    try:
        task_token = parent_batch[BatchMetadataTableAttributes.STATE_TOKEN]
    except KeyError as err:
        raise Exception(f"missing state token on batch: {parent_batch_id}") from err

    # Send status token to step functions
    response = sfn_client.send_task_success(
        taskToken=task_token, output=json.dumps({"batch_id": parent_batch_id})
    )
    log.logger.info(f"Response for Step function token {task_token}: {response}")


def process_new_status(job_arn, job_status, invoked_function_arn):
    """Runs any database mutations needed to update the job based on the new status.

    Parameters
    ----------
    job_arn: string, required arn of the ground truth job the status applies to
    job_status: string, required the new status of the job

    Returns
    ------
    None
    """
    log.logger.info(f"Processing job arn '{job_arn}' with status '{job_status}'")

    labeling_job_name = extract_labeling_job(job_arn)
    batch_metadata = db.get_batch_metadata_by_labeling_job_name(
        labeling_job_name, BatchMetadataType.JOB_LEVEL
    )

    if len(batch_metadata) > 0:
        mark_job_batch_complete(batch_metadata[0])


def lambda_handler(event, context):
    """Lambda function that responds to changes in labeling job status, updating
    the corresponding dynamo db tables and publishing to sns after a job is cancelled.

    Parameters
    ----------
    event: dict, required API gateway request with an input SQS arn, output SQS arn
    context: object, required Lambda Context runtime methods and attributes
    Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    Lambda Output Format: dict
    Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    log.log_request_and_context(event, context)

    job_status = event["status"]
    job_arns = event["job_arns"]
    if len(job_arns) != 1:
        raise ValueError("incorrect number of job arns in event: ", job_arns)

    job_arn = job_arns[0]

    # We received a new status for the job_arn.
    process_new_status(job_arn, job_status, context.invoked_function_arn)

    return "success"
