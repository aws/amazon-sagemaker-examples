"""
Handles error situations where one of the previous states failed.

Do handle the failure we:
    1) Mark all batches/frames in DB as failed.
    2) Publish job failed SNS.
"""

import json
import os

import boto3
import botocore
from shared import db
from shared.log import log_request_and_context, logger


def lambda_handler(event, context):
    """Lambda function that copies any worker logs to s3 and publishes batch finish to SNS.

    Parameters
    ----------
    event: dict, required
    context: object, required Lambda Context runtime methods and attributes
    Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    Lambda Output Format: dict
    """
    log_request_and_context(event, context)

    execution_id = event["execution_id"]
    request = event["input"]

    validation_output = request.get("transformation_step_output")
    if validation_output is None:
        raise Exception("no batch id stored with validation output, can't write sns")
    parent_batch_id = validation_output["batch_id"]

    error_info = request["error-info"]
    error_type = "Unknown"
    error_message = ""
    try:
        # If cause is json parsable, get more specific error details (like from python exception).
        # This avoids sending stack traces to the SNS receiver.
        cause = json.loads(error_info["Cause"])
        error_type = cause["errorType"]
        error_message = cause["errorMessage"]
    except (ValueError, KeyError):
        # Error message isn't json parseable, default to just put the whole "Cause" string.
        error_type = error_info["Error"]
        error_message = error_info["Cause"]

    try:
        db.mark_batch_and_children_failed(parent_batch_id, f"{error_type}: {error_message}")
    except botocore.exceptions.ClientError as err:
        # Soft failure, we want to still publish error sns even if we can't update db.
        logger.error("failed to set batch status to error: %s", err)

    message = {
        "batchId": parent_batch_id,
        "message": "Batch processing failed",
        "errorType": error_type,
        "errorString": error_message,
        "token": execution_id,
        "status": "FAILED",
    }

    output_sns_arn = os.getenv("DEFAULT_STATUS_SNS_ARN")
    if "destinationSnsArn" in request:
        output_sns_arn = request["destinationSnsArn"]
    sns = boto3.resource("sns")
    topic = sns.Topic(output_sns_arn)

    try:
        topic.publish(
            Message=json.dumps(message, indent=4),
        )
    except botocore.exceptions.ClientError as err:
        raise Exception(
            f"Service error publishing SNS response for batch id: {parent_batch_id}"
        ) from err

    return {
        "published_sns": message,
        "output_sns_arn": output_sns_arn,
    }
