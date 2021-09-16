"""
Handles cleaning up after a batch has succesfully been processed.

Sends message to status SNS on completion.
"""

import json
import os

import boto3
import botocore
from shared import db
from shared.api_helpers import input_batch_to_human_readable
from shared.constants import BatchStatus
from shared.log import log_request_and_context, logger

sns = boto3.resource("sns")


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

    try:
        request_input = parse_input(event)
    except KeyError as err:
        logger.error("Input event missing required args: %s: %s", event, err)
        raise Exception("Failed to parse input lambda handler") from err

    batch_id = request_input["batch_id"]
    # Mark the batch as completed.
    try:
        db.update_batch_status(batch_id, BatchStatus.COMPLETE)
    except botocore.exceptions.ClientError as err:
        raise Exception(f"failed to mark batch {batch_id} complete") from err

    batch_metadata = db.get_batch_metadata(batch_id)
    batch_info = input_batch_to_human_readable(batch_metadata)

    message = {
        "batchId": batch_id,
        "message": "Batch processing has completed successfully.",
        "batchInfo": batch_info,
        "token": request_input["execution_id"],
        "status": "SUCCESS",
    }

    output_sns_arn = os.getenv("DEFAULT_STATUS_SNS_ARN")
    if request_input["output_sns_arn"]:
        output_sns_arn = request_input["output_sns_arn"]

    topic = sns.Topic(output_sns_arn)
    try:
        topic.publish(
            Message=json.dumps(message, indent=4, default=str),
        )
    except botocore.exceptions.ClientError as err:
        raise Exception(f"Service error publishing SNS response for batch id: {batch_id}") from err

    return {
        "published_sns": message,
        "output_sns_arn": output_sns_arn,
    }


def parse_input(event):
    """Parses all input required from step function."""
    input_request = event["input"]

    return {
        "batch_id": input_request["transformation_step_output"]["batch_id"],
        "output_sns_arn": input_request.get("destinationSnsArn"),
        "execution_id": event["execution_id"],
    }
