"""
Step function state that waits for batch completion by storing a step token in db.
"""

import json
import os
from urllib.parse import urlparse

import boto3
from shared import db, log

sns = boto3.resource("sns")
first_level_job_completion_sns_arn = os.getenv("FIRST_LEVEL_JOB_COMPLETION_SNS")


def lambda_handler(event, context):
    """Lambda function that stores the current step function state token into dynamo, and sends the sns notification.

    Parameters
    ----------
    event: dict, required
    context: object, required Lambda Context runtime methods and attributes
    Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    Lambda Output Format: dict
    """
    batch_id, step_token, output_location = (
        event["batch_id"],
        event["token"],
        event["output_location"],
    )

    db.update_batch_step_token(batch_id, step_token)

    topic = sns.Topic(first_level_job_completion_sns_arn)
    one_week_in_sec = 604800

    message = {
        "s3_output_location": output_location,
        "s3_output_location_signed_url": create_pre_signed_url(output_location, one_week_in_sec),
        "batchId": batch_id,
    }
    log.logger.info(
        "Publishing SNS notification to : " + first_level_job_completion_sns_arn + str(message)
    )

    response = topic.publish(Message=json.dumps(message, indent=4))
    log.logger.info(response)
    return {
        "batch_id": batch_id,
        "step_token": step_token,
    }


def create_pre_signed_url(s3_prefix, expiration=3600):
    """Generate a presigned URL to share an S3 object"""

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client("s3")
    parse_result = urlparse(s3_prefix, allow_fragments=False)
    object_key = parse_result.path.lstrip("/")

    response = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": parse_result.netloc, "Key": object_key},
        ExpiresIn=expiration,
    )

    # The response contains the pre-signed URL
    return response
