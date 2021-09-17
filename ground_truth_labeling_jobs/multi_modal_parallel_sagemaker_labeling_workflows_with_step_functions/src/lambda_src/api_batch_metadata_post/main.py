"""
API handler for batch creation.

Handles starting a new step function execution for the batch
and validating input.
"""

import json

from shared import db, log
from shared.constants import BatchMetadataTableAttributes, BatchMetadataType, BatchStatus
from shared.lambda_context import get_boto_client


def construct_validation_error(message):
    return {
        "statusCode": 400,
        "body": f"Error: {message}",
    }


def lambda_handler(event, context):
    """Lambda function that executes batch creation API

    Parameters
    ----------
    event: dict, required API gateway request with an input SQS arn, output SQS arn
    context: object, required Lambda Context runtime methods and attributes
    Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    Lambda Output Format: dict
    Return doc:
    https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    sfn_client = get_boto_client("stepfunctions", context.invoked_function_arn)

    log.log_request_and_context(event, context)

    body = json.loads(event.get("body"))

    batch_id = body.get("batchId")
    down_sampling_rate = body.get("downSamplingRate")

    input_metadata_batch_id = f"{batch_id}-{BatchMetadataType.HUMAN_INPUT_METADATA.lower()}"
    input_metadata_batch = db.get_batch_metadata(input_metadata_batch_id)

    if input_metadata_batch:
        return construct_validation_error(
            "The system indicates the you have already input the down sampling rate "
            + f'{input_metadata_batch.get("DownSamplingRate")}'
        )

    if batch_id is None:
        return construct_validation_error("BatchId is required.")
    if down_sampling_rate is None:
        return construct_validation_error("DownSampling rate is required.")

    batch_metadata = db.get_batch_metadata(batch_id)

    if not batch_metadata:
        return construct_validation_error(f"BatchMetadata not found for the batchId: {batch_id}")
    else:
        if down_sampling_rate < 0 or down_sampling_rate > 100:
            return construct_validation_error("Expected down sampling range in between 0 to 100.")

    first_level_batch = db.get_child_batch_metadata(batch_id, BatchMetadataType.FIRST_LEVEL)
    job_output_location = first_level_batch[BatchMetadataTableAttributes.JOB_OUTPUT_LOCATION]

    state_token = batch_metadata.get(BatchMetadataTableAttributes.STATE_TOKEN)

    if not state_token:
        return construct_validation_error(
            f"The system indicates the batch exeuction is not currently at the wait step {batch_metadata}"
        )

    sfn_client.send_task_success(
        taskToken=batch_metadata[BatchMetadataTableAttributes.STATE_TOKEN],
        output=json.dumps(
            {
                "batch_id": batch_metadata[
                    BatchMetadataTableAttributes.FIRST_LEVEL_BATCH_METADATA_ID
                ],
                "s3_output_path": job_output_location,
                "down_sampling_rate": down_sampling_rate,
                "token_sent_source_arn": context.invoked_function_arn,
            }
        ),
    )

    db.insert_batch_metadata_input(
        batch_id=input_metadata_batch_id,
        parent_batch_id=batch_id,
        down_sampling_rate=down_sampling_rate,
        input_manifest=job_output_location,
        batch_status=BatchStatus.COMPLETE,
    )

    response = {
        "statusCode": 200,
        "body": "Successfully input metadata to resume batch execution : "
        + f"batchId : {batch_id}, downSamplingRate: {down_sampling_rate}",
        "isBase64Encoded": False,
    }
    return response
