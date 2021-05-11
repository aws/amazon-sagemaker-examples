"""
Lambda responsible for handling API requests to show the list of batches, or to
show a particular batch.
"""

import json

import botocore
from shared import db
from shared.api_helpers import input_batch_to_human_readable
from shared.constants import BatchMetadataType, BatchStatus
from shared.log import log_request_and_context, logger


def get_all_batches():
    """
    Generate a dictionary of batch ids by status

    :returns: json serializable dictionary indexed by batch status and containing
    list of batch ids
    """
    batch_ids_by_status = {}

    for status in [
        BatchStatus.IN_PROGRESS,
        BatchStatus.VALIDATION_FAILURE,
        BatchStatus.INTERNAL_ERROR,
        BatchStatus.COMPLETE,
    ]:
        batches = db.get_batches_by_type_status(BatchMetadataType.INPUT, status)
        batch_ids = [batch["BatchId"] for batch in batches]
        batch_ids_by_status[status] = batch_ids

    return batch_ids_by_status


def get_batch_description(batch_id):
    """
    Looks up a batch using the given batch id and validates that the batch
    is of appropriate type, then returns a human readable representation.

    :param batch_id: Id of batch to convert to human readable description
    :returns: json serializable description of a given batch
    """
    batch_metadata = db.get_batch_metadata(batch_id)

    # User should only be querying for parent batches of type "INPUT", not frame
    # level batches.
    if batch_metadata["BatchMetadataType"] != BatchMetadataType.INPUT:
        logger.error(
            "User requested existing batch, but it is of the wrong type (not INPUT): %s", batch_id
        )
        return None

    # Convert batch metadata to something user presentable.
    return input_batch_to_human_readable(batch_metadata)


def handle_request(request):
    """
    Handles requests for all batches or specific batch information

    :param request: Dictionary containing "batchId"
    :returns: Dictionary consisting of the api response body.
    """
    batch_id = request["batchId"]
    if batch_id is None:
        return get_all_batches()

    return get_batch_description(batch_id)


def parse_request(event):
    """
    Parses a given request's url params.

    :param event: API gateway input event for GET request
    :returns: Parsed request params dictionary
    """
    url_params = event.get("multiValueQueryStringParameters")
    if url_params is None:
        return {"batchId": None}

    batch_ids = url_params.get("batchId")

    if len(batch_ids) != 1:
        return {"batchId": None}

    batch_id = batch_ids[0]
    return {
        "batchId": batch_id,
    }


def lambda_handler(event, context):
    """Lambda function that responds shows active batch information.

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
    log_request_and_context(event, context)

    try:
        request = parse_request(event)
    except (KeyError, ValueError) as err:
        logger.error("Failed to parse request: %s", err)
        return {
            "statusCode": 400,
            "body": "Error: failed to parse request.",
        }

    try:
        batch_info = handle_request(request)
    except botocore.exceptions.ClientError as err:
        logger.error("Boto call failed to execute during request handling: {err}")
        return {
            "statusCode": 500,
            "body": "Error: internal error",
        }

    if batch_info is None:
        logger.error("Batch id not found, request: %s", request)
        return {
            "statusCode": 400,
            "body": f"batch id: {request['batchId']} not found",
            "headers": {"X-Amzn-ErrorType": "InvalidParameterException"},
        }

    response = {
        "statusCode": 200,
        "body": json.dumps(batch_info, indent=4, default=str),
        "isBase64Encoded": False,
    }
    return response
