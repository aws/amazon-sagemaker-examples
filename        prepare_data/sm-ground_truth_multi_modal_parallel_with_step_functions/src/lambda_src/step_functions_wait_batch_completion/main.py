"""
Step function state that waits for batch completion by storing a step token in db.
"""

import json

from shared import db, log
from shared.constants import BatchMetadataTableAttributes, BatchStatus
from shared.lambda_context import get_boto_client


def lambda_handler(event, context):
    """Lambda function that stores the current step function state token into dynamo.

    Parameters
    ----------
    event: dict, required
    context: object, required Lambda Context runtime methods and attributes
    Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    Lambda Output Format: dict
    """
    step_input = extract_input(event)
    batch_id, step_token = step_input["batch_id"], step_input["step_token"]

    batch_metadata = db.get_batch_metadata(batch_id)
    if batch_metadata is None:
        raise Exception(f"Failed to find batch corresponding to id: {batch_id}")

    num_child = batch_metadata.get(BatchMetadataTableAttributes.NUM_CHILD_BATCHES)
    if num_child is not None and num_child == 0:
        log.logger.info("No children in batch, skipping wait for batch completion.")
        # Mark the db entry as complete and send task success to unblock the step function.
        sfn_client = get_boto_client("stepfunctions", context.invoked_function_arn)

        db.update_batch_status(batch_id, BatchStatus.COMPLETE)

        # Send status token to step function.
        response = sfn_client.send_task_success(
            taskToken=step_token, output=json.dumps({"batch": batch_id})
        )
        log.logger.info("Response for Step function token %s: %s", step_token, response)
    else:
        # Not skipping wait for batch completion, the listener is responsible for marking
        # the batch as complete now.
        db.update_batch_step_token(batch_id, step_token)

    return {
        "batch_id": batch_id,
        "step_token": step_token,
    }


def extract_input(event):
    """Validates input fields exist and raises KeyError if not.

    Parameters
    ----------
    event: dict, required

    Returns
    ------
    Output Format: dict of batch id and step token from input.
    """
    # Require these two keys to be present in the input.
    return {
        "batch_id": event["batch_id"],
        "step_token": event["token"],
    }
