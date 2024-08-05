import os
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key
from shared import log

from .constants import BatchCurrentStep
from .constants import BatchMetadataTableAttributes as Attributes
from .constants import BatchMetadataType, BatchStatus

BATCH_EXECUTION_METADATA_TABLE_NAME = os.getenv("BATCH_EXECUTION_METADATA_TABLE_NAME", "")

dynamodb = boto3.resource("dynamodb")
batch_execution_metadata_table = dynamodb.Table(BATCH_EXECUTION_METADATA_TABLE_NAME)


def get_child_batch_metadata(parent_batch_id, metadata_type):
    """Returns all the metadata associated to the parent batch.

    Parameters
    ----------
    batch_id: id to retrieve the associated batch_execution_metadata

    Returns
    ------
    Dynamo db items in question
    :param metadata_type: type of the metadata
    :param parent_batch_id: id of the parent batch metadata of interest
    """
    response = batch_execution_metadata_table.query(
        IndexName="ParentBatchIdIndex",
        KeyConditionExpression=Key(Attributes.PARENT_BATCH_ID).eq(parent_batch_id),
    )

    items = []
    for item in response["Items"]:
        if item[Attributes.BATCH_METADATA_TYPE] == metadata_type:
            items.append(item)

    return items


def get_batch_metadata(batch_id):
    """Fetches a batch execution metadata by the batch_execution_job_id.

    Parameters
    ----------
    batch_id: id to retrieve the associated batch_execution_metadata

    Returns
    ------
    Dynamo db item in question
    """
    response = batch_execution_metadata_table.get_item(
        Key={
            Attributes.BATCH_ID: batch_id,
        },
    )
    return response["Item"] if "Item" in response else None


def get_child_batch_metadata_all(
    parent_batch_id,
):
    """Returns all metadata associated with parent batch without filtering

    :param batch_id: id to retrieve the associated batch_execution_metadata
    """
    response = batch_execution_metadata_table.query(
        IndexName="ParentBatchIdIndex",
        KeyConditionExpression=Key(Attributes.PARENT_BATCH_ID).eq(parent_batch_id),
    )

    return response["Items"]


def get_batch_metadata_by_labeling_job_name(labeling_job_name, metadata_type=None):
    """Returns all the metadata associated to the parent batch.

    Parameters
    ----------
    labeling_job_name: labeling_job associated with the batch

    Returns
    ------
    Dynamo db items in question
    :param labeling_job_name: Name of the sagemaker GT Labeling Job
    :param metadata_type: metadata type of the batch
    """
    response = batch_execution_metadata_table.query(
        IndexName="LabelingJobNameIndex",
        KeyConditionExpression=Key(Attributes.LABELING_JOB_NAME).eq(labeling_job_name),
    )

    if metadata_type is None:
        return response["Items"]

    singular_item = []
    for item in response["Items"]:
        if (
            item[Attributes.LABELING_JOB_NAME] == labeling_job_name
            and item[Attributes.BATCH_METADATA_TYPE] == metadata_type
        ):
            singular_item.append(item)

    # Return a list to make API return type to be consistent.
    return singular_item


def update_batch_status(
    batch_execution_job_id,
    status,
):
    """Updates the status of a given batch_execution_job_id

    Parameters
    ----------
    batch_execution_job_id: id of the job to update the status
    status: BatchExecutionMetadata status to be in

    Returns
    ------
    Dynamo db update item response
    """
    response = batch_execution_metadata_table.update_item(
        Key={
            Attributes.BATCH_ID: batch_execution_job_id,
        },
        UpdateExpression="set #st=:s",
        ExpressionAttributeValues={
            ":s": status,
        },
        ExpressionAttributeNames={
            "#st": Attributes.BATCH_STATUS,
        },
    )
    return response


def update_batch_child_count(
    batch_execution_job_id,
    added_count,
):
    """Updates the status of a given batch_execution_job_id and returns true if completed

    Parameters
    ----------
    batch_execution_job_id: id of the job to update the status
    status: BatchExecutionMetadata status to be in
    added_frame_count: Additional frame count to atomically add.

    Returns
    ------
    Dynamo db update item response
    """
    response = batch_execution_metadata_table.update_item(
        Key={
            Attributes.BATCH_ID: batch_execution_job_id,
        },
        UpdateExpression="set #response_count_attr=#response_count_attr+:added_count_val",
        ExpressionAttributeValues={":added_count_val": added_count},
        ExpressionAttributeNames={
            "#response_count_attr": Attributes.NUM_CHILD_BATCHES_COMPLETE,
        },
        ReturnValues="ALL_NEW",
    )

    attributes = response.get("Attributes")
    if attributes is None:
        raise Exception("No attributes returnd")

    num_child_batches = attributes[Attributes.NUM_CHILD_BATCHES]
    num_child_batches_complete = attributes[Attributes.NUM_CHILD_BATCHES_COMPLETE]
    log.logger.info(
        f"Number child batches {num_child_batches_complete} total child batches {num_child_batches}"
    )

    complete = num_child_batches == num_child_batches_complete
    if complete:
        update_batch_status(batch_execution_job_id, BatchStatus.COMPLETE)
    return complete


def update_batch_step_token(
    batch_id,
    step_token,
):
    """Updates the step token of the given batch.
    Used for async waiting in step function invocations.

    Parameters
    ----------
    batch_id: id of the batch id to update.
    step_token: step token corresponding update

    Returns
    ------
    Dynamo db update item response
    """
    response = batch_execution_metadata_table.update_item(
        Key={
            Attributes.BATCH_ID: batch_id,
        },
        UpdateExpression="set #st=:s",
        ExpressionAttributeValues={
            ":s": step_token,
        },
        ExpressionAttributeNames={
            "#st": Attributes.STATE_TOKEN,
        },
    )
    return response


def insert_transformed_input_batch_metadata(
    batch_id,
    batch_status,
    batch_current_step,
    batch_metadata_type,
    error_message,
    labeling_jobs,
):
    """Creates a new record in the batch execution metadata table.

    Parameters
    ----------
    batch_id: id to retrieve the associated batch_execution_metadata
    Returns
    ------
    Created Dynamo db item as defined
    :param batch_current_step: current step of the batch execute : One of the defined ENUM values in
    BatchExecutionCurrentStep
    :param batch_metadata_type: type of the current metadata. One of the defined ENUM values in
    BatchExecutionMetadataType
    :param batch_id: Unique identifier of the execute state
    :param batch_status: Status of the execute state to be in
    """
    dynamo_db_item = {
        Attributes.BATCH_ID: batch_id,
        Attributes.BATCH_STATUS: batch_status,
        Attributes.BATCH_CURRENT_STEP: batch_current_step,
        Attributes.BATCH_METADATA_TYPE: batch_metadata_type,
        Attributes.MESSAGE: error_message,
        Attributes.LABELING_JOBS: labeling_jobs,
    }
    return batch_execution_metadata_table.put_item(Item=dynamo_db_item)


def insert_batch_metadata_input(
    batch_id, parent_batch_id, down_sampling_rate, input_manifest, batch_status
):
    """Inserts new batch metadata input post First level job

    Parameters
    ----------
    :param batch_status: status of the current batch metadata
    :param input_manifest: output location of the first level job
    :param down_sampling_rate: down_sampling rate
    :param parent_batch_id: id of the parent bath
    :param batch_id: current batchId to be inserted
    :param error_message: message to indicate any issue.

    Returns
    ------
    Dynamo db update item response
    """

    dynamo_db_item = {
        Attributes.BATCH_ID: batch_id,
        Attributes.DOWN_SAMPLING_RATE: Decimal(str(down_sampling_rate)),
        Attributes.BATCH_METADATA_TYPE: BatchMetadataType.HUMAN_INPUT_METADATA,
        Attributes.JOB_OUTPUT_LOCATION: input_manifest,
        Attributes.PARENT_BATCH_ID: parent_batch_id,
        Attributes.BATCH_STATUS: batch_status,
    }

    return batch_execution_metadata_table.put_item(Item=dynamo_db_item)


def update_batch_status(batch_id, status, error_message=""):
    """Updates the status of a given batch_id

    Parameters
    ----------
    batch_execution_job_id: id of the job to update the status
    status: batch_meta_Data status to be in

    Returns
    ------
    Dynamo db update item response
    :param error_message: message to indicate any issue.
    """

    response = batch_execution_metadata_table.update_item(
        Key={
            Attributes.BATCH_ID: batch_id,
        },
        UpdateExpression="set #st=:s, #errorMessage=:message",
        ExpressionAttributeValues={":s": status, ":message": error_message},
        ExpressionAttributeNames={
            "#st": Attributes.BATCH_STATUS,
            "#errorMessage": Attributes.MESSAGE,
        },
    )
    return response


def update_batch_current_step(
    batch_id,
    current_step,
):
    """Updates the status of a given batch_id

    Parameters
    ----------
    batch_id: id of the job to update the step
    current_step: currently being executed step

    Returns
    ------
    Dynamo db update item response
    """

    response = batch_execution_metadata_table.update_item(
        Key={
            Attributes.BATCH_ID: batch_id,
        },
        UpdateExpression="set #step=:current_step",
        ExpressionAttributeValues={":current_step": current_step},
        ExpressionAttributeNames={"#step": Attributes.BATCH_CURRENT_STEP},
    )
    return response


def associate_with_child_batch(
    parent_batch_id,
    child_batch_id,
    child_batch_metadata_type,
):
    """Updates the status of a given batch_id

    Parameters
    ----------
    batch_id: id of the job to update the step
    current_step: currently being executed step

    Returns
    ------
    Dynamo db update item response
    :param parent_batch_id: Id of the parent batch to update with the child metadata_id

    :param child_batch_id:  Id of the child batch metadata
    :param child_batch_metadata_type: type of child batch metadata to associate
    """

    if BatchMetadataType.FIRST_LEVEL == child_batch_metadata_type:
        attribute = Attributes.FIRST_LEVEL_BATCH_METADATA_ID
    elif BatchMetadataType.SECOND_LEVEL == child_batch_metadata_type:
        attribute = Attributes.SECOND_LEVEL_BATCH_METADATA_ID

    response = batch_execution_metadata_table.update_item(
        Key={
            Attributes.BATCH_ID: parent_batch_id,
        },
        UpdateExpression="set #child_batch_id_attr=:child_batch_id_val",
        ExpressionAttributeValues={":child_batch_id_val": child_batch_id},
        ExpressionAttributeNames={"#child_batch_id_attr": attribute},
    )
    return response


def insert_perform_labeling_job_metadata(
    parent_batch_id,
    batch_id,
    batch_status,
    batch_metadata_type,
    num_children_batches,
):
    """Creates a new record in the batch execution metadata table.

    Parameters
    ----------
    batch_id: id to retrieve the associated batch_execution_metadata
    Returns
    ------
    Created Dynamo db item as defined
    :param parent_batch_id: batch_id associated to this state specific batch_id
    :param batch_metadata_type: type of the current metadata. One of the defined ENUM values in BatchExecutionMetadataType
    :param batch_id: Unique identifier of the execute state
    :param batch_status: Status of the execute state to be in
    """

    dynamo_db_item = {
        Attributes.PARENT_BATCH_ID: parent_batch_id,
        Attributes.BATCH_ID: batch_id,
        Attributes.BATCH_STATUS: batch_status,
        Attributes.BATCH_METADATA_TYPE: batch_metadata_type,
        Attributes.NUM_CHILD_BATCHES: num_children_batches,
        Attributes.NUM_CHILD_BATCHES_COMPLETE: 0,
    }
    return batch_execution_metadata_table.put_item(Item=dynamo_db_item)


def insert_job_level_metadata(
    parent_batch_id,
    batch_id,
    batch_status,
    labeling_job_name,
    label_attribute_name,
    label_category_s3_uri,
    job_input_s3_uri,
    job_output_s3_uri,
    num_frames=None,
):
    """Creates a new record in the batch execution metadata table.

    Parameters
    ----------
    batch_id: id to retrieve the associated batch_execution_metadata
    Returns
    ------
    Created Dynamo db item as defined
    :param parent_batch_id: batch_id associated to this state specific batch_id
    :param batch_metadata_type: type of the current metadata. One of the defined ENUM values in BatchExecutionMetadataType
    :param batch_id: Unique identifier of the execute state
    :param batch_status: Status of the execute state to be in
    """

    dynamo_db_item = {
        Attributes.PARENT_BATCH_ID: parent_batch_id,
        Attributes.BATCH_ID: batch_id,
        Attributes.BATCH_STATUS: batch_status,
        Attributes.BATCH_METADATA_TYPE: BatchMetadataType.JOB_LEVEL,
        Attributes.LABELING_JOB_NAME: labeling_job_name,
        Attributes.LABEL_CATEGORY_CONFIG: label_category_s3_uri,
        Attributes.LABEL_ATTRIBUTE_NAME: label_attribute_name,
        Attributes.JOB_INPUT_LOCATION: job_input_s3_uri,
        Attributes.JOB_OUTPUT_LOCATION: job_output_s3_uri,
    }

    if num_frames is not None:
        # We are tracking frame level completions.
        dynamo_db_item[Attributes.NUM_CHILD_BATCHES] = num_frames
        dynamo_db_item[Attributes.NUM_CHILD_BATCHES_COMPLETE] = 0

    return batch_execution_metadata_table.put_item(Item=dynamo_db_item)


def update_batch_down_sample_location(
    batch_id,
    down_sample_location,
):
    """Append new down_sample location to the existing batchId
    Parameters
    ----------
    batch_id: id of the job to update the step
    down_sample_location: down_sample location
    Returns
    ------
    Dynamo db update item response
    """
    response = batch_execution_metadata_table.update_item(
        Key={
            Attributes.BATCH_ID: batch_id,
        },
        UpdateExpression="set #attr_down_sample_location=:down_sample_location",
        ExpressionAttributeValues={":down_sample_location": down_sample_location},
        ExpressionAttributeNames={
            "#attr_down_sample_location": Attributes.JOB_DOWN_SAMPLE_LOCATION
        },
    )
    return response


def insert_processed_input_batch_metadata(parent_batch_id, batch_id, job_name, job_input_location):
    """Inserts a single frame of data.
    :param parent_batch_id: Batch that owns this frame.
    :param batch_id: Unique ID, "batchId/frameIndex" for frames, matches SNS Deduplication key
    :param job_name: Consumer LabelingJob which will use the data
    :param job_input_lcoation: Input to be used by the supplied labeling job_name


    We are re-using the batch metadata table.
    """
    dynamo_db_item = {
        Attributes.PARENT_BATCH_ID: parent_batch_id,
        Attributes.BATCH_ID: batch_id,
        Attributes.BATCH_METADATA_TYPE: BatchMetadataType.PROCESS_LEVEL,
        Attributes.LABELING_JOB_NAME: job_name,
        Attributes.JOB_INPUT_LOCATION: job_input_location,
    }
    return batch_execution_metadata_table.put_item(Item=dynamo_db_item)


def insert_frame_batch_metadata(
    parent_batch_id,
    batch_id,
    batch_status,
    frame_index,
):
    """Inserts a single frame of data.
    :param parent_batch_id: Batch that owns this frame.
    :param batch_id: Unique ID, "batchId/frameIndex" for frames, matches SNS Deduplication key
    :param batch_status: Current status of the frame, in progress / completed
    :param frame_index: Index of the frame within the batch, eg 0, 1, 2


    We are re-using the batch metadata table.
    """
    dynamo_db_item = {
        Attributes.PARENT_BATCH_ID: parent_batch_id,
        Attributes.BATCH_ID: batch_id,
        Attributes.BATCH_STATUS: batch_status,
        Attributes.BATCH_METADATA_TYPE: BatchMetadataType.FRAME_LEVEL,
        Attributes.FRAME_INDEX: frame_index,
    }
    return batch_execution_metadata_table.put_item(Item=dynamo_db_item)


def get_frame_batch_metadata(batch_frame_id):
    """Inserts a single frame of data.
    :param batch_frame_id: Unique ID, "batchId/frameIndex" for frames, matches SNS Deduplication key

    We are re-using the batch metadata table.
    """
    response = batch_execution_metadata_table.get_item(
        Key={
            Attributes.BATCH_ID: batch_frame_id,
        },
    )
    return response["Item"] if "Item" in response else None


def mark_batch_and_children_failed(
    batch_id,
    error_message="",
):
    """Marks a whole batch as failed, including all children batches.

    :param batch_id: Unique ID of the batch to mark as deleted.
    :param error_message: Optional error message to store in dynamo with frame.
    """
    update_batch_status(batch_id, BatchStatus.INTERNAL_ERROR, error_message=error_message)

    items = get_child_batch_metadata_all(batch_id)
    for item in items:
        mark_batch_and_children_failed(item["BatchId"], error_message)


def get_batches_by_type_status(batch_type, batch_status):
    """Return list of smgt jobs

    Parameters
    ----------
    category: type of job (PRIMARY or SECONDARY)
    status: smgt jobs for the status to be in
    Returns
    ------
    Dynamo db item or None if doesn't exist
    """
    try:
        # TODO: Replace with get_item
        response = batch_execution_metadata_table.query(
            IndexName="BatchMetadataTypeStatusIndex",
            KeyConditionExpression=Key("BatchMetadataType").eq(batch_type)
            & Key("BatchStatus").eq(batch_status),
        )
        log.logger.info("DDB response {}".format(response))
    except Exception as err:
        log.logger.error(
            f"failed to query DynamoDB for batch type {batch_type} status {batch_status}, error: {err}"
        )
        return None

    items = response.get("Items")
    if items is None:
        return []
    return items
