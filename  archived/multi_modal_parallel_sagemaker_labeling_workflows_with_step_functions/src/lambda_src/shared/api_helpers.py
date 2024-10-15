"""
Contains common functionality used in API layer, for example
conversions between database types and API json.
"""

import boto3
import botocore
from shared import db
from shared.constants import BatchMetadataTableAttributes as Attributes
from shared.constants import BatchMetadataType
from shared.log import logger


def split_uri(s3_uri):
    """Split an s3 uri into the bucket and object name

    :param s3_uri: string
    """
    if not s3_uri.startswith("s3://"):
        # This is a local path, indicate using None
        raise ValueError(f"failed to parse s3 uri: {s3_uri}")
    bucket, key = s3_uri.split("s3://")[1].split("/", 1)
    return bucket, key


def create_presigned_url(s3_uri, expiration=86400):
    """Generate a presigned URL to share an S3 object for validation of 24 hours

    :param s3_uri: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    bucket_name, object_name = split_uri(s3_uri)

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client("s3", config=botocore.config.Config(signature_version="s3v4"))
    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration,
        )
    except botocore.exceptions.ClientError as err:
        # Soft failure.
        logger.error("failed to generate presigned url: %s", err)
        return None

    # The response contains the presigned URL
    return response


def job_to_human_readable(db_job):
    """Generates a human readable version of a SMGT job metadata (with correct casing for API use)"""
    return {
        "jobName": db_job["JobName"],
        "status": db_job["Status"],
        "category": db_job["Category"],
        "jobArn": db_job["JobArn"],
        "inputSnsArn": db_job["InputSnsArn"],
        "s3OutputPath": db_job["S3OutputPath"],
        "workteamArn": db_job["WorkteamArn"],
        "labelCategoryConfigS3Uri": db_job["LabelCategoryConfigS3Uri"],
    }


def job_level_to_human_readable(batch):
    """Job level metadata to human readable"""
    response = {
        "batchId": batch[Attributes.BATCH_ID],
        "batchStatus": batch[Attributes.BATCH_STATUS],
        "labelingJobName": batch[Attributes.LABELING_JOB_NAME],
        "labelAttributeName": batch[Attributes.LABEL_ATTRIBUTE_NAME],
        "labelCategoryS3Uri": batch[Attributes.LABEL_CATEGORY_CONFIG],
        "jobInputS3Uri": batch[Attributes.JOB_INPUT_LOCATION],
        "jobInputS3Url": create_presigned_url(batch[Attributes.JOB_INPUT_LOCATION]),
        "jobOutputS3Uri": batch[Attributes.JOB_OUTPUT_LOCATION],
        "jobOutputS3Url": create_presigned_url(batch[Attributes.JOB_OUTPUT_LOCATION]),
    }

    num_frames = batch.get(Attributes.NUM_CHILD_BATCHES)
    num_frames_completed = batch.get(Attributes.NUM_CHILD_BATCHES_COMPLETE)
    if num_frames is not None and num_frames_completed is not None:
        response["numFrames"] = num_frames
        response["numFramesCompleted"] = num_frames_completed

    return response


def first_or_second_level_to_human_readable(batch):
    """Converts a first or second level batch to human readable"""
    job_level_batches = db.get_child_batch_metadata(
        batch[Attributes.BATCH_ID], BatchMetadataType.JOB_LEVEL
    )
    job_responses = [
        job_level_to_human_readable(job_level_batch) for job_level_batch in job_level_batches
    ]

    return {
        "status": batch[Attributes.BATCH_STATUS],
        "numChildBatches": batch[Attributes.NUM_CHILD_BATCHES],
        "numChildBatchesComplete": batch[Attributes.NUM_CHILD_BATCHES_COMPLETE],
        "jobLevels": job_responses,
    }


def input_batch_to_human_readable(batch):
    """
    Generates a human friendly version of an INPUT batch metadata with presigned urls

    :param batch_metadata: Batch metadata dictionary
    :returns: json serializable dictionary of batch info
    """

    # User should only be querying for parent batches of type "INPUT", not frame
    # level batches.
    if batch[Attributes.BATCH_METADATA_TYPE] != BatchMetadataType.INPUT:
        logger.error(
            "User requested existing batch, but it is of the wrong input type: %s",
            batch[Attributes.BATCH_ID],
        )
        return None

    response = {
        "batchId": batch[Attributes.BATCH_ID],
        "status": batch[Attributes.BATCH_STATUS],
        # Straight copy of request labeling jobs to acknowledge the request.
        "inputLabelingJobs": batch[Attributes.LABELING_JOBS],
    }

    stage_attributes = [
        ("firstLevel", BatchMetadataType.FIRST_LEVEL),
        ("secondLevel", BatchMetadataType.SECOND_LEVEL),
        ("thirdLevel", BatchMetadataType.THIRD_LEVEL),
    ]

    for field_name, attribute in stage_attributes:
        first_or_second_level_batches = db.get_child_batch_metadata(
            batch[Attributes.BATCH_ID], attribute
        )
        for first_or_second_level_batch in first_or_second_level_batches:
            response[field_name] = first_or_second_level_to_human_readable(
                first_or_second_level_batch
            )

    return response
