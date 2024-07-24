"""
Downsample frames using downsample rate.
"""

import os
import random

from shared import db, log, s3_accessor
from shared.constants import BatchMetadataTableAttributes, BatchMetadataType

batch_processing_bucket_name = os.getenv("BATCH_PROCESSING_BUCKET_ID")


def down_sample_to_proportion(rows, percentile=100):
    """Performs downsampling by choosing a percentage of frames from rows."""
    # TODO:  Selection should be based on frames in DDB table
    sampled_size = int((percentile / 100) * len(rows))

    return random.sample(rows, sampled_size)


def lambda_handler(event, context):
    """Lambda function that ...
    Down sampling of the input manifest to send to the next step

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
    batch_id = event["batch_id"]
    batch_metadata = db.get_batch_metadata(batch_id)
    current_metadata_type = batch_metadata[BatchMetadataTableAttributes.BATCH_METADATA_TYPE]

    if current_metadata_type == BatchMetadataType.FIRST_LEVEL:
        processing_output_job_level = 1
    elif current_metadata_type == BatchMetadataType.SECOND_LEVEL:
        processing_output_job_level = 2
    else:
        processing_output_job_level = 3

    parent_batch_id = batch_metadata[BatchMetadataTableAttributes.PARENT_BATCH_ID]
    parent_batch_metadata = db.get_batch_metadata(parent_batch_id)

    current_level_completed_labeling_jobs = []
    future_level_labeling_jobs = []

    labeling_jobs = parent_batch_metadata["LabelingJobs"]

    for labeling_job in labeling_jobs:
        if labeling_job["jobLevel"] == processing_output_job_level:
            current_level_completed_labeling_jobs.append(labeling_job)
        elif labeling_job["jobLevel"] > processing_output_job_level:
            future_level_labeling_jobs.append(labeling_job)

    for completed_labeling_job in current_level_completed_labeling_jobs:

        completed_labeling_job_name = completed_labeling_job["jobName"]

        for future_level_labeling_job in future_level_labeling_jobs:
            if completed_labeling_job_name == future_level_labeling_job["inputConfig"][
                "chainFromJobName"
            ] and future_level_labeling_job["inputConfig"].get("downSamplingRate"):

                future_level_labeling_job_name = future_level_labeling_job["jobName"]

                job_level_batch_metadata = db.get_batch_metadata_by_labeling_job_name(
                    completed_labeling_job_name, BatchMetadataType.JOB_LEVEL
                )[0]

                completed_job_output_location = job_level_batch_metadata[
                    BatchMetadataTableAttributes.JOB_OUTPUT_LOCATION
                ]

                s3_object = s3_accessor.fetch_s3(completed_job_output_location)

                content = s3_object.decode("utf-8")
                items = content.splitlines()

                down_sample_rate = future_level_labeling_job["inputConfig"]["downSamplingRate"]
                down_sampled_data = down_sample_to_proportion(items, down_sample_rate)

                future_level_labeling_input_location = (
                    f"s3://{batch_processing_bucket_name}/batch_manifests/"
                    f"{future_level_labeling_job_name}/processed/data.manifest"
                )

                s3_accessor.put_s3(
                    future_level_labeling_input_location, "\n".join(down_sampled_data)
                )

                batch_id = (
                    f"{parent_batch_id}-{future_level_labeling_job_name}-"
                    f"{BatchMetadataType.PROCESS_LEVEL.lower()}"
                )

                db.insert_processed_input_batch_metadata(
                    parent_batch_id,
                    batch_id,
                    future_level_labeling_job_name,
                    future_level_labeling_input_location,
                )
    return None
