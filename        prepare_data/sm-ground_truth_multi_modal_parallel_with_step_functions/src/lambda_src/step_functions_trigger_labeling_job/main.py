"""
Handles sending a batch manifest frames to either
the first or second level smgt jobs.
"""

import json
import os
from collections import namedtuple
from urllib.parse import urlparse

import boto3
import botocore
from shared import db, label_arn, log
from shared.constants import (
    SNS_DEDUPLICATION_KEY_NAME,
    BatchCurrentStep,
    BatchMetadataTableAttributes,
    BatchMetadataType,
    BatchStatus,
    SmgtJobType,
)
from shared.s3_accessor import fetch_s3, put_s3

sess = boto3.session.Session()
region = sess.region_name
sagemaker = boto3.client("sagemaker")
s3 = boto3.resource("s3")
sns = boto3.resource("sns")

# Bucket used for storing batch level output manifest files.
batch_processing_bucket_name = os.getenv("BATCH_PROCESSING_BUCKET_ID")
first_level_team_arn = os.getenv("SMGT_QA_FIRST_LEVEL_TEAM_ARN")
second_level_team_arn = os.getenv("SMGT_QA_SECOND_LEVEL_TEAM_ARN")
labeler_execution_role_arn = os.getenv("SMGT_QA_LABELER_EXECUTION_ROLE_ARN")


def construct_labeling_job_input(
    parent_batch_id,
    input_manifest_url,
    audit_label_attribute_name,
    label_category_config_uri,
    job_params,
    output_path,
):
    labeling_job_name = job_params["jobName"]
    labeling_job_type = job_params["jobModality"]
    task_availibility_lifetime_seconds = int(job_params["taskAvailabilityLifetimeInSeconds"])
    task_time_limit_seconds = int(job_params["taskTimeLimitInSeconds"])
    max_concurrent_task_count = int(job_params["maxConcurrentTaskCount"])
    workteam_arn = job_params["workteamArn"]

    label_attribute_name = job_params.get(
        "labelAttributeName",
        label_arn.JobModality.job_name_to_label_attribute(labeling_job_type, labeling_job_name),
    )

    if audit_label_attribute_name is not None:
        # We're auditing a job that already exists, use its label attribute name.
        label_category_s3_url = urlparse(label_category_config_uri)
        s3_object = s3.Object(
            label_category_s3_url.netloc,
            label_category_s3_url.path.lstrip("/"),
        )
        content = s3_object.get()["Body"].read().decode("utf-8")
        json_content = json.loads(content)
        json_content["auditLabelAttributeName"] = audit_label_attribute_name
        log.logging.info(f"Regenerating category file json : {json_content}")
        label_category_config_uri = f"s3://{batch_processing_bucket_name}/label_category_input/{parent_batch_id}-{labeling_job_name}/category-file.json"
        put_s3(label_category_config_uri, bytes(json.dumps(json_content).encode("UTF-8")))

    return {
        "LabelingJobName": labeling_job_name,
        "HumanTaskConfig": {
            "AnnotationConsolidationConfig": label_arn.annotation_consolidation_config(
                region, labeling_job_type
            ),
            "MaxConcurrentTaskCount": max_concurrent_task_count,
            "NumberOfHumanWorkersPerDataObject": 1,
            "PreHumanTaskLambdaArn": label_arn.pre_human_task_lambda_arn(region, labeling_job_type),
            "TaskAvailabilityLifetimeInSeconds": task_availibility_lifetime_seconds,
            "TaskDescription": "Audit the labeling job",
            "TaskTimeLimitInSeconds": task_time_limit_seconds,
            "TaskTitle": f"Point Cloud Audit: {labeling_job_name}",
            "UiConfig": label_arn.ui_config(region, labeling_job_type),
            "WorkteamArn": workteam_arn,
        },
        "InputConfig": {
            "DataAttributes": {
                "ContentClassifiers": [
                    "FreeOfPersonallyIdentifiableInformation",
                    "FreeOfAdultContent",
                ]
            },
            "DataSource": {"S3DataSource": {"ManifestS3Uri": input_manifest_url}},
        },
        "LabelAttributeName": label_attribute_name,
        "LabelCategoryConfigS3Uri": label_category_config_uri,
        "OutputConfig": {
            "S3OutputPath": output_path,
        },
        "StoppingConditions": {"MaxPercentageOfInputDatasetLabeled": 100},
        "RoleArn": labeler_execution_role_arn,
    }


def trigger_batch_job(parent_batch_id, job_input, job_params):
    """Start a batch job"""
    job_name = job_params["jobName"]
    job_modality = job_params["jobModality"]

    batch_id = f"{parent_batch_id}-{job_name}"

    output_path = (
        f"s3://{batch_processing_bucket_name}/batch_manifests/{job_modality}/{batch_id}/output"
    )

    # If a label category file wasn't provided as API input, use the previous
    # job's label category file.
    label_category_config_uri = job_input.label_category_s3_uri
    if "labelCategoryConfigS3Uri" in job_params:
        label_category_config_uri = job_params["labelCategoryConfigS3Uri"]

    # batch_job_input_data = event["batch_job_input"]
    labeling_job_request = construct_labeling_job_input(
        parent_batch_id=parent_batch_id,
        input_manifest_url=job_input.input_manifest_s3_uri,
        audit_label_attribute_name=job_input.label_attribute_name,
        label_category_config_uri=label_category_config_uri,
        job_params=job_params,
        output_path=output_path,
    )

    sagemaker.create_labeling_job(**labeling_job_request)
    s3_output_path = f"{output_path}/{job_name}/manifests/output/output.manifest"

    db.insert_job_level_metadata(
        parent_batch_id=parent_batch_id,
        batch_id=batch_id,
        batch_status=BatchStatus.WAIT_FOR_SMGT_RESPONSE,
        labeling_job_name=job_name,
        label_attribute_name=labeling_job_request["LabelAttributeName"],
        label_category_s3_uri=labeling_job_request["LabelCategoryConfigS3Uri"],
        job_input_s3_uri=labeling_job_request["InputConfig"]["DataSource"]["S3DataSource"][
            "ManifestS3Uri"
        ],
        job_output_s3_uri=s3_output_path,
    )


def chainable_batches(parent_batch_id, job_level):
    """Returns all batches that have completed and we could possibly chain from"""

    if job_level == 1:
        raise Exception("can't chain in job_level 1")

    if job_level == 2:
        return db.get_child_batch_metadata(parent_batch_id, BatchMetadataType.FIRST_LEVEL)

    if job_level == 3:
        first_level_batches = db.get_child_batch_metadata(
            parent_batch_id, BatchMetadataType.FIRST_LEVEL
        )
        second_level_batches = db.get_child_batch_metadata(
            parent_batch_id, BatchMetadataType.SECOND_LEVEL
        )

        return first_level_batches + second_level_batches

    raise Exception("unsupported job level")


def input_config_to_job_input(input_batch_id, job_name, job_level, input_config):

    """Finds input data information from a static manifest or from previous job"""
    JobInput = namedtuple(
        "JobInput",
        ["input_manifest_s3_uri", "label_attribute_name", "label_category_s3_uri"],
    )

    input_manifest_s3_uri = input_config.get("inputManifestS3Uri")
    if input_manifest_s3_uri is not None:
        return JobInput(
            input_manifest_s3_uri=input_manifest_s3_uri,
            label_attribute_name=None,
            label_category_s3_uri=None,
        )

    chain_to_job_name = job_name
    chain_from_job_name = input_config["chainFromJobName"]

    # Only support jobs within the current batch for now.
    if job_level == 1:
        raise Exception("can't chain in job_level 1")

    batches = chainable_batches(input_batch_id, job_level)
    if len(batches) == 0:
        raise Exception("no chainable batches found")

    processed_job_level_batch = next(
        iter(
            db.get_batch_metadata_by_labeling_job_name(
                chain_to_job_name, BatchMetadataType.PROCESS_LEVEL
            )
        ),
        None,
    )

    prev_level_jobs = []
    for batch in batches:
        prev_level_jobs += db.get_child_batch_metadata(
            batch["BatchId"], BatchMetadataType.JOB_LEVEL
        )

    for job in prev_level_jobs:
        if job[BatchMetadataTableAttributes.LABELING_JOB_NAME] == chain_from_job_name:
            # If available, use the downsampled manifest file as input to the new job
            if processed_job_level_batch:
                processed_data_location = processed_job_level_batch[
                    BatchMetadataTableAttributes.JOB_INPUT_LOCATION
                ]
            else:
                processed_data_location = None

            batch_output_location = (
                processed_data_location or job[BatchMetadataTableAttributes.JOB_OUTPUT_LOCATION]
            )

            return JobInput(
                input_manifest_s3_uri=batch_output_location,
                label_attribute_name=job[BatchMetadataTableAttributes.LABEL_ATTRIBUTE_NAME],
                label_category_s3_uri=job[BatchMetadataTableAttributes.LABEL_CATEGORY_CONFIG],
            )

    raise Exception(f"chain job {chain_from_job_name} not found")


def trigger_labeling_job(input_batch_id, batch_id, job_params):
    """Start a labeling job and store metadata in JOB_LEVEL DB entry"""

    job_input = input_config_to_job_input(
        input_batch_id, job_params["jobName"], job_params["jobLevel"], job_params["inputConfig"]
    )

    if job_params["jobType"] == SmgtJobType.BATCH:
        trigger_batch_job(batch_id, job_input, job_params)


def lambda_handler(event, context):
    """Lambda function that ...

    Reads the S3 Input manifest, and sends the batch of the data to the SMGT Job.

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

    parent_batch_id = event["parent_batch_id"]
    job_level = event["job_level"]

    parent_batch = db.get_batch_metadata(parent_batch_id)
    if parent_batch is None:
        raise Exception(f"Invalid parent batch id: {parent_batch_id}")

    if job_level == 1:
        meta_data_type = BatchMetadataType.FIRST_LEVEL
    elif job_level == 2:
        meta_data_type = BatchMetadataType.SECOND_LEVEL
    elif job_level == 3:
        meta_data_type = BatchMetadataType.THIRD_LEVEL

    # Filter jobs by job level
    labeling_jobs = parent_batch[BatchMetadataTableAttributes.LABELING_JOBS]
    current_jobs = [job for job in labeling_jobs if job["jobLevel"] == job_level]
    log.logging.info("Kicking off %d jobs for level %d", len(current_jobs), job_level)

    batch_id = f"{parent_batch_id}-{meta_data_type.lower()}"
    for job in current_jobs:
        trigger_labeling_job(parent_batch_id, batch_id, job)

    try:
        db.insert_perform_labeling_job_metadata(
            parent_batch_id=parent_batch_id,
            batch_id=batch_id,
            batch_status=BatchStatus.IN_PROGRESS,
            batch_metadata_type=meta_data_type,
            num_children_batches=len(current_jobs),
        )
    except botocore.exceptions.ClientError as err:
        raise Exception(f"failed to put batch id {batch_id}") from err

    return {
        "batch_id": batch_id,
    }
