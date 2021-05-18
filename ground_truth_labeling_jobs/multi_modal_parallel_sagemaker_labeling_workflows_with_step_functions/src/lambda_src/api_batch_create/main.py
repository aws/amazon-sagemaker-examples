"""
API handler for batch creation.

Handles starting a new step function execution for the batch
and validating input.
"""

import json
import os
import re
import uuid

import boto3
import botocore
from shared import db, label_arn, log
from shared.constants import SmgtJobType

batch_step_function_arn = os.getenv("BATCH_CREATION_STEP_FUNCTION_ARN")
sagemaker = boto3.client("sagemaker")


def validate_regex(name):
    """Validate regex used for names/ids compatible with SMGT"""
    return re.match("^[a-zA-Z0-9_-]*$", name)


def validate_common_body(body):
    """Verify main batch fields exist"""
    batch_id = body.get("batchId")

    if batch_id is None:
        error_message = "batchId must be provided"
        return error_message

    if not validate_regex(batch_id):
        return "batchId can only contain Lower case Alphanumeric, '-'"

    batch_metadata = db.get_batch_metadata(batch_id)
    if batch_metadata is not None:
        error_message = f"Provided batchId already exists : {batch_id} : {batch_metadata}"
        return error_message

    return None


def execute_step_function(step_function_arn, batch_id, execution_input_payload):
    """Start a step function execution"""
    step_function = boto3.client("stepfunctions")

    try:
        name = f"{batch_id}-{str(uuid.uuid4())[:8]}"
        step_function.start_execution(
            stateMachineArn=step_function_arn,
            name=name,
            input=execution_input_payload,
        )
    except botocore.exceptions.ClientError as error:
        log.logger.error(
            "Step function execution failed with executionId : %s" " input : %s due to %s",
            name,
            execution_input_payload,
            error,
        )
        return {
            "statusCode": 500,
            "body": "Error: Internal Service Failure",
        }

    response = {
        "statusCode": 200,
        "body": f"Batch processor successfully triggered with BatchId : {batch_id}",
        "isBase64Encoded": False,
    }
    return response


def validate_attributes_exist(obj, attributes):
    """Returns validation error string if not all attributes present"""
    for attribute in attributes:
        if not obj.get(attribute):
            return f"Missing '{attribute}' from {obj}"
    return None


def validate_smgt_job_name(job_name):
    """Validates we can use this job name, it matches regex and doesn't exist"""
    if not validate_regex(job_name):
        return "jobName failed regex check, must be lower case, '-'"
    return None


def validate_smgt_job_doesnt_exist(job_name):
    """Validate that there is no job with the given name"""
    exists = True
    try:
        sagemaker.describe_labeling_job(LabelingJobName=job_name)
    except botocore.exceptions.ClientError:
        # TODO: find more specific exception for resource not found
        exists = False

    if exists:
        return f"GroundTruth job with name {job_name} already exists"

    return None


def validate_input_config(config):
    """Validate that the input source is valid"""

    input_manifest_s3_uri = config.get("inputManifestS3Uri")
    chain_from_job_name = config.get("chainFromJobName")

    if input_manifest_s3_uri and not chain_from_job_name:
        return None
    if not input_manifest_s3_uri and chain_from_job_name:
        return None

    return "Must specify single 'inputManifestS3Uri' or 'chainFromJobName' as input config"


def validate_job_common(job):
    """Validate common fields for batch jobs"""

    attr_error = validate_attributes_exist(
        job,
        [
            "jobName",
            "jobLevel",
            "inputConfig",
        ],
    )
    if attr_error:
        return attr_error

    if job.get("jobLevel") not in [1, 2, 3]:
        return "jobLevel must be 1 or 2 or 3"

    input_config_error = validate_input_config(job.get("inputConfig"))
    if input_config_error:
        return input_config_error

    name_error = validate_smgt_job_name(job.get("jobName"))
    if name_error:
        return name_error

    return None


def validate_batch_job_type(job):
    """Validate a batch job type"""
    common_error = validate_job_common(job)
    if common_error:
        return common_error

    attr_error = validate_attributes_exist(
        job,
        [
            "jobModality",
        ],
    )
    if attr_error:
        return attr_error

    if not label_arn.JobModality.is_member(job.get("jobModality")):
        return f"jobModality must be in list: {label_arn.JobModality}"

    if (
        not job.get("labelCategoryConfigS3Uri")
        and job["inputConfig"].get("chainFromJobName") is None
    ):
        return f"Must provide label category config file if not chaining job: {job['jobName']}"

    exists_error = validate_smgt_job_doesnt_exist(job["jobName"])
    if exists_error:
        return exists_error

    return None


def validate_job_input(job):
    """Validates a single labeling job entry"""

    job_type = job.get("jobType")
    if job_type == SmgtJobType.BATCH:
        return validate_batch_job_type(job)

    return None


def validate_job_dependencies(labeling_jobs):
    """Check that we are only chaining from first level jobs"""
    jobs_by_name = {job["jobName"]: job for job in labeling_jobs}

    for job in labeling_jobs:
        job_name = job["jobName"]
        if "chainFromJobName" not in job["inputConfig"]:
            continue

        chain_job_name = job["inputConfig"]["chainFromJobName"]

        # We can later relax this and allow chaining from jobs that already exist in SMGT.
        if chain_job_name not in jobs_by_name:
            return f"Job {job_name} can't chain from {chain_job_name}, chain job name unknown"

        chain_job = jobs_by_name[chain_job_name]

        job_level, chain_job_level = job["jobLevel"], chain_job["jobLevel"]

        if job_level <= chain_job_level:
            return (
                f"Job {job_name} can't chain from {chain_job_name}, incorrect job levels: "
                f"job level {job_level}, chain from job level {chain_job_level}",
            )

    return None


def add_missing_keys(obj, default_args):
    """Add any default values that don't exist in the original argument"""
    for key, value in default_args.items():
        if key not in obj:
            obj[key] = value


def add_defaults(body):
    """Adds any defaults to the validated input request"""
    common_defaults = {
        "maxConcurrentTaskCount": 100,
    }

    batch_job_defaults = {
        **common_defaults,
        "workteamArn": os.environ["DEFAULT_WORKTEAM_ARN"],
        "taskAvailabilityLifetimeInSeconds": 864000,
        "taskTimeLimitInSeconds": 604800,
    }

    for labeling_job in body["labelingJobs"]:
        if labeling_job["jobType"] == SmgtJobType.BATCH:
            add_missing_keys(labeling_job, batch_job_defaults)


def validate_multi_job_input(body):
    """Validate all fields we expect are in the body and that inputs are valid"""
    error = validate_common_body(body)
    if error:
        return error

    labeling_jobs_key = "labelingJobs"
    labeling_jobs = body.get(labeling_jobs_key)
    if not labeling_jobs:
        return f"Must provide '{labeling_jobs_key}' key in body"

    errors = [validate_job_input(job) for job in labeling_jobs]
    errors = [error for error in errors if error]
    if errors:
        return "\n".join(errors)

    dependency_error = validate_job_dependencies(labeling_jobs)
    if dependency_error:
        return dependency_error

    return None


def execute_multi_job(body):
    """Start a multi-job batch step function execution"""

    # Perform validation on the input.
    validation_error_message = validate_multi_job_input(body)
    if validation_error_message:
        return {
            "statusCode": 400,
            "body": f"Error: {validation_error_message}",
        }

    batch_id = body.get("batchId")

    # Add any non-mandatory variables to the labeling jobs.
    add_defaults(body)

    step_function_payload = body

    log.logger.info("Triggering Step Function with inputs %s", step_function_payload)
    return execute_step_function(
        batch_step_function_arn, batch_id, json.dumps(step_function_payload)
    )


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
    log.log_request_and_context(event, context)

    body = json.loads(event.get("body"))
    return execute_multi_job(body)
