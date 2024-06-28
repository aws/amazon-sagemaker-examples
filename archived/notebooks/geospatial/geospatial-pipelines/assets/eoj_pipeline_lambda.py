import json
import logging
import os
import boto3
import botocore
import ast
import time

logger = logging.getLogger()
logger.setLevel(os.getenv("LOGGING_LEVEL", logging.INFO))

geospatial_client = boto3.client("sagemaker-geospatial", region_name="us-west-2")
sagemaker_clients_by_region = {}

def get_sagemaker_client(region):
    if region not in sagemaker_clients_by_region:
        sagemaker_clients_by_region[region] = boto3.client("sagemaker", region_name=region)
    return sagemaker_clients_by_region[region]
    
def start_eoj(event):
    logger.debug(f"Creating EOJ from event:\n{json.dumps(event)}")
    if "RasterDataCollectionQuery" in str(event["eoj_input_config"]):
        # Input is a Raster Data Collection Query
        input_config = ast.literal_eval(event["eoj_input_config"])
    elif "arn" in str(event["eoj_input_config"]):
        # Input is chaining results of another EOJ
        input_config = {"PreviousEarthObservationJobArn": event["eoj_input_config"]}
    logger.info("Starting EOJ {}".format(event["eoj_name"]))
    response = geospatial_client.start_earth_observation_job(
        Name=event["eoj_name"],
        ExecutionRoleArn=event["role"],
        InputConfig=input_config,
        JobConfig=ast.literal_eval(event["eoj_config"]),
    )
    logger.info("Started EOJ for {} with ARN: {}".format(event["eoj_name"], response["Arn"]))
    time.sleep(3)
    return response

def export_eoj(event):
    logger.debug(f"Will export EOJ with event:\n{json.dumps(event)}")
    logger.info("Exporting EOJ with ARN: {}".format(event["eoj_arn"]))
    response = geospatial_client.export_earth_observation_job(
        Arn=event["eoj_arn"],
        ExecutionRoleArn=event["role"],
        OutputConfig=ast.literal_eval(event["eoj_export_config"]),
    )
    logger.info("Started EOJ export with ARN: {}".format(event["eoj_arn"]))
    return response

def handle_sqs_callback_trigger(event, sagemaker_client):
    logger.debug(f"Will check status of EOJ with event:\n{json.dumps(event)}")
    for record in event["Records"]:
        payload = json.loads(record["body"])
        token = payload["token"]
        eoj_arn = payload["arguments"]["eoj_arn"]
        logger.info("Check EOJ or export with ARN: {}".format(eoj_arn))
        response = geospatial_client.get_earth_observation_job(Arn=eoj_arn)
        if response["Status"] == "COMPLETED":
            # EOJ is COMPLETED
            logger.info("EOJ completed, resuming pipeline...")
            sagemaker_client.send_pipeline_execution_step_success(
                CallbackToken=token,
                OutputParameters=[{"Name": "eoj_status", "Value": response["Status"]}],
            )
        elif response["Status"] == "SUCCEEDED":
            # Export of EOJ SUCCEEDED
            logger.info("Export EOJ succeeded, resuming pipeline...")
            sagemaker_client.send_pipeline_execution_step_sucess(
                CallbackToken=token,
                OutputParameters=[
                    {"Name": "export_eoj_status", "Value": response["Status"]}
                ],
            )
        elif response["Status"] == "FAILED":
            logger.info("EOJ or export failed, stopping pipeline...")
            sagemaker_client.send_pipeline_execution_step_failure(
                CallbackToken=token, FailureReason=response["ErrorDetails"]
            )
        else:
            # EOJ is still running IN_PROGRESS, we must check again later
            # Note we must raise an exception for having the message put back to the SNS queue
            logger.info("EOJ or export with status: {}".format(response["Status"]))
            raise RuntimeError("EOJ or export still running...")
        return response

def lambda_handler(event, context):
    """
    Manages SageMaker geospatial EOJs as required.
    """
    try:
        if "region" in event:
            sagemaker_client = get_sagemaker_client(event["region"])
        else:
            sagemaker_client = get_sagemaker_client("us-west-2")

        if "eoj_name" in event:
            response = start_eoj(event)
        elif "eoj_export_config" in event:
            response = export_eoj(event)
        elif "Records" in event:
            response = handle_sqs_callback_trigger(event, sagemaker_client)
    except botocore.exceptions.ClientError as e:
        error_msg = f"EOJ or export call failed: {e.response['Error']['Code']}, {e.response['Error']['Message']}"
        raise RuntimeError(error_msg)

    try:
        response
    except NameError:
        response = None

    if response is not None:
        logger.info("EOJ ARN: {}".format(response["Arn"]))
    else:
        response = {"Arn": ""}

    return {"statusCode": 200, "eoj_arn": response["Arn"]}