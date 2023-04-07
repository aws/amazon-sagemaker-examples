import json
import logging
import os
import boto3
import botocore
import ast
import time

logger = logging.getLogger()
logger.setLevel(os.getenv("LOGGING_LEVEL", logging.INFO))


def lambda_handler(event, context):
    """
    Manages SageMaker geospatial EOJs as required.
    """
    try:
        if "region" in event:
            region = event["region"]
        else:
            region = "us-west-2"
        geospatial_client = boto3.client("sagemaker-geospatial", region_name="us-west-2")

        if "eoj_name" in event:
            logger.debug(f"Creating EOJ from event:\n{json.dumps(event)}")
            if "RasterDataCollectionQuery" in str(event["eoj_input_config"]):
                # Input is a Raster Data Collection Query
                input_config = ast.literal_eval(event["eoj_input_config"])
            elif "arn" in str(event["eoj_input_config"]):
                # Input is chaining results of another EOJ
                input_config = {"PreviousEarthObservationJobArn": event["eoj_input_config"]}
            logger.info(f'Starting EOJ {event["eoj_name"]}')
            response = geospatial_client.start_earth_observation_job(
                Name=event["eoj_name"],
                ExecutionRoleArn=event["role"],
                InputConfig=input_config,
                JobConfig=ast.literal_eval(event["eoj_config"]),
            )
            logger.info(f'Started EOJ for {event["eoj_name"]} with eoj_arn: {response["Arn"]}\n')
            time.sleep(3)

        elif "eoj_export_config" in event:
            # Export an EOJ...
            logger.debug(f"Will export EOJ with event:\n{json.dumps(event)}")
            logger.info(f'Exporting EOJ with Arn {event["eoj_arn"]}')
            response = geospatial_client.export_earth_observation_job(
                Arn=event["eoj_arn"],
                ExecutionRoleArn=event["role"],
                OutputConfig=ast.literal_eval(event["eoj_export_config"]),
            )
            logger.info(f'Export eoj_arn: {response["Arn"]}\n')

        elif "Records" in event:
            # Check status of previous EOJ...
            logger.debug(f"Will check status of EOJ with event:\n{json.dumps(event)}")
            for record in event["Records"]:
                payload = json.loads(record["body"])
                token = payload["token"]
                eoj_arn = payload["arguments"]["eoj_arn"]
                logger.info(f"Check EOJ or export with ARN: {eoj_arn}")
                response = geospatial_client.get_earth_observation_job(Arn=eoj_arn)
                if response["Status"] == "COMPLETED":
                    # EOJ is COMPLETED
                    logger.info("EOJ completed, resuming pipeline...")
                    sagemaker = boto3.client("sagemaker", region_name=region)
                    sagemaker.send_pipeline_execution_step_success(
                        CallbackToken=token,
                        OutputParameters=[{"Name": "eoj_status", "Value": response["Status"]}],
                    )
                elif response["Status"] == "SUCCEEDED":
                    # Export of EOJ SUCCEEDED
                    logger.info("Export EOJ succeeded, resuming pipeline...")
                    sagemaker = boto3.client("sagemaker", region_name=region)
                    sagemaker.send_pipeline_execution_step_sucess(
                        CallbackToken=token,
                        OutputParameters=[
                            {"Name": "export_eoj_status", "Value": response["Status"]}
                        ],
                    )
                elif response["Status"] == "FAILED":
                    logger.info("EOJ or export failed, stopping pipeline...")
                    sagemaker = boto3.client("sagemaker", region_name=region)
                    sagemaker.send_pipeline_execution_step_failure(
                        CallbackToken=token, FailureReason=response["ErrorDetails"]
                    )
                else:
                    # EOJ is still running IN_PROGRESS, we must check again later
                    # Note we must raise an exception for having the message put back to the SNS queue
                    logger.info(f'EOJ or export with status: {response["Status"]}')
                    raise Exception("EOJ or export still running...")
    except botocore.exceptions.ClientError as e:
        error_msg = f"EOJ or export call failed: {e.response['Error']['Code']}, {e.response['Error']['Message']}"
        raise Exception(error_msg)

    try:
        response
    except NameError:
        response = None

    if response is not None:
        logger.info(f'eoj_arn: {response["Arn"]}\n')
    else:
        response = {"Arn": ""}

    return {"statusCode": 200, "eoj_arn": response["Arn"]}