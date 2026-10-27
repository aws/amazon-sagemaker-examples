import boto3
import json
import logging

sagemaker_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    try:
        payload = json.loads(event["Records"][0]["body"])
        callback_token = payload["token"]
        autopilot_job = sagemaker_client.describe_auto_ml_job(
            AutoMLJobName=payload["arguments"]["AutopilotJobName"]
        )
        autopilot_job_status = autopilot_job["AutoMLJobStatus"]
        if autopilot_job_status == "Completed":
            sagemaker_client.send_pipeline_execution_step_success(
                CallbackToken=callback_token
            )
        elif autopilot_job_status in ["InProgress", "Stopping"]:
            raise ValueError("Autopilot training not finished yet. Retrying later...")
        else:
            sagemaker_client.send_pipeline_execution_step_failure(
                CallbackToken=callback_token,
                FailureReason=autopilot_job.get(
                    "FailureReason",
                    f"Autopilot training job (status: {autopilot_job_status}) failed to finish.",
                ),
            )
    except ValueError:
        raise
    except Exception as e:
        logging.exception(e)
        sagemaker_client.send_pipeline_execution_step_failure(
            CallbackToken=callback_token,
            FailureReason=str(e),
        )
