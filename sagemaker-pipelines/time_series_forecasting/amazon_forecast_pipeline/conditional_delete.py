#!/usr/bin/env python

import os
from glob import glob
import argparse
import tarfile
import botocore.exceptions
import boto3
import json
import time

os.system("du -a /opt/ml")


def wait_till_delete(callback, check_time=5, timeout=150):
    """Only move to the next line of code once a delete has successfully occurred.

    Parameters:
        callback: deletion to execute
        check_time (int): number of seconds before checking again
        timeout (int): number of seconds after which a TimeoutError will be raised if deletion has not yet occurred
    """

    elapsed_time = 0
    while timeout is None or elapsed_time < timeout:
        try:
            out = callback()
        except botocore.exceptions.ClientError as e:
            # When given the resource not found exception, deletion has occured
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                print("Successful delete")
                return
            else:
                raise
        time.sleep(check_time)  # units of seconds
        elapsed_time += check_time

    raise TimeoutError("Forecast resource deletion timed-out.")


def delete_forecast_attributes(forecast, model_params):
    """Deletes forecast attributes.

    Parameters:
        forecast (boto3)
        model_params (dict): all model parameters passed from Training Step
    """
    if model_params["forecast_arn_predictor"] != None:
        wait_till_delete(
            lambda: forecast.delete_predictor(
                PredictorArn=model_params["forecast_arn_predictor"]
            )
        )

    for arn in ["target_import_job_arn", "related_import_job_arn"]:
        if model_params[arn] != None:
            wait_till_delete(
                lambda: forecast.delete_dataset_import_job(
                    DatasetImportJobArn=model_params[arn]
                )
            )

    for arn in ["target_dataset_arn", "related_dataset_arn"]:
        if model_params[arn] != None:
            wait_till_delete(
                lambda: forecast.delete_dataset(DatasetArn=model_params[arn])
            )

    if model_params["dataset_group_arn"] != None:
        wait_till_delete(
            lambda: forecast.delete_dataset_group(
                DatasetGroupArn=model_params["dataset_group_arn"]
            )
        )

    print("All attributes successfully deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str)
    parser.add_argument("--region", type=str)
    parser.add_argument("--maximum-score", type=float)
    args = parser.parse_args()
    print(boto3.__version__)

    metric = args.metric
    region = args.region
    maximum_score = args.maximum_score

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    print("Loading jsons.")
    with open("evaluation_metrics.json", "r") as f:
        eval_metrics = json.load(f)

    with open("model_parameters.json", "r") as f:
        model_params = json.load(f)

    if eval_metrics[metric] > maximum_score:
        session = boto3.Session(region_name=region)
        forecast = session.client(service_name="forecast")
        delete_forecast_attributes(forecast, model_params)

    else:
        print("Score is sufficient. Amazon Forecast resources will not be deleted.")
