# Evaluation script for measure customizable metrics

import json
from datetime import datetime
import boto3
from time import sleep
import os
import argparse

alogorithms_arn_dict = {
    "CNNQR": "arn:aws:forecast:::algorithm/CNN-QR",
    "NPTS": "arn:aws:forecast:::algorithm/NPTS",
    "DEEPAR": "arn:aws:forecast:::algorithm/Deep_AR_Plus",
    "Prophet": "arn:aws:forecast:::algorithm/Prophet",
    "ARIMA": "arn:aws:forecast:::algorithm/ARIMA",
    "ETS": "arn:aws:forecast:::algorithm/ETS",
}

target_schema = {
    "Attributes": [
        {"AttributeName": "item_id", "AttributeType": "string"},
        {"AttributeName": "timestamp", "AttributeType": "timestamp"},
        {"AttributeName": "target_value", "AttributeType": "float"},
    ]
}

related_schema = {
    "Attributes": [
        {"AttributeName": "item_id", "AttributeType": "string"},
        {"AttributeName": "timestamp", "AttributeType": "timestamp"},
        {"AttributeName": "workingday", "AttributeType": "float"},
    ]
}

# Can be optionally specified. Not included in this example.
meta_schema = None


def create_dataset_group_and_datasets(
    region,
    dataset_group_name,
    frequency,
    target_schema,
    related_schema=None,
    meta_schema=None,
):
    """Create a dataset group and datasets within the group.

    Parameters:
        dataset_group_name (str): the name of the Dataset group
        frequency (str): frequency of the samples, 'M' for monthly e.g.
        target_schema (dict): schema of the target dataset
        related_schema (dict): schema of the related features dataset
        meta_schema (dict): schema of the meta dataset

    Returns:
        forecast: boto3 client
        dataset_group_arn (str): the ARN of the Dataset group
        target_dataset_arn (str): the ARN of the Target Dataset
        related_dataset_arn (str): the ARN of the Related Features Dataset
        meta_dataset_arn (str): the ARN of the Meta Dataset
    """
    forecast = boto3.client(service_name="forecast", region_name=region)

    # Create Datasetgroup
    create_dataset_group_response = forecast.create_dataset_group(
        DatasetGroupName=dataset_group_name, Domain="CUSTOM"
    )
    dataset_group_arn = create_dataset_group_response["DatasetGroupArn"]
    print(dataset_group_arn)

    # Create Target Dataset
    target_response = forecast.create_dataset(
        Domain="CUSTOM",
        DatasetType="TARGET_TIME_SERIES",
        DatasetName=dataset_group_name + "_target",
        DataFrequency=frequency,
        Schema=target_schema,
    )
    target_dataset_arn = target_response["DatasetArn"]
    print(target_dataset_arn)

    # Create Related Dataset
    if related_schema:
        related_response = forecast.create_dataset(
            Domain="CUSTOM",
            DatasetType="RELATED_TIME_SERIES",
            DatasetName=dataset_group_name + "_related",
            DataFrequency=frequency,
            Schema=related_schema,
        )
        related_dataset_arn = related_response["DatasetArn"]
    else:
        related_dataset_arn = None
    print(related_dataset_arn)

    # Create Meta Dataset
    if meta_schema:
        meta_response = forecast.create_dataset(
            Domain="CUSTOM",
            DatasetType="ITEM_METADATA",
            DatasetName=dataset_group_name + "_meta",
            Schema=meta_schema,
        )
        meta_dataset_arn = meta_response["DatasetArn"]
    else:
        meta_dataset_arn = None
    print(meta_dataset_arn)

    # Attach the Dataset to the Dataset Group
    datasets_arns = [target_dataset_arn]
    if related_dataset_arn:
        datasets_arns.append(related_dataset_arn)
    if meta_dataset_arn:
        datasets_arns.append(meta_dataset_arn)
    forecast.update_dataset_group(
        DatasetGroupArn=dataset_group_arn, DatasetArns=datasets_arns
    )

    return (
        forecast,
        dataset_group_arn,
        target_dataset_arn,
        related_dataset_arn,
        meta_dataset_arn,
    )


def check_import_status(forecast, import_job_arn):
    """Check dataset import job status

    Parameters:
       forecast: boto3 client
       import_job_arn (str): the ARN of the Dataset Import Job
    """
    for i in range(100):
        dataImportStatus = forecast.describe_dataset_import_job(
            DatasetImportJobArn=import_job_arn
        )["Status"]
        print(dataImportStatus)
        if dataImportStatus != "ACTIVE" and dataImportStatus != "CREATE_FAILED":
            sleep(30)
        else:
            break


def check_training_status(forecast, forecast_arn_predictor):
    """Check dataset import job status

    Parameters:
       forecast: boto3 client
       forecast_arn_predictor (str): the ARN of the Amazon Forecast predictor
    """
    for i in range(300):
        data_import_status = forecast.describe_predictor(
            PredictorArn=forecast_arn_predictor
        )["Status"]
        print(data_import_status)
        if data_import_status != "ACTIVE" and data_import_status != "CREATE_FAILED":
            sleep(60)
        else:
            break


def import_time_series_from_s3(
    forecast,
    dataset_group_name,
    ts_dataset_arn,
    ts_s3_data_path,
    role_arn,
    timestamp_format,
):
    """Import target timeseries from S3

     Parameters:
        forecast: boto3 client
        dataset_group_name (str): the name of the Dataset group
        ts_dataset_arn (str): the ARN of the time series
        ts_s3_data_path (str): S3 URI of the time series
        role_arn (str): the ARN of the role that will be used the import job
        ts_s3_data_path (str): the format of the timestamp in the time series
    Return:
        ts_import_job_arn (str): the ARN of the import job
    """
    ts_import_job_response = forecast.create_dataset_import_job(
        DatasetImportJobName=dataset_group_name,
        DatasetArn=ts_dataset_arn,
        DataSource={"S3Config": {"Path": ts_s3_data_path, "RoleArn": role_arn}},
        TimestampFormat=timestamp_format,
    )
    ts_import_job_arn = ts_import_job_response["DatasetImportJobArn"]
    check_import_status(forecast, ts_import_job_arn)
    return ts_import_job_arn


if __name__ == "__main__":
    print(boto3.__version__)
    # Parsing the required arguments for the training and evaluation job
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--forecast_horizon", type=str)
    parser.add_argument("--forecast_algorithm", type=str)
    parser.add_argument("--dataset_frequency", type=str)
    parser.add_argument("--timestamp_format", type=str)
    parser.add_argument("--number_of_backtest_windows", type=str)
    parser.add_argument("--s3_directory_target", type=str)
    parser.add_argument("--s3_directory_related", type=str)
    parser.add_argument("--role_arn", type=str)
    parser.add_argument("--region", type=str)
    args = parser.parse_args()

    role_arn = args.role_arn

    region = args.region
    forecast_horizon = int(args.forecast_horizon)
    back_test_window_offsets = forecast_horizon
    forecast_algorithm = args.forecast_algorithm
    # What is your forecast time unit granularity?
    # Choices are: ^Y|M|W|D|H|30min|15min|10min|5min|1min$
    dataset_frequency = args.dataset_frequency
    timestamp_format = args.timestamp_format
    number_of_backtest_windows = int(args.number_of_backtest_windows)

    target_s3_data_path = os.path.join(args.s3_directory_target, "target.csv")
    related_s3_data_path = os.path.join(args.s3_directory_related, "related.csv")

    print("Create Dataset group")
    training_date = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
    dataset_group_name = f"MLOPS_Pipeline_{training_date}"

    (
        forecast,
        dataset_group_arn,
        target_dataset_arn,
        related_dataset_arn,
        meta_dataset_arn,
    ) = create_dataset_group_and_datasets(
        region,
        dataset_group_name,
        dataset_frequency,
        target_schema,
        related_schema,
        meta_schema,
    )

    print("Import Target and Related TimeSeries from S3 to Forecast")
    target_import_job_arn = import_time_series_from_s3(
        forecast,
        dataset_group_name,
        target_dataset_arn,
        target_s3_data_path,
        role_arn,
        timestamp_format,
    )

    related_import_job_arn = import_time_series_from_s3(
        forecast,
        dataset_group_name,
        related_dataset_arn,
        related_s3_data_path,
        role_arn,
        timestamp_format,
    )

    print("Train Predictor: ", forecast_algorithm)
    predictor_name = dataset_group_name + "_" + forecast_algorithm
    create_predictor_response = forecast.create_predictor(
        PredictorName=predictor_name,
        AlgorithmArn=alogorithms_arn_dict[forecast_algorithm],
        ForecastHorizon=forecast_horizon,
        PerformAutoML=False,
        PerformHPO=False,
        EvaluationParameters={
            "NumberOfBacktestWindows": number_of_backtest_windows,
            "BackTestWindowOffset": back_test_window_offsets,
        },
        InputDataConfig={"DatasetGroupArn": dataset_group_arn},
        FeaturizationConfig={"ForecastFrequency": dataset_frequency},
    )

    forecast_arn_predictor = create_predictor_response["PredictorArn"]

    print("Forecast Predictor ARN: ", forecast_arn_predictor)

    check_training_status(forecast, forecast_arn_predictor)

    print("Training completed \n Saving the resources ARN")

    forecast_details = {
        "dataset_group_arn": dataset_group_arn,
        "dataset_group_name": dataset_group_name,
        "target_dataset_arn": target_dataset_arn,
        "target_import_job_arn": target_import_job_arn,
        "related_dataset_arn": related_dataset_arn,
        "related_import_job_arn": related_import_job_arn,
        "meta_dataset_arn": meta_dataset_arn,
        "alogrithm_name": forecast_algorithm,
        "alogorithms_arn_dict": alogorithms_arn_dict[forecast_algorithm],
        "forecast_arn_predictor": forecast_arn_predictor,
        "role_arn": role_arn,
    }

    with open(os.path.join(args.model_dir, "model_parameters.json"), "w") as f:
        f.write(json.dumps(forecast_details))

    print("Model Evaluation")
    response = forecast.get_accuracy_metrics(PredictorArn=forecast_arn_predictor)
    evaluation_metrics = response["PredictorEvaluationResults"][0]["TestWindows"][0][
        "Metrics"
    ]["ErrorMetrics"][0]

    # Print metrics so that they are picked up by SageMaker
    print("WAPE={};".format(evaluation_metrics["WAPE"]))
    print("RMSE={};".format(evaluation_metrics["RMSE"]))
    print("MASE={};".format(evaluation_metrics["MASE"]))
    print("MAPE={};".format(evaluation_metrics["MAPE"]))

    with open(os.path.join(args.model_dir, "evaluation_metrics.json"), "w") as f:
        f.write(json.dumps(evaluation_metrics))
