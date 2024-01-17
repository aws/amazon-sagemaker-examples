import argparse
import logging
import os
import json
import pathlib
import requests
import tempfile

import boto3

import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--default_bucket', type=str, default="")
    parser.add_argument('--model_package_group_name', type=str, default="")

    args, _ = parser.parse_known_args()
    
    default_bucket = args.default_bucket
    model_package_group_name = args.model_package_group_name
    
    client = boto3.client(service_name="sagemaker", region_name="us-east-1")
    runtime = boto3.client(service_name="sagemaker-runtime", region_name="us-east-1")
    s3_client = boto3.client(service_name='s3', region_name="us-east-1")

    
    model_package_arn = client.list_model_packages(ModelPackageGroupName=model_package_group_name)[
        "ModelPackageSummaryList"
    ][0]["ModelPackageArn"]
    model_package_arn = "arn:aws:sagemaker:us-east-1:495659485974:model-package/model-monitor-clarify-group/1"
    
    model_package_metrics = client.describe_model_package(ModelPackageName=model_package_arn)["ModelMetrics"]

    model_quality_s3_key = model_package_metrics["ModelQuality"]["Statistics"]["S3Uri"].split(f"{default_bucket}/")[1]
    model_quality_bias = model_package_metrics["Bias"]
    model_quality_pretrain_bias_key = model_quality_bias["PreTrainingReport"]["S3Uri"].split(f"{default_bucket}/")[1]
    model_quality__post_train_bias_key = model_quality_bias["PostTrainingReport"]["S3Uri"].split(f"{default_bucket}/")[1]
    model_explainability_s3_key = model_package_metrics["Explainability"]["Report"]["S3Uri"].split(f"{default_bucket}/")[1]
    
    # MODEL QUALITY
    s3_obj = s3_client.get_object(Bucket=default_bucket, Key=model_quality_s3_key)
    s3_obj_data = s3_obj['Body'].read().decode('utf-8')
    model_quality_json = json.loads(s3_obj_data)
    mae = model_quality_json["regression_metrics"]["mae"]["value"]
    print(f"Mean Absolute Error: {mae}")
    
    mae_threshold = 1.5
    mae_check = True if mae < mae_threshold else False
    
    # MODEL QUALITY PRETRAINIG BIAS
    s3_obj = s3_client.get_object(Bucket=default_bucket, Key=model_quality_pretrain_bias_key)
    s3_obj_data = s3_obj['Body'].read().decode('utf-8')
    model_quality_pretrain_bias_json = json.loads(s3_obj_data)
    kullback_liebler = model_quality_pretrain_bias_json["pre_training_bias_metrics"][
        "facets"]["column_8"][0]["metrics"][4]["value"]
    print(f"Kullback Liebler: {kullback_liebler}")
    
    kullback_liebler_check_threshold = 0.5
    kullback_liebler_check = True if kullback_liebler < kullback_liebler_check_threshold else False
    
    # MODEL QUALITY POSTTRAINING BIAS
    s3_obj = s3_client.get_object(Bucket=default_bucket, Key=model_quality__post_train_bias_key)
    s3_obj_data = s3_obj['Body'].read().decode('utf-8')
    model_quality__post_train_bias_json = json.loads(s3_obj_data)
    treatment_equity = model_quality__post_train_bias_json["post_training_bias_metrics"][
        "facets"]["column_8"][0]["metrics"][-1]["value"]
    print(f"Treatment Equity: {treatment_equity}")
    
    treatment_equity_check_threshold = 0
    treatment_equity_check = True if treatment_equity < treatment_equity_check_threshold else False
    
    # MODEL EXPLAINABILITY REPORT
    s3_obj = s3_client.get_object(Bucket=default_bucket, Key=model_explainability_s3_key)
    s3_obj_data = s3_obj['Body'].read().decode('utf-8')
    model_explainability_s3_json = json.loads(s3_obj_data)
    age_shap_label_0 = model_explainability_s3_json[
        "explanations"]["kernel_shap"]["label0"]["global_shap_values"]["column_1"]
    print(f"Age Shap Value: {age_shap_label_0}")
    
    age_shap_label_0_threshold = 0.5
    age_shap_label_0_check = True if age_shap_label_0 < age_shap_label_0_threshold else False
    
    checks = {
        "mae": mae_check,
        "kullback_liebler": kullback_liebler_check,
        "treatment_equity": treatment_equity_check,
        "age_shap_label_0": age_shap_label_0_check
    }
    
    with open('/opt/ml/processing/output/checks.json', 'w') as fp:
        json.dump(checks, fp)
        
    print("Done!")

    