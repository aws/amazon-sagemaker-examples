from darts.models import LinearRegressionModel
from darts.timeseries import TimeSeries
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging

def model_fn(model_dir, context):
    # model_path = os.path.join(model_dir, "model.pth")
    model = None
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model = LinearRegressionModel.load(f)
    return model

def input_fn(request_body, request_content_type, context):
    # return request_body
    record_dict = json.loads(request_body)
    print(record_dict)
    record_list = record_dict["instances"]
    # record = record_list[0]
    return record_list

def predict_fn(input_object, model, context):
    # return model.predict(10)
    start = input_object["start"]
    target = input_object["target"]
    dynamic_feat = input_object["dynamic_feat"]

    start_datetime = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    datetimes = [start_datetime + timedelta(minutes=i*10) for i in range(len(target))]
    target_df = pd.DataFrame({'target': target}, index=datetimes)
    target_ts = TimeSeries.from_dataframe(target_df, value_cols=['target'])

    past_cov_df = pd.DataFrame({f"feature_{i+1}": feats[:len(target)] for i, feats in enumerate(dynamic_feat)}, index=datetimes)
    past_cov_ts = TimeSeries.from_dataframe(past_cov_df)

    start_future_datetime = datetimes[-1] + timedelta(minutes=10)
    num_future_steps = len(dynamic_feat[0]) - len(target)
    future_datetimes = [start_future_datetime + timedelta(minutes=i*10) for i in range(num_future_steps)]

    # Create the DataFrame for future_covariates
    future_cov_df = pd.DataFrame({
        f"feature_{i+1}": feats[len(target):] for i, feats in enumerate(dynamic_feat)
    }, index=future_datetimes)
    future_cov_ts = TimeSeries.from_dataframe(future_cov_df)

    predictions = model.predict(10, series=target_ts, past_covariates=past_cov_ts, future_covariates=future_cov_ts)
    return predictions


def output_fn(prediction, content_type, context):
    predictions_list = [item for sublist in prediction.values() for item in sublist]
    logging.warning('----------------')
    
    # Create a dictionary in the desired format
    pred_dict = {'predictions': {'mean': predictions_list}}
    logging.warning(f'Predictions: {pred_dict}')
    # Convert the dictionary to a JSON string
    pred_json = json.dumps(pred_dict)
    return pred_json