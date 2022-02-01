#!/usr/bin/env python

import os
import pandas as pd
from glob import glob
import argparse
import boto3
import botocore
os.system("du -a /opt/ml")

SRC_TS = glob("/opt/ml/processing/input_train/*.csv")[0]
print(SRC_TS)

DST_TRAIN_TS = "/opt/ml/processing/target/target.csv"
DST_RELATED_TS = "/opt/ml/processing/related/related.csv"

def create_dataframes(forecast_horizon, source_train_ts):
    """Create the target and related dataframe in a suitable format for Amazon Forecast.
    
    Parameters:
        forecast_horizon (int): number of time units you want to forecast
        source_train_ts (str): location of train.csv

    Returns: 
        target_df (pd.DataFrame): target dataframe in Forecast format
        rts_df (pd.DataFrame): related dataframe in Forecast format
    """
    bike_df = pd.read_csv(source_train_ts, dtype = object)
    
    # Use 2.5 weeks of hourly data to train Amazon Forecast. This is to save costs in generating the forecast.
    bike_df = bike_df[-2*7*24-24*3:].copy()
    bike_df['count'] = bike_df['count'].astype('float')
    bike_df['workingday'] = bike_df['workingday'].astype('float')
    bike_df['item_id'] = "bike_12"
    
    target_df = bike_df[['item_id', 'datetime', 'count']][:-forecast_horizon]
    rts_df = bike_df[['item_id', 'datetime', 'workingday']]
    
    return target_df, rts_df


if __name__ == "__main__":
    print(boto3.__version__)
    print(botocore.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast_horizon', type=str)
    args = parser.parse_args()
    
    forecast_horizon = int(args.forecast_horizon)
    target_df, rts_df = create_dataframes(forecast_horizon, SRC_TS)
    
    print(f"{len(target_df)} + {forecast_horizon} = {len(rts_df)}")
    
    # Assert equivalent lengths of dataframes. If no equivalence, a predictor cannot be created.
    assert len(target_df) + forecast_horizon == len(rts_df), "length doesn't match"
    
    # Assert that the related timeseries is not missing entries. If it is, a predictor cannot be created.
    assert len(rts_df) == len(pd.date_range(
        start=list(rts_df['datetime'])[0],
        end=list(rts_df['datetime'])[-1],
        freq='H'
    )), "missing entries in the related time series"
    
    
    # Writing both dataframes to a csv file.
    target_df.to_csv(
        path_or_buf=DST_TRAIN_TS,
        header=False,
        index=False,
    )

    rts_df.to_csv(
        path_or_buf=DST_RELATED_TS,
        header=False,
        index=False,
    )
    
    