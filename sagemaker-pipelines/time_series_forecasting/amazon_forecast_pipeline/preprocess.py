#!/usr/bin/env python

import os
import pandas as pd
from glob import glob
import argparse

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
    df = pd.read_csv(source_train_ts, index_col=0, parse_dates=True)
    df = df.resample("H").sum() / 4
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "datetime", "MT_001": "kw"})

    # Use 2.5 weeks of hourly data to train Amazon Forecast. This is to save costs in generating the forecast.
    df = df[-2 * 7 * 24 - 24 * 3 :].copy()
    df["kw"] = df["kw"].astype("float")
    df["workingday"] = df["datetime"].dt.weekday.apply(lambda x: 1 if x < 5 else 0).astype("float")
    df["item_id"] = "client_1"
    target_df = df[["item_id", "datetime", "kw"]][:-forecast_horizon]
    rts_df = df[["item_id", "datetime", "workingday"]]

    return target_df, rts_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast_horizon", type=str)
    args = parser.parse_args()

    forecast_horizon = int(args.forecast_horizon)
    target_df, rts_df = create_dataframes(forecast_horizon, SRC_TS)

    print(f"{len(target_df)} + {forecast_horizon} = {len(rts_df)}")

    # Assert equivalent lengths of dataframes. If no equivalence, a predictor cannot be created.
    assert len(target_df) + forecast_horizon == len(rts_df), "length doesn't match"

    # Assert that the related timeseries is not missing entries. If it is, a predictor cannot be created.
    assert len(rts_df) == len(
        pd.date_range(
            start=list(rts_df["datetime"])[0],
            end=list(rts_df["datetime"])[-1],
            freq="H",
        )
    ), "missing entries in the related time series"

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
