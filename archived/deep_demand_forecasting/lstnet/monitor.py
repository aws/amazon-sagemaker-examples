import os
from os import path as osp
from pathlib import Path

import pandas as pd
from gluonts.dataset.util import to_pandas

from data import load_multivariate_datasets
from utils import get_logger


LOG_CONFIG = os.getenv("LOG_CONFIG_PATH", Path(osp.abspath(__file__)).parent / "log.ini")

logger = get_logger(LOG_CONFIG)


def multivar_df(ds):
    df = pd.DataFrame()
    for i in range(ds["target"].shape[0]):
        tmp = {}
        for k in ds:
            if k == "target":
                tmp["target"] = ds["target"][i]
            else:
                tmp[k] = ds[k]
        tmp_df = to_pandas(tmp).to_frame().rename(columns={0: f"ts_{i}"})
        df = pd.concat([df, tmp_df], axis=1, sort=True)

    return df.reset_index().rename(columns={"index": "time"})


def prepare_data(path: str):
    ds = load_multivariate_datasets(Path(path))
    train_ds = next(iter(ds.train))
    test_ds = next(iter(ds.test))
    logger.info(
        f"original train data shape {train_ds['target'].shape}, test data shape {test_ds['target'].shape}"
    )
    train_df = multivar_df(train_ds)
    test_df = multivar_df(test_ds)
    assert all(train_df.columns == test_df.columns)
    return train_df, test_df


def create_data_viz(train_df, test_df, context_length, prediction_length, num_sample_ts):
    num_sample_ts = min(num_sample_ts, train_df.shape[1])
    ts_col_names = list(train_df.columns)
    selected_cols = ts_col_names[1:num_sample_ts]

    selected_train_df = train_df.loc[:, ["time"] + selected_cols]
    train_df_melt = pd.melt(
        selected_train_df.tail(context_length),
        id_vars=["time"],
        value_vars=selected_cols,
    )
    train_df_melt.rename(columns={"variable": "covariate"}, inplace=True)
    selected_test_df = test_df.loc[:, ["time"] + selected_cols]
    num_train = selected_train_df.shape[0]
    test_df_melt = pd.melt(
        selected_test_df.iloc[num_train : num_train + prediction_length],
        id_vars=["time"],
        value_vars=selected_cols,
    )
    test_df_melt.rename(columns={"variable": "covariate"}, inplace=True)
    return train_df_melt, test_df_melt, selected_cols
