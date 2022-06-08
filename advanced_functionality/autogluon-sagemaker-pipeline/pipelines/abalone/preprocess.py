"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/abalone-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    os.unlink(fn)

    logger.info("Raw data shape: %s", df.shape)
    logger.info("Writing full data for next steps")
    pathlib.Path(f"{base_dir}/full_data").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{base_dir}/full_data/full_data.csv", index=False)

    logger.info("full data: %s", df.head())

    logger.info("Splitting dataset")
    train_data, test = train_test_split(df, test_size=0.1, random_state=123)
    train, validation = train_test_split(train_data, test_size=0.1, random_state=123)

    logger.info("Raw data shape: %s", df.shape)
    logger.info("Writing full data for next steps")
    pathlib.Path(f"{base_dir}/full_data").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{base_dir}/full_data/full_data.csv", index=False)

    logger.info("Splitting dataset")
    train, test = train_test_split(df, test_size=0.1, random_state=123)

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
