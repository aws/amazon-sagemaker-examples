%%writefile preprocess.py
from __future__ import print_function, unicode_literals
import argparse
import json
import logging
import os
import sys
import time
import warnings
import boto3
import numpy as np
import pandas as pd
from tornado import gen
import dask.dataframe as dd
import joblib
from dask.distributed import Client
from sklearn.compose import make_column_transformer
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (KBinsDiscretizer, LabelBinarizer,
                                   OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
s3_client = boto3.resource("s3")
attempts_counter = 3
attempts = 0

def upload_objects(bucket, prefix, local_path):
    try:
        bucket_name = bucket  # s3 bucket name
        root_path = local_path  # local folder for upload

        s3_bucket = s3_client.Bucket(bucket_name)

        for path, subdirs, files in os.walk(root_path):
            for file in files:
                s3_bucket.upload_file(
                    os.path.join(path, file), prefix + "/output/" + file
                )
    except Exception as err:
        logging.exception(err)

def print_shape(df):
    negative_examples, positive_examples = np.bincount(df["income"])
    print(
        "Data shape: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    # Get processor scrip arguments
    args_iter = iter(sys.argv[1:])
    script_args = dict(zip(args_iter, args_iter))
    scheduler_ip = sys.argv[-1]

    # Start the Dask cluster client
    try:
        client = Client("tcp://" + str(scheduler_ip) + ":8786")
        logging.info("Printing cluster information: {}".format(client))
    except Exception as err:
        logging.exception(err)
        
    columns = [
        "age",
        "education",
        "major industry code",
        "class of worker",
        "num persons worked for employer",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "income",
    ]
    class_labels = [" - 50000.", " 50000+."]
    input_data_path = "s3://" + os.path.join(
        script_args["s3_input_bucket"],
        script_args["s3_input_key_prefix"],
        "census-income.csv",
    )
    print(input_data_path)

    # Creating the necessary paths to save the output files
    if not os.path.exists("/opt/ml/processing/train"):
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/test")

    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df, columns=columns)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.replace(class_labels, [0, 1], inplace=True)

    negative_examples, positive_examples = np.bincount(df["income"])
    print(
        "Data after cleaning: {}, {} positive examples, {} negative examples".format(
            df.shape, positive_examples, negative_examples
        )
    )

    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("income", axis=1), df["income"], test_size=split_ratio, random_state=0
    )

    preprocess = make_column_transformer(
        (
            KBinsDiscretizer(encode="onehot-dense", n_bins=2),
            ["age", "num persons worked for employer"],
        ),
        (
            StandardScaler(),
            ["capital gains", "capital losses", "dividends from stocks"],
        ),
        (
            OneHotEncoder(sparse=False),
            ["education", "major industry code", "class of worker"],
        ),
    )

    print("Running preprocessing and feature engineering transformations in Dask")
    with joblib.parallel_backend("dask"):
        train_features = preprocess.fit_transform(X_train)
        test_features = preprocess.transform(X_test)

    print("Train data shape after preprocessing: {}".format(train_features.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))

    train_features_output_path = os.path.join(
        "/opt/ml/processing/train", "train_features.csv"
    )
    train_labels_output_path = os.path.join(
        "/opt/ml/processing/train", "train_labels.csv"
    )

    test_features_output_path = os.path.join(
        "/opt/ml/processing/test", "test_features.csv"
    )
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test_labels.csv")

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(
        train_features_output_path, header=False, index=False
    )

    print("Saving test features to {}".format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(
        test_features_output_path, header=False, index=False
    )

    print("Saving training labels to {}".format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)

    print("Saving test labels to {}".format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)
    upload_objects(
        script_args["s3_output_bucket"],
        script_args["s3_output_key_prefix"],
        "/opt/ml/processing/train/",
    )
    upload_objects(
        script_args["s3_output_bucket"],
        script_args["s3_output_key_prefix"],
        "/opt/ml/processing/test/",
    )
    
    #wait for the file creation
    while attempts < attempts_counter:
        if os.path.exists(train_features_output_path) and os.path.isfile(train_features_output_path): 
            try:
                # Calculate the processed dataset baseline statistics on the Dask cluster
                dask_df = dd.read_csv(train_features_output_path)
                dask_df = client.persist(dask_df)
                baseline = dask_df.describe().compute()
                print(baseline)
                break
                
            except:
                time.sleep(2)
    if attempts == attempts_counter:
        raise Exception(
            "Output file {} couldn't be found".format(train_features_output_path)
        )
    
    print(baseline)
    sys.exit(os.EX_OK)