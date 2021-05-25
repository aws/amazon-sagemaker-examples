
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.compose import make_column_transformer

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    parser.add_argument("--random-split", type=int, default=0)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    input_data_path = os.path.join("/opt/ml/processing/input", "rawdata.csv")

    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    df.sample(frac=1)

    COLS = df.columns
    newcolorder = (
        ["PAY_AMT1", "BILL_AMT1"]
        + list(COLS[1:])[:11]
        + list(COLS[1:])[12:17]
        + list(COLS[1:])[18:]
    )

    split_ratio = args.train_test_split_ratio
    random_state = args.random_split

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("Label", axis=1), df["Label"], test_size=split_ratio, random_state=random_state
    )

    preprocess = make_column_transformer(
        (["PAY_AMT1"], StandardScaler()), (["BILL_AMT1"], MinMaxScaler()), remainder="passthrough"
    )

    print("Running preprocessing and feature engineering transformations")
    train_features = pd.DataFrame(preprocess.fit_transform(X_train), columns=newcolorder)
    test_features = pd.DataFrame(preprocess.transform(X_test), columns=newcolorder)

    # concat to ensure Label column is the first column in dataframe
    train_full = pd.concat(
        [pd.DataFrame(y_train.values, columns=["Label"]), train_features], axis=1
    )
    test_full = pd.concat([pd.DataFrame(y_test.values, columns=["Label"]), test_features], axis=1)

    print("Train data shape after preprocessing: {}".format(train_features.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))

    train_features_headers_output_path = os.path.join(
        "/opt/ml/processing/train_headers", "train_data_with_headers.csv"
    )

    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_data.csv")

    test_features_output_path = os.path.join("/opt/ml/processing/test", "test_data.csv")

    print("Saving training features to {}".format(train_features_output_path))
    train_full.to_csv(train_features_output_path, header=False, index=False)
    print("Complete")

    print("Save training data with headers to {}".format(train_features_headers_output_path))
    train_full.to_csv(train_features_headers_output_path, index=False)

    print("Saving test features to {}".format(test_features_output_path))
    test_full.to_csv(test_features_output_path, header=False, index=False)
    print("Complete")
