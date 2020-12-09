"""Feature engineers the customer churn dataset."""
import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print(input_data)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/raw-data.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")


    #read in csv
    df = pd.read_csv(fn)

    #drop the "Phone" feature column
    df = df.drop(["Phone"], axis=1)
    #chagne the data type of "Area Code"
    df["Area Code"] = df["Area Code"].astype(object)
    #drop several other columns
    df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    #Convert categorical variables into dummy/indicator variables.
    model_data = pd.get_dummies(df)

    #Create one binary classification target column
    model_data = pd.concat(
        [
            model_data["Churn?_True."],
            model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
        ],
        axis=1,
    )

    #Split the data
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )

    pd.DataFrame(train_data).to_csv(
        f"{base_dir}/train/train.csv", header=False, index=False
    )
    pd.DataFrame(validation_data).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test_data).to_csv(
        f"{base_dir}/test/test.csv", header=False, index=False
    )
