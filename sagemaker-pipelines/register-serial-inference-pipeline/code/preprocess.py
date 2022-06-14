import glob
import numpy as np
import pandas as pd
import os
import json
import joblib
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tarfile

try:
    from sagemaker_containers.beta.framework import (
        content_types,
        encoders,
        env,
        modules,
        transformer,
        worker,
        server,
    )
except ImportError:
    pass

feature_columns = [
    "longitude",
    "latitude",
    "housingMedianAge",
    "totalRooms",
    "totalBedrooms",
    "population",
    "households",
    "medianIncome",
]
label_column = "medianHouseValue"

base_dir = "/opt/ml/processing"
base_output_dir = "/opt/ml/output/"

if __name__ == "__main__":
    df = pd.read_csv(f"{base_dir}/input/raw_data_all.csv")
    feature_data = df.drop(label_column, axis=1, inplace=False)
    label_data = df[label_column]
    x_train, x_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.33)

    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    train_dataset = pd.concat([pd.DataFrame(x_train), y_train.reset_index(drop=True)], axis=1)
    test_dataset = pd.concat([pd.DataFrame(x_test), y_test.reset_index(drop=True)], axis=1)

    train_dataset.columns = feature_columns + [label_column]
    test_dataset.columns = feature_columns + [label_column]

    train_dataset.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    test_dataset.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    joblib.dump(scaler, "model.joblib")
    with tarfile.open(f"{base_dir}/scaler_model/model.tar.gz", "w:gz") as tar_handle:
        tar_handle.add(f"model.joblib")


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)

        if len(df.columns) == len(feature_columns) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns + [label_column]
        elif len(df.columns) == len(feature_columns):
            # This is an unlabelled example.
            df.columns = feature_columns

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append(row)
        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features


def model_fn(model_dir):
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
