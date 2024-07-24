import os
os.system('python3 -m pip install -U sagemaker xgboost==1.0.1 scikit-learn')

import glob
import boto3
import json
import pathlib
import pickle
import tarfile
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import xgboost
import argparse

from sklearn.metrics import mean_squared_error

from sagemaker.session import Session
from sagemaker.experiments.run import Run, load_run

session = Session(boto3.session.Session(region_name="us-east-1"))


def read_parameters():
    """
    Read job parameters
    Returns:
        (Namespace): read parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/opt/ml/processing/model")
    parser.add_argument('--test_csv', type=str, default="/opt/ml/processing/output/test")
    parser.add_argument('--output_path', type=str, default="/opt/ml/processing/evaluation")
    params, _ = parser.parse_known_args()
    return params


if __name__ == "__main__":
    
    # reading job parameters
    args = read_parameters()
    
    available_model = glob.glob(os.path.join(args.model_path, "*.tar.gz"))
    assert len(available_model) == 1, f"multiple tar gz models found in models dir: {available_model}"
    print(f"Model found: {available_model}")
    model_path = available_model[0]

    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    model = pickle.load(open("xgboost-model", "rb"))

    test_path = os.path.join(args.test_csv, "test.csv")
    df = pd.read_csv(test_path, header=None)

    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)

    X_test = xgboost.DMatrix(df.values)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse, "standard_deviation": std},
        }
    }
    
    print(report_dict)

    output_dir = args.output_path
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
    print("Done")
