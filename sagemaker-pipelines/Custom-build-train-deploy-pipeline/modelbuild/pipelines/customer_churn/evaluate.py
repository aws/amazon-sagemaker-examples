"""Evaluation script for measuring model accuracy."""

import json
import os
import tarfile
import logging
import pickle

import pandas as pd
import xgboost

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

#May need to import additional metrics depending on what you are measuring.
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    print("Loading test input data")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    print("Creating classification evaluation report")
    acc = accuracy_score(y_test, predictions.round())
    auc = roc_auc_score(y_test, predictions.round())
    report_dict = {
        "binary_classification_metrics": { #This can change based on the model used, but it must be a specific name per <insert doc link>
            "accuracy": {"value": acc, "standard_deviation": "NaN"}, #Metrics can change, but follow the format in the docs <insert doc link>
            "auc": {"value": auc, "standard_deviation": "NaN"},
        },
    }

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join(
        "/opt/ml/processing/evaluation", "evaluation.json"
    )
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
