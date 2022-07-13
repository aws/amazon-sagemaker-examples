# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Evaluation script for measuring model accuracy."""

import json
import logging
import os
import pickle
import tarfile

import pandas as pd
import numpy

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# May need to import additional metrics depending on what you are measuring.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

if __name__ == "__main__":

    tar_model_path = "/opt/ml/processing/model/model.tar.gz"
    model_path = "/opt/ml/processing/model/decision-tree-model.pkl"

    with tarfile.open(tar_model_path) as tar:
        tar.extractall(path="/opt/ml/processing/model/")

    logger.debug("Loading DTree model.")

    model = pickle.load(open(model_path, "rb"))

    test_path = "/opt/ml/processing/test/test.csv"


    logger.info("Loading test input data")
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = numpy.array(df.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    logger.info("Creating classification evaluation report")

    mse = mean_squared_error(y_test, predictions)
    r2s = r2_score(y_test, predictions)

    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse},
            "r2s": {"value": r2s},
        },
    }

    logger.info("Regression report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "dtree_evaluation.json")
    logger.info("Saving regression report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
