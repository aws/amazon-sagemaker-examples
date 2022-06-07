
import os
import json
import sys
import numpy as np
import pandas as pd
import pathlib
import tarfile


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

if __name__ == "__main__":

    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("./model")
    import tensorflow as tf

    model = tf.keras.models.load_model("./model/1")
    test_path = "/opt/ml/processing/test/"
    df = pd.read_csv(test_path + "/test.csv")
    x_test = df[feature_columns].to_numpy()
    y_test = df[label_column].to_numpy()
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest MSE :", scores)

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "regression_metrics": {
            "mse": {"value": scores, "standard_deviation": "NaN"},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
