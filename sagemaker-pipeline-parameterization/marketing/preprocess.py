"""Feature engineers the marketing dataset."""
import logging
import numpy as np
import pandas as pd
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("Starting preprocessing.")

    input_data_path = os.path.join("/opt/ml/processing/input", "marketing-dataset.csv")

    try:
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/validation")
        os.makedirs("/opt/ml/processing/test")
    except:
        pass

    logger.info("Reading input data")

    # read csv
    data = pd.read_csv(input_data_path, sep=";")

    data["no_previous_contact"] = np.where(
        data["pdays"] == 999, 1, 0
    )  # Indicator variable to capture when pdays takes a value of 999

    data["not_working"] = np.where(
        np.in1d(data["job"], ["student", "retired", "unemployed"]), 1, 0
    )  # Indicator for individuals not actively employed

    model_data = pd.get_dummies(data)  # Convert categorical variables to sets of indicators

    model_data = model_data.drop(
        ["duration", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
        axis=1,
    )

    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )

    pd.concat([train_data["y_yes"], train_data.drop(["y_no", "y_yes"], axis=1)], axis=1).to_csv(
        "/opt/ml/processing/train/train.csv", index=False, header=False
    )

    pd.concat(
        [validation_data["y_yes"], validation_data.drop(["y_no", "y_yes"], axis=1)], axis=1
    ).to_csv("/opt/ml/processing/validation/validation.csv", index=False, header=False)

    pd.concat([test_data["y_yes"], test_data.drop(["y_no", "y_yes"], axis=1)], axis=1).to_csv(
        "/opt/ml/processing/test/test.csv", index=False, header=False
    )
