"""Feature engineers the credit dataset."""
import logging
import numpy as np
import pandas as pd
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("Starting preprocessing.")

    input_data_path = os.path.join("/opt/ml/processing/input", "german_credit_data.csv")

    try:
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/validation")
        os.makedirs("/opt/ml/processing/test")
    except:
        pass

    logger.info("Reading input data")
    
    data = pd.read_csv(input_data_path, sep=",")
    
    model_data = pd.get_dummies(data)  # Convert categorical variables to sets of indicators
    
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )


    pd.concat([train_data["risk_high risk"], train_data.drop(["risk_low risk", "risk_high risk"], axis=1)], axis=1).to_csv(
        "/opt/ml/processing/train/train.csv", index=False, header=False
    )

    pd.concat(
        [validation_data["risk_high risk"], validation_data.drop(["risk_low risk", "risk_high risk"], axis=1)], axis=1
    ).to_csv("/opt/ml/processing/validation/validation.csv", index=False, header=False)

    pd.concat([test_data["risk_high risk"], test_data.drop(["risk_low risk", "risk_high risk"], axis=1)], axis=1).to_csv(
        "/opt/ml/processing/test/test.csv", index=False, header=False
    )