import argparse
import os
import warnings

import pandas as pd
import numpy as np
import tarfile
import sklearn
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

columns = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

if __name__ == "__main__":

    # Read the arguments passed to the script.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    # Read input data into a Pandas dataframe.
    input_data_path = os.path.join("/opt/ml/processing/input", "train.csv")
    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path, names=None, header=0, sep=",")

    # Defining one-hot encoders.
    print("performing one hot encoding")
    transformer = make_column_transformer(
        (
            [
                "credit_history",
                "purpose",
                "personal_status_sex",
                "other_debtors",
                "property",
                "other_installment_plans",
                "housing",
                "job",
                "telephone",
                "foreign_worker",
            ],
            OneHotEncoder(sparse=False),
        ),
        remainder="passthrough",
    )

    print("preparing the features and labels")
    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]

    print("building sklearn transformer")
    featurizer_model = transformer.fit(X)
    features = featurizer_model.transform(X)
    labels = LabelEncoder().fit_transform(y)

    # Splitting.
    split_ratio = args.train_test_split_ratio
    print("Splitting data into train and validation sets with ratio {}".format(split_ratio))
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=split_ratio, random_state=0
    )

    print("Train features shape after preprocessing: {}".format(X_train.shape))
    print("Validation features shape after preprocessing: {}".format(X_val.shape))

    # Saving outputs.
    train_features_output_path = os.path.join("/opt/ml/processing/train", "train_features.csv")
    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train_labels.csv")

    val_features_output_path = os.path.join("/opt/ml/processing/val", "val_features.csv")
    val_labels_output_path = os.path.join("/opt/ml/processing/val", "val_labels.csv")

    print("Saving training features to {}".format(train_features_output_path))
    pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)

    print("Saving training labels to {}".format(train_labels_output_path))
    pd.DataFrame(y_train).to_csv(train_labels_output_path, header=False, index=False)

    print("Saving validation features to {}".format(val_features_output_path))
    pd.DataFrame(X_val).to_csv(val_features_output_path, header=False, index=False)

    print("Saving validation labels to {}".format(val_labels_output_path))
    pd.DataFrame(y_val).to_csv(val_labels_output_path, header=False, index=False)

    # Saving model.
    model_path = os.path.join("/opt/ml/processing/model", "model.joblib")
    model_output_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")

    print("Saving featurizer model to {}".format(model_output_path))
    joblib.dump(featurizer_model, model_path)
    tar = tarfile.open(model_output_path, "w:gz")
    tar.add(model_path, arcname="model.joblib")
    tar.close()
