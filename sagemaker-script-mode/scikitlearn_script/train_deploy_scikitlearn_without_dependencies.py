import argparse
import numpy as np
import os
import pandas as pd
import re
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sagemaker_training import environment


def parse_args():
    """
    Parse arguments.
    """
    env = environment.Environment()

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=env.num_cpus)
    parser.add_argument("--n-estimators", type=int, default=120)

    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


def load_dataset(path):
    """
    Load entire dataset.
    """
    # Take the set of files and read them all into a single pandas dataframe
    files = [
        os.path.join(path, file) for file in os.listdir(path) if file.endswith("csv")
    ]

    if len(files) == 0:
        raise ValueError("Invalid # of files in dir: {}".format(path))

    raw_data = [pd.read_csv(file, sep=",", header=None) for file in files]
    data = pd.concat(raw_data)

    # labels are in the first column
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    return X, y


def start(args):
    """
    Train a Random Forest Regressor
    """
    print("Training mode")

    try:
        X_train, y_train = load_dataset(args.train)
        X_test, y_test = load_dataset(args.test)

        hyperparameters = {
            "max_depth": args.max_depth,
            "verbose": 1,  # show all logs
            "n_jobs": args.n_jobs,
            "n_estimators": args.n_estimators,
        }
        print("Training the classifier")
        model = RandomForestRegressor()
        model.set_params(**hyperparameters)
        model.fit(X_train, y_train)
        print("Score: {}".format(model.score(X_test, y_test)))
        # joblib.dump(model, open(os.path.join(args.model_dir, "iris_model.pkl"), "wb"))
        joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, "failure"), "w") as s:
            s.write("Exception during training: " + str(e) + "\\n" + trc)

        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during training: " + str(e) + "\\n" + trc, file=sys.stderr)

        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


def model_fn(model_dir):
    """
    Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    args, _ = parse_args()

    start(args)
