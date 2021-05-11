import argparse
import io
import json
import subprocess
import sys

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "awswrangler"])
import awswrangler as wr
import sagemaker.amazon.common as smac

# Parse argument variables passed via the DeployModel processing step
parser = argparse.ArgumentParser()
parser.add_argument("--data-s3-uri", type=str)
args = parser.parse_args()

data_s3_uri = args.data_s3_uri


def loadDataset(dataframe):
    n_rows = dataframe.shape[0]
    n_customers = int(dataframe["CustomerID"][0].split(",")[0].strip("()"))
    n_items = int(dataframe["StockCode"][0].split(",")[0].strip("()"))
    n_countries = int(dataframe["Country"][0].split(",")[0].strip("()"))
    n_tokens = int(dataframe["Description_features"][0].split(",")[0].strip("()"))
    n_features = (
        n_customers + n_items + n_countries + n_tokens + 1
    )  # plus one is for the UnitPrice feature

    # Features are one-hot encoded in a sparse matrix
    X = lil_matrix((n_rows, n_features)).astype("float32")
    # Labels are stored in a vector
    y = []

    for ix, row in dataframe.iterrows():
        desc = row["Description_features"]

        X[ix, 0] = row["UnitPrice"]
        X[ix, int(row["CustomerID"].split(",")[1].strip("[]")) + 1] = 1
        X[ix, n_customers + int(row["StockCode"].split(",")[1].strip("[]")) + 1] = 1
        X[ix, n_customers + n_items + int(row["Country"].split(",")[1].strip("[]")) + 1] = 1

        for col_idx in desc.split(",[")[1].strip("]").split(","):
            X[ix, n_customers + n_items + n_countries + int(col_idx) + 1] = 1

        y.append(row["Quantity"])

    y = np.array(y).astype("float32")

    return X, y


def writeProtobuftoDisk(X, y, fname):
    buf = io.BytesIO()
    smac.write_spmatrix_to_sparse_tensor(buf, X, y)
    buf.seek(0)

    with open(fname, "wb") as f:
        f.write(buf.read())


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    df = wr.s3.read_csv(path=data_s3_uri, dataset=True)

    X, y = loadDataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    writeProtobuftoDisk(X_train, y_train, f"{base_dir}/output/train/train.protobuf")
    writeProtobuftoDisk(X_test, y_test, f"{base_dir}/output/test/test.protobuf")

    n_features = X.shape[1]
    state_data = {"n_features": n_features}
    with open(f"{base_dir}/output/pipeline_state/state.json", "w") as f:
        f.write(json.dumps(state_data))
