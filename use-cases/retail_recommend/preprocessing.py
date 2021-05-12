import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])
import sagemaker.amazon.common as smac


def loadDataset(dataframe):
    enc = OneHotEncoder(handle_unknown="ignore")
    onehot_cols = ["StockCode", "CustomerID", "Country"]
    ohe_output = enc.fit_transform(dataframe[onehot_cols])

    vectorizer = TfidfVectorizer(min_df=2)
    unique_descriptions = dataframe["Description"].unique()
    vectorizer.fit(unique_descriptions)
    tfidf_output = vectorizer.transform(dataframe["Description"])

    row = range(len(dataframe))
    col = [0] * len(dataframe)
    unit_price = csr_matrix((dataframe["UnitPrice"].values, (row, col)), dtype="float32")

    X = hstack([ohe_output, tfidf_output, unit_price], format="csr", dtype="float32")

    y = dataframe["Quantity"].values.astype("float32")

    return X, y


def writeProtobuftoDisk(X, y, fname):
    buf = io.BytesIO()
    smac.write_spmatrix_to_sparse_tensor(buf, X, y)
    buf.seek(0)

    with open(fname, "wb") as f:
        f.write(buf.read())


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    df = pd.read_csv(f"{base_dir}/input/Online Retail.csv")

    df.dropna(subset=["CustomerID"], inplace=True)

    df_grouped = df.groupby(["StockCode", "Description", "CustomerID", "Country", "UnitPrice"])[
        "Quantity"
    ].sum()
    df_grouped = df_grouped.loc[df_grouped > 0].reset_index()

    X, y = loadDataset(df_grouped)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    writeProtobuftoDisk(X_train, y_train, f"{base_dir}/output/train/train.protobuf")
    writeProtobuftoDisk(X_test, y_test, f"{base_dir}/output/test/test.protobuf")

#     prefix = 'personalization'

#     train_key      = 'train.protobuf'
#     train_prefix   = f'{prefix}/train'

#     test_key       = 'test.protobuf'
#     test_prefix    = f'{prefix}/test'

#     train_data = writeDatasetToProtobuf(X_train, y_train, bucket, train_prefix, train_key)
#     test_data  = writeDatasetToProtobuf(X_test, y_test, bucket, test_prefix, test_key)
