import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pickle
from collections import Counter
from io import StringIO
from itertools import islice
from timeit import default_timer as timer

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from autogluon.tabular import TabularDataset, TabularPredictor
    from prettytable import PrettyTable


def make_str_table(df):
    table = PrettyTable(["index"] + list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    return str(table)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def preprocess(df, columns, target):
    features = copy.deepcopy(columns)
    features.remove(target)
    first_row_list = df.iloc[0].tolist()

    if set(first_row_list) >= set(features):
        df.drop(0, inplace=True)
    if len(first_row_list) == len(columns):
        df.columns = columns
    if len(first_row_list) == len(features):
        df.columns = features

    return df


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network) and the column info.
    """
    print(f"Loading model from {model_dir} with contents {os.listdir(model_dir)}")

    net = TabularPredictor.load(model_dir, verbosity=True)
    with open(f"{model_dir}/code/columns.pkl", "rb") as f:
        column_dict = pickle.load(f)
    return net, column_dict


def transform_fn(models, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param models: The Gluon model and the column info.
    :param data: The request payload.
    :param input_content_type: The request content type. ('text/csv')
    :param output_content_type: The (desired) response content type. ('text/csv')
    :return: response payload and content type.
    """
    start = timer()
    net = models[0]
    column_dict = models[1]
    label_map = net.class_labels_internal_map  ###

    # text/csv
    if "text/csv" in input_content_type:
        # Load dataset
        columns = column_dict["columns"]

        if type(data) == str:
            # Load dataset
            df = pd.read_csv(StringIO(data), header=None)
        else:
            df = pd.read_csv(StringIO(data.decode()), header=None)

        df_preprosessed = preprocess(df, columns, net.label)

        ds = TabularDataset(data=df_preprosessed)

        try:
            predictions = net.predict_proba(ds)
            predictions_ = net.predict(ds)
        except:
            try:
                predictions = net.predict_proba(ds.fillna(0.0))
                predictions_ = net.predict(ds.fillna(0.0))
                warnings.warn("Filled NaN's with 0.0 in order to predict.")
            except Exception as e:
                response_body = e
                return response_body, output_content_type

        # threshold = 0.5
        # predictions_label = [[k for k, v in label_map.items() if v == 1][0] if i > threshold else [k for k, v in label_map.items() if v == 0][0] for i in predictions]
        predictions_label = predictions_.tolist()

        # Print prediction counts, limit in case of regression problem
        pred_counts = Counter(predictions_label)
        n_display_items = 30
        if len(pred_counts) > n_display_items:
            print(
                f"Top {n_display_items} prediction counts: "
                f"{dict(take(n_display_items, pred_counts.items()))}"
            )
        else:
            print(f"Prediction counts: {pred_counts}")

        # Form response
        output = StringIO()
        pd.DataFrame(predictions).to_csv(output, header=False, index=False)
        response_body = output.getvalue()

        # If target column passed, evaluate predictions performance
        target = net.label
        if target in ds:
            print(
                f"Label column ({target}) found in input data. "
                "Therefore, evaluating prediction performance..."
            )
            try:
                performance = net.evaluate_predictions(
                    y_true=ds[target], y_pred=np.array(predictions_label), auxiliary_metrics=True
                )
                print(json.dumps(performance, indent=4, default=pd.DataFrame.to_json))
                time.sleep(0.1)
            except Exception as e:
                # Print exceptions on evaluate, continue to return predictions
                print(f"Exception: {e}")
    else:
        raise NotImplementedError("content_type must be 'text/csv'")

    elapsed_time = round(timer() - start, 3)
    print(f"Elapsed time: {round(timer()-start,3)} seconds")

    return response_body, output_content_type
