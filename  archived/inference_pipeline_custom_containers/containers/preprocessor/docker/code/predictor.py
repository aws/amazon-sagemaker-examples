# This is the file that implements a flask server to do inferences. It's the file that you will modify
# to implement the prediction for your own algorithm.

from __future__ import print_function

import csv
import glob
import json
import os
import shutil
import stat
import sys
from io import StringIO

import flask
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, make_response, request
from joblib import dump, load
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)

from utils import (
    load_json_object,
    print_files_in_path,
    print_json_object,
    save_model_artifacts,
    write_failure_file,
)

model_artifacts_path = "/opt/ml/model/"
feature_column = "words"
label_column = "label"
preprocessor = None
le = None

# The flask app for serving predictions
app = flask.Flask(__name__)


def load_model():
    global preprocessor
    global le
    if not preprocessor:
        preprocessor = load(os.path.join(model_artifacts_path, "model.joblib"))
    if not le:
        le = load(os.path.join(model_artifacts_path, "label.joblib"))


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    load_model()
    health = preprocessor is not None and le is not None

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():

    print("data: ", request.data[:100])
    print("cookies: ", request.cookies)
    print("headers: ", dict(request.headers))
    print("args: ", request.args)

    load_model()

    content_type = request.headers["Content-Type"]
    print("Content type", content_type)
    accept = request.headers["Accept"]
    print("Accept", accept)

    input_data = request.data.decode()

    first_entry = input_data.split("\n", 1)[0].split(",", 1)[0]
    print("First entry is: ", first_entry)
    df = None

    if first_entry == "label" or first_entry.startswith("category_"):
        recs = [(row[0], set(row[1:])) for row in csv.reader(StringIO(input_data))]
        if first_entry == "label":
            df = pd.DataFrame.from_records(recs[1:], columns=[label_column, feature_column])
        else:
            df = pd.DataFrame.from_records(recs, columns=[label_column, feature_column])
        # This is a labelled example, includes the ring label
        print("Length indicates that label is included")
    else:
        print("Length indicates that label is not included.")
        # This is an unlabelled example.
        recs = [(set(row),) for row in csv.reader(StringIO(input_data))]
        df = pd.DataFrame.from_records(recs, columns=[feature_column])

    print("merged df", df.head())
    features = preprocessor.transform(df["words"])
    prediction = None

    if label_column in df:
        print("label_column in input_data")
        labels = le.transform(df[label_column])
        # Return the label (as the first column) and the set of features.
        prediction = np.insert(features.todense(), 0, labels, axis=1)
    else:
        print("label_column not in input_data")
        # Return only the set of features
        prediction = features.todense()

    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return Response(json.dumps(json_output), mimetype=accept)
    # TODO: use custom flag to indicate that this is in a pipeline rather than relying on the '*/*'
    elif accept == "text/csv" or accept == "*/*":
        return Response(encoders.encode(prediction, "text/csv"), mimetype="text/csv")
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))
