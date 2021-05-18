# This is the file that implements a flask server to do inferences. It's the file that you will modify
# to implement the prediction for your own algorithm.

from __future__ import print_function

import csv
import glob
import json
import os
import random
import shutil
import stat
import sys
from io import StringIO

import boto3
import flask
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from joblib import dump, load
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)

model_artifacts_path = "/opt/ml/model/"
feature_column = "words"
label_column = "label"
feature_store_table = "PipelineLookupTable"

aws_region = "us-west-2"
client = boto3.client("dynamodb", region_name=aws_region)

# The flask app for serving predictions
app = flask.Flask(__name__)

le = None


def load_model():
    global le
    if not le:
        le = load(os.path.join(model_artifacts_path, "label.joblib"))


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    load_model()
    health = le is not None

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


def prepare_category_prediction(row):
    return row[9:]


def get_agent_by_category(category):
    return get_agent_by_category_ddb(category)


def get_agent_by_category_ddb(category):
    response = client.query(
        TableName=feature_store_table,
        ExpressionAttributeValues={":v1": {"S": category}, ":v2": {"BOOL": True}},
        ExpressionAttributeNames={"#A": "Available"},
        FilterExpression="#A = :v2",
        KeyConditionExpression="Specialty = :v1",
    )
    pick = random.choice(response["Items"])
    agent = {
        "ID": pick["ID"]["S"],
        "FirstName": pick["FirstName"]["S"],
        "LastName": pick["LastName"]["S"],
    }

    print("Found agent with ID: ", agent["ID"])
    return agent


def get_agent_by_category_naive(category):
    agents = {
        "itemization": ["Frank", "Abigale", "John"],
        "estate taxes": ["Peter", "Samuel", "Beth"],
        "medical": ["Samantha", "Mohammed", "Eve"],
        "deferments": ["Lee", "Claudia", "Annabel"],
        "investments": ["Milly", "Ray", "Stacey"],
        "properties": ["Rosa", "Meghan", "Cleo"],
    }

    if category in agents.keys():
        return random.choice(agents[category])
    else:
        raise RuntimeError("Category {} is not supported by this script.".format(category))


@app.route("/invocations", methods=["POST"])
def transformation():
    print("data: ", request.data[:100])
    print("cookies: ", request.cookies)
    print("headers: ", dict(request.headers))
    print("args: ", request.args)

    load_model()

    # We want to get the 'text/csv' bit from something like 'text/csv; charset=utf-8'
    content_type = request.headers["Content-Type"].split(";", 1)[0]
    print("Content type", content_type)
    accept = request.headers["Accept"]
    print("Accept", accept)

    input_data = request.data.decode()
    features = []

    if content_type == "application/json":
        decoded_payload = json.loads(input_data)
        entries = [entry["predicted_label"] for entry in decoded_payload["predictions"]]
        predictions = np.array(entries)
        features = le.inverse_transform(predictions)
    elif content_type == "text/csv":
        entries = [int(float(label)) for label in input_data.split(",")]
        predictions = np.array(entries)
        features = le.inverse_transform(predictions)
    else:
        raise RuntimeError("{} content type is not supported by this script.".format(content_type))

    if accept == "application/json":
        instances = []
        for row in features.tolist():
            category = prepare_category_prediction(row)
            print("Category is: ", category)
            agent = get_agent_by_category(category)
            instances.append({"category": category, "agent": agent})

        json_output = {"response": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    # TODO: use custom flag to indicate that this is in a pipeline rather than relying on the '*/*'
    elif accept == "text/csv" or accept == "*/*":
        # TODO: this is wrong. fix it
        return worker.Response(encoders.encode(features, accept), mimetype="text/csv")
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))
