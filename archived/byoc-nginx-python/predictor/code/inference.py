from io import StringIO
import os
import json
import flask
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, Response, Request
import csv

app = Flask(__name__)
model = None
MODEL_PATH = "/opt/ml/model"


def load_model():
    """
    Load the XGBoost model from the specified MODEL_PATH.

    Returns:
        xgb.Booster: The loaded XGBoost model.
    """
    xgb_model_path = os.path.join(MODEL_PATH, "xgboost-model")

    # Load the model from the file
    with open(xgb_model_path, "rb") as f:
        model = joblib.load(f)

    return model


def preprocess(input_data, content_type):
    """
    Preprocess the input data and convert it into an XGBoost DMatrix.

    Args:
        input_data (str): The input data as a string (CSV format).
        content_type (str): The content type of the input data (expected: "text/csv; charset=utf-8").

    Returns:
        xgb.DMatrix: The preprocessed data in XGBoost DMatrix format.
    """
    if content_type == "text/csv; charset=utf-8":
        df = pd.read_csv(StringIO(input_data), header=None)
        data = xgb.DMatrix(data=df)
        return data


def predict(input_data):
    """
    Make predictions using the preprocessed input data.

    Args:
        input_data (xgb.DMatrix): The preprocessed data in XGBoost DMatrix format.

    Returns:
        list: A list of predictions or an empty list if there's an error.
    """
    try:
        # Load the model
        model = load_model()

        # Make predictions using the input data
        predictions = model.predict(input_data)

        # Convert predictions (numpy array) to a list and return
        return predictions.tolist()

    except Exception as e:
        # Log the exception and return an empty list in case of an error
        print(f"Error while making predictions: {e}", flush=True)
        return []


@app.route("/ping", methods=["GET"])
def ping():
    """
    Check the health of the model server by verifying if the model is loaded.

    Returns a 200 status code if the model is loaded successfully, or a 500
    status code if there is an error.

    Returns:
        flask.Response: A response object containing the status code and mimetype.
    """
    model = load_model()
    status = 200 if model is not None else 500
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Handle prediction requests by preprocessing the input data, making predictions,
    and returning the predictions as a JSON object.

    This function checks if the request content type is supported (text/csv; charset=utf-8),
    and if so, decodes the input data, preprocesses it, makes predictions, and returns
    the predictions as a JSON object. If the content type is not supported, a 415 status
    code is returned.

    Returns:
        flask.Response: A response object containing the predictions, status code, and mimetype.
    """
    print(f"Predictor: received content type: {flask.request.content_type}")
    if flask.request.content_type == "text/csv; charset=utf-8":
        input = flask.request.data.decode("utf-8")
        transformed_data = preprocess(input, flask.request.content_type)
        predictions = predict(transformed_data)

        # Return the predictions as a JSON object
        return json.dumps({"result": predictions})
    else:
        print(f"Received: {flask.request.content_type}", flush=True)
        return flask.Response(
            response=f"XGBPredictor: This predictor only supports CSV data; Received: {flask.request.content_type}",
            status=415,
            mimetype="text/plain",
        )
