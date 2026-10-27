from enum import IntEnum
import json
import logging
import re
from flask import Flask
from flask import request
from joblib import dump, load
import numpy as np
import os

logger = logging.getLogger(__name__)


class IrisLabel(IntEnum):
    setosa = 0
    versicolor = 1
    virginica = 2


class IrisModel:
    LABELS = IrisLabel
    NUM_FEATURES = 4

    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None

    # Cache the model to prevent repeatedly loading it for every request
    @property
    def model(self):
        if self._model is None:
            self._model = load(self.model_path)
        return self._model

    def predict_from_csv(self, lines, **kwargs):
        data = np.genfromtxt(lines.split("\n"), delimiter=",")
        return self.predict(data, **kwargs)

    def predict_from_json(self, obj, **kwargs):
        req = json.loads(obj)
        instances = req["instances"]
        x = np.array([instance["features"] for instance in instances])
        return self.predict(x, **kwargs)

    def predict_from_jsonlines(self, obj, **kwargs):
        x = np.array([json.loads(line)["features"] for line in obj.split("\n")])
        return self.predict(x, **kwargs)

    def predict(self, x, return_names=True):
        label_codes = self.model.predict(x.reshape(-1, IrisModel.NUM_FEATURES))

        if return_names:
            predictions = [IrisModel.LABELS(code).name for code in label_codes]
        else:
            predictions = label_codes.tolist()

        return predictions


SUPPORTED_REQUEST_MIMETYPES = ["application/json", "application/jsonlines", "text/csv"]
SUPPORTED_RESPONSE_MIMETYPES = ["application/json", "application/jsonlines", "text/csv"]

app = Flask(__name__)
model = IrisModel(model_path="/opt/ml/model/model-artifacts.joblib")

# Create a path for health checks
@app.route("/ping")
def endpoint_ping():
    return ""


# Create a path for inference
@app.route("/invocations", methods=["POST"])
def endpoint_invocations():
    try:
        logger.info(f"Processing request: {request.headers}")
        logger.debug(f"Payload: {request.headers}")

        if request.content_type not in SUPPORTED_REQUEST_MIMETYPES:
            logger.error(f"Unsupported Content-Type specified: {request.content_type}")
            return f"Invalid Content-Type. Supported Content-Types: {', '.join(SUPPORTED_REQUEST_MIMETYPES)}"
        elif request.content_type == "text/csv":
            # Step 1: Decode payload into input format expected by model
            data = request.get_data().decode("utf8")
            # Step 2: Perform inference with the loaded model
            predictions = model.predict_from_csv(data)
        elif request.content_type == "application/json":
            data = request.get_data().decode("utf8")
            predictions = model.predict_from_json(data)
        elif request.content_type == "application/jsonlines":
            data = request.get_data().decode("utf8")
            predictions = model.predict_from_jsonlines(data)

        # Step 3: Process predictions into the specified response type (if specified)
        response_mimetype = request.accept_mimetypes.best_match(
            SUPPORTED_RESPONSE_MIMETYPES, default="application/json"
        )

        if response_mimetype == "text/csv":
            response = "\n".join(predictions)
        elif response_mimetype == "application/jsonlines":
            response = "\n".join([json.dumps({"class": pred}) for pred in predictions])
        elif response_mimetype == "application/json":
            response = json.dumps({"predictions": [{"class": pred} for pred in predictions]})

        return response
    except Exception as e:
        return f"Error during model invocation: {str(e)} for input: {request.get_data()}"
