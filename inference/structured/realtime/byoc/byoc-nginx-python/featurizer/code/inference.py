import numpy as np
import pandas as pd
import io
from io import StringIO
import flask
from flask import Flask
import os
import joblib
import sklearn
import csv

app = Flask(__name__)
featurizer = None
MODEL_PATH = "/opt/ml/model"


def load_model():
    """
    Load the featurizer model from the specified path.

    This function reads the featurizer model file, preprocess.joblib, located
    in the MODEL_PATH directory. It then loads the model using joblib and
    prints a message to indicate successful loading.

    Returns:
        featurizer(sklearn.compose._column_transformer.ColumnTransformer): The loaded featurizer model.
    """
    # Construct the path to the featurizer model file
    ft_model_path = os.path.join(MODEL_PATH, "preprocess.joblib")

    featurizer = None

    try:
        # Open the model file and load the featurizer using joblib
        with open(ft_model_path, "rb") as f:
            featurizer = joblib.load(f)
            print("Featurizer model loaded", flush=True)
    except FileNotFoundError:
        print(f"Error: Featurizer model file not found at {ft_model_path}", flush=True)
    except Exception as e:
        print(f"Error loading featurizer model: {e}", flush=True)

    # Return the loaded featurizer model, or None if there was an error
    return featurizer


# sagemaker inference.py script
def transform_fn(request_body, request_content_type):
    """
    Transform the request body into a usable numpy array for the model.

    This function takes the request body and content type as input, and
    returns a transformed numpy array that can be used as input for the
    prediction model.

    Parameters:
        request_body (str): The request body containing the input data.
        request_content_type (str): The content type of the request body.

    Returns:
        data (np.ndarray): Transformed input data as a numpy array.
    """
    # Define the column names for the input data
    feature_columns_names = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ]
    label_column = "rings"

    # Check if the request content type is supported (text/csv)
    if request_content_type == "text/csv":
        # Load the featurizer model
        featurizer = load_model()

        # Check if the featurizer is a ColumnTransformer
        if isinstance(
            featurizer, sklearn.compose._column_transformer.ColumnTransformer
        ):
            print(f"Featurizer model loaded", flush=True)

        # Read the input data from the request body as a CSV file
        df = pd.read_csv(StringIO(request_body), header=None)

        # Assign column names based on the number of columns in the input data
        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names

        # Transform the input data using the featurizer
        data = featurizer.transform(df)

        # Return the transformed data as a numpy array
        return data
    else:
        # Raise an error if the content type is unsupported
        raise ValueError("Unsupported content type: {}".format(request_content_type))


@app.route("/ping", methods=["GET"])
def ping():
    """
    Check the health of the model server by attempting to load the model.

    Returns a 200 status code if the model is loaded successfully, or a 500
    status code if there is an error.

    Returns:
        flask.Response: A response object containing the status code and mimetype.
    """
    # Check if the model can be loaded, set the status accordingly
    featurizer = load_model()
    status = 200 if featurizer is not None else 500

    # Return the response with the determined status code
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Handle prediction requests by transforming the input data and returning the
    transformed data as a CSV string.

    This function checks if the request content type is supported (text/csv),
    and if so, decodes the input data, transforms it using the transform_fn
    function, and returns the transformed data as a CSV string. If the content
    type is not supported, a 415 status code is returned.

    Returns:
        flask.Response: A response object containing the transformed data,
                        status code, and mimetype.
    """

    # Convert from JSON to dict
    print(f"Featurizer: received content type: {flask.request.content_type}")
    if flask.request.content_type == "text/csv":
        # Decode input data and transform
        input = flask.request.data.decode("utf-8")
        transformed_data = transform_fn(input, flask.request.content_type)

        # Format transformed_data into a csv string
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)

        for row in transformed_data:
            csv_writer.writerow(row)

        csv_buffer.seek(0)

        # Return the transformed data as a CSV string in the response
        return flask.Response(response=csv_buffer, status=200, mimetype="text/csv")
    else:
        print(f"Received: {flask.request.content_type}", flush=True)
        return flask.Response(
            response="Transformer: This predictor only supports CSV data",
            status=415,
            mimetype="text/plain",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
