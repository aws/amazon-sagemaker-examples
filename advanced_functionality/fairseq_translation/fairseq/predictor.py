import os

import flask
from sagemaker_translate import input_fn, model_fn, output_fn, predict_fn

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

print("in predictor.py")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = model_fn(model_path)
        return cls.model

    @classmethod
    def predict(cls, serialized_input_data):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        input_data = input_fn(serialized_input_data)
        output = predict_fn(input_data, clf)
        return output_fn(output)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data."""
    data = None
    data = flask.request.data.decode("utf-8")

    # Do the prediction
    result, accept = ScoringService.predict(data)

    return flask.Response(response=result, status=200, mimetype="text/json")
