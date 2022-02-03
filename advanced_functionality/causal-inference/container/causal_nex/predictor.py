# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import ast
import os
import pickle
import signal
import sys
import traceback
from collections import defaultdict

import flask
import pandas as pd
from causalnex.structure.structuremodel import StructureModel
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict and an intervention functions that does a prediction (or intervention) based on the model and the input data.


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls, model_name="causal_model.pkl"):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, model_name), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    # pass node as param
    def predict(cls, input, target_node):
        """For the input, do the predictions and return them.

        Args:
            input (a list of dictionaries): The data on which to do the predictions. There will be
                one prediction per row in the result dataframe"""
        bn = cls.get_model()
        result = bn.predict(
            pd.DataFrame.from_dict(input, orient="columns"), target_node
        )

        return result

    @classmethod
    def intervention(cls, input):

        """Users can apply an intervention to any node in the data, updating its distribution using a do operator, examining the effect of that intervention by querying marginals and resetting any interventions

        Args:
           input (a list of dictionaries): The data on which to do the interventions.
        """

        from causalnex.inference import InferenceEngine

        bn = cls.get_model()
        ie = InferenceEngine(bn)
        i_node = input["node"]
        i_states = input["states"]
        i_target = input["target_node"]

        print(i_node, i_states, i_target)
        lst = []

        # i_states is a list of dict
        for state in i_states:
            state = {int(k): int(v) for k, v in state.items()}
            ie.do_intervention(i_node, state)
            intervention_result = ie.query()[i_target]
            lst.append(intervention_result)
            print("Updated marginal", intervention_result)
            ie.reset_do(i_node)

        return lst


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
        ScoringService.get_model() is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as JSON, convert
    it to a list of dictionaries for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from JSON to dict
    if flask.request.content_type == "application/json":
        input = flask.request.data.decode("utf-8")
        input = ast.literal_eval(input)

        data = input["data"]
        pred_type = input["pred_type"]

    else:
        return flask.Response(
            response="This predictor only supports JSON data",
            status=415,
            mimetype="text/plain",
        )

    print("Invoked with {} records".format(len(data)))

    out = io.StringIO()

    if pred_type == "prediction":
        output = ScoringService.predict(data, input["target_node"])
        output.to_csv(out, header=False, index=False)

    elif pred_type == "intervention":
        output = ScoringService.intervention(data)
        pd.DataFrame({"results": output}).to_csv(out, header=False, index=False)

    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")
