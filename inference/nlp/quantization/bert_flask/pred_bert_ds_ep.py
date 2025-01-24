# This is the file that implements a flask server to do inferences. It's the file that you will modify
# to implement the prediction for your own algorithm.

from __future__ import print_function

import os, sys, stat
import json
import shutil
import flask
from flask import Flask, jsonify, request
import glob
from quantize_with_ds_ep import predict_fn_ep, model_fn_ep, model_fn_ep_fp32

MODEL_PATH = '/opt/ml/model/'
VERSION_INT8 = "INT8"
# Default context in case it is not supplied as part of the request
CONTEXT_TEXT = ("The Panthers finished the regular season with a 15-1 record, and quarterback Cam Newton was named the NFL Most Valuable Player (MVP). "
"They defeated the Arizona Cardinals 49-15 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995."
" The Broncos finished the regular season with a 12-4 record, and "
"denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 20-18 in the AFC Championship Game."
" They joined the Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bowl.")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ClassificationService(object):
    
    @classmethod
    def get_model(cls, version):
        if version == VERSION_INT8:
            return model_fn_ep(MODEL_PATH)
        else:
            return model_fn_ep_fp32(MODEL_PATH)

    @classmethod
    def predict(cls, question, context, version):
        print("*** Predict ***")
        """For the input, do the predictions and return them."""
        model_dict = cls.get_model(version)
        return predict_fn_ep(model_dict, question, context)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ClassificationService.get_model(VERSION_INT8) is not None  

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def predict():
    print("*** invocation/predict entry ***")
    request_json = request.json
    print("*** request_json ***")
    print(request_json)
    if request_json and request_json['context']:
        print(request_json['context'])
        context = request_json['context']
    else:
        print("*** Context not sent in request JSON, using default ***")
        context = CONTEXT_TEXT
    
    if request_json and request_json['question']:
        print(request_json['question'])
        question = request_json['question']
    else:
        print("*** Question not sent in request JSON, using default ***")
        question = "Who denied Patriots??"
    
    if request_json and request_json['version']:
        print(request_json['version'])
        version = request_json['version']
    else:
        print("*** Version not sent in request JSON, using INT8 ***")
        version = VERSION_INT8

    predictions = ClassificationService.predict(question, context, version)
    if predictions is not None:
        return_value = {
            'question': question,
            'answer': predictions
        }
    else:
        return flask.Response(response='\n', status=500, mimetype='application/json')
    
    
    print('*** return value ***')
    print(return_value)
    return jsonify(return_value) 
