# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
#
# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback
import json
import flask

from io import StringIO, BytesIO

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import logging

import boto3

MAX_LEN = 100
prefix = '/opt/ml/'
model_path = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    def __init__(self):
        # This bucket should be updated based on the value in Part 2: Bring Your Own Model to an Active Learning Workflow
        # notebook after the preprocessing is done.
        tokenizer_bucket = '<Update tokenizer bucket here>'
        tokenizer_key = 'sagemaker-byoal/tokenizer.pickle'
        pickle_file_name = tokenizer_key.split('/')[-1]
        boto3.resource('s3').Bucket(tokenizer_bucket).download_file(tokenizer_key, pickle_file_name)
        with open(pickle_file_name, 'rb') as handle:
           self.tokenizer = pickle.load(handle)
        print("Successfully initialized tokenizer.")
    
    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""

        if self.model is None:
            self.model = tf.keras.models.load_model(os.path.join(model_path, "keras_news_classifier_model.h5"))
            print("Successfully loaded model.")

        return self.model


    def predict(self, input):
        """For the input, do the predictions and return them.

        Args:
            input (a single news headline): The data on which to do the predictions. """
        model = self.get_model()
        seq = self.tokenizer.texts_to_sequences([input])
        d = pad_sequences(seq, maxlen=MAX_LEN)
        prediction = model.predict(np.array(d))
        print("prediction received {}".format(prediction))

        probs = np.array(prediction).flatten()
        descending_sorted_index = (-probs).argsort()
        return {
                  "label": ["__label__{}".format(index) for index in descending_sorted_index],
                  "prob": list(probs[descending_sorted_index].astype(float))
               }

# The flask app for serving predictions
app = flask.Flask(__name__)
app.logger.setLevel(logging.DEBUG)
scoring_service = ScoringService()

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = scoring_service.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

def _load_json_instance(instance):
    source = instance.get('source')
    if source is None:
        print("Instance does not have source. Unexpected input to batch transform {}".format(instance))
        return None
    return source.encode('utf-8').decode('utf-8')

def _dump_jsonlines_entry(prediction):
    return (json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single news headline. """
    data = None

    if flask.request.content_type == 'application/jsonlines':
        payload = flask.request.data
        if len(payload) == 0:
            return flask.Response(response="", status=204)

        print("prediction input size in bytes:{} content:{}".format(len(payload), payload))

        fr = StringIO(payload.decode("utf-8"))
        texts = [_load_json_instance(json.loads(line)) for line in iter(lambda: fr.readline(), "")]

        predictions = [scoring_service.predict(text[0]) for text in texts if text is not None]

        bio = BytesIO()
        for line in predictions:
            bio.write(_dump_jsonlines_entry(line))
        return flask.Response(response=bio.getvalue(), status=200, mimetype="application/jsonlines")
    else:
        return flask.Response(response='This predictor only supports application/jsonlines format', status=415, mimetype='text/plain')
