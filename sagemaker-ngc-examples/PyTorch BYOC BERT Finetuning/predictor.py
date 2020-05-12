import os
import json
import pickle
import sys
import signal
import traceback
import re
import flask
import numpy as np
import boto3
import tarfile
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling import BertForQuestionAnswering, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)
from types import SimpleNamespace
import torch
from pathlib import Path
import warnings
from helper_funcs import *


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

prefix = "/opt/ml/"

PATH = Path(os.path.join(prefix, "model"))

PRETRAINED_PATH = Path(os.path.join(prefix, "code"))

BERT_PRETRAINED_PATH = (
    PRETRAINED_PATH / "pretrained-weights" / "uncased_L-12_H-768_A-12/"
)

with open('hyperparameters.json') as f:
    params = json.load(f)
    
bucket = params['save_to_s3']

MODEL_PATH = 'model.pth' #"pytorch_model.bin"

vocab_file =  '/workspace/bert/vocab'
config_file = '/workspace/bert/bert_config.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_predictor_model(cls):

        config = BertConfig.from_json_file(config_file)
        model = BertForQuestionAnswering(config)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')["model"])
        model.to(device)
        cls.model = model

        return cls.model

    @classmethod
    def predict(cls, context, question, bing_key=None, max_seq_length=384, max_query_length=64,
               n_best_size=3, do_lower_case=True, can_give_negative_answer=True, max_answer_length=30, 
               null_score_diff_threshold=-11.0):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        predictor_model = cls.get_predictor_model()
        
        doc_tokens = context.split()
        tokenizer = BertTokenizer(vocab_file, do_lower_case=True, max_len=max_seq_length)
        query_tokens = tokenizer.tokenize(question)
        feature = preprocess_tokenized_text(doc_tokens, 
                                            query_tokens, 
                                            tokenizer, 
                                            max_seq_length=max_seq_length, 
                                            max_query_length=max_query_length)
        tensors_for_inference, tokens_for_postprocessing = feature

        input_ids = torch.tensor(tensors_for_inference.input_ids, dtype=torch.long, device=device).unsqueeze(0)
        segment_ids = torch.tensor(tensors_for_inference.segment_ids, dtype=torch.long, device=device).unsqueeze(0)
        input_mask = torch.tensor(tensors_for_inference.input_mask, dtype=torch.long, device=device).unsqueeze(0) 

        # run prediction
        with torch.no_grad():
            start_logits, end_logits = predictor_model(input_ids, segment_ids, input_mask)

        # post-processing
        start_logits = start_logits[0].detach().cpu().tolist()
        end_logits = end_logits[0].detach().cpu().tolist()
        prediction = get_predictions(doc_tokens, tokens_for_postprocessing, 
                                 start_logits, end_logits, n_best_size, 
                                 max_answer_length, do_lower_case, 
                                 can_give_negative_answer, 
                                 null_score_diff_threshold)        

        return prediction

    @classmethod
    def searching_all_files(cls, directory: Path):
        file_list = []  # A list for storing files existing in directories

        for x in directory.iterdir():
            if x.is_file():
                file_list.append(str(x))
            else:
                file_list.append(cls.searching_all_files(x))

        return file_list


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
        ScoringService.get_predictor_model() is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json") # can change to x-npy


# @app.route("/execution-parameters", method=["GET"])
# def get_execution_parameters():
#     params = {
#         "MaxConcurrentTransforms": 3,
#         "BatchStrategy": "MULTI_RECORD",
#         "MaxPayloadInMB": 6,
#     }
#     return flask.Response(
#         response=json.dumps(params), status="200", mimetype="application/json"
#     )


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    #text = None

    if flask.request.content_type == "application/json":
        print("calling json launched")
        data = flask.request.get_json(silent=True)
        print(data)
        #data = json.loads(data)
        #print(data)

        context = data["context"] # need to get the context and question here 
        question = data["question"]
#         try:
#             bing_key = data["bing_key"]
#         except:
#             bing_key = None

    else:
        return flask.Response(
            response="This predictor only supports JSON data",
            status=415,
            mimetype="text/plain",
        )

    #print("Invoked with text: {}.".format(text.encode("utf-8")))

    # Do the prediction
    predictions = ScoringService.predict(context, question) 

    result = json.dumps(predictions) # may need to fix this 

    return flask.Response(response=result, status=200, mimetype="application/json")