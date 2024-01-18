import json
import os

import sagemaker
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline


def model_fn(model_dir):
    session = sagemaker.Session()
    bucket = os.getenv("MODEL_ASSETS_S3_BUCKET")
    prefix = os.getenv("MODEL_ASSETS_S3_PREFIX")
    session.download_data(path=model_dir, bucket=bucket, key_prefix=prefix)
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad", cache_dir=model_dir
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad", cache_dir=model_dir
    )
    answerer = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
    model_assets = {"answerer": answerer}
    return model_assets


def input_fn(request_body_str, request_content_type):
    assert request_content_type == "application/json", "content_type must be 'application/json'"
    request_body = json.loads(request_body_str)
    return request_body


def get_parameter(request_body, parameter_name, default):
    parameter = default
    if "parameters" in request_body:
        if parameter_name in request_body["parameters"]:
            parameter = request_body["parameters"][parameter_name]
    return parameter


def predict_fn(request_body, model_assets):
    question = request_body["question"]
    context = request_body["context"]
    answerer = model_assets["answerer"]
    answers = answerer(question=question, context=context, topk=get_parameter(request_body, "topk", 3))
    return {"answers": answers}


def output_fn(prediction, response_content_type):
    assert response_content_type == "application/json", "accept must be 'application/json'"
    response_body_str = json.dumps(prediction)
    return response_body_str
