import json
import os

import sagemaker
from transformers import AutoModelWithLMHead
from transformers import AutoTokenizer
from transformers import pipeline


def model_fn(model_dir):
    session = sagemaker.Session()
    bucket = os.getenv("MODEL_ASSETS_S3_BUCKET")
    prefix = os.getenv("MODEL_ASSETS_S3_PREFIX")
    session.download_data(path=model_dir, bucket=bucket, key_prefix=prefix)
    model = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir=model_dir)
    tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir=model_dir)
    summarizer = pipeline(task="summarization", model=model, tokenizer=tokenizer)
    model_assets = {"summarizer": summarizer}
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
    input_text = request_body["text"]
    summarizer = model_assets["summarizer"]
    try:
        summaries = summarizer(
            input_text,
            max_length=get_parameter(request_body, "max_length", 130),
            min_length=get_parameter(request_body, "min_length", 30),
            do_sample=get_parameter(request_body, "do_sample", "true") == "true",
        )
    except AssertionError as e:
        print(e)
        # intermittent hugging face issue with beam search:
        # https://github.com/huggingface/transformers/issues/3188
        if "Beam should always be full" in str(e):
            summaries = summarizer(
                input_text,
                max_length=get_parameter(request_body, "max_length", 130),
                min_length=get_parameter(request_body, "min_length", 30),
                do_sample=False,
            )
        else:
            raise e
    summary = summaries[0]["summary_text"]
    return {"summary": summary}


def output_fn(prediction, response_content_type):
    assert response_content_type == "application/json", "accept must be 'application/json'"
    response_body_str = json.dumps(prediction)
    return response_body_str
