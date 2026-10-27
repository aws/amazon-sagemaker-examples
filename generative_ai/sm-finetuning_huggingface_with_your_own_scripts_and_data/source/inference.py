# This is the script that will be used in the inference container
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    """
    Load the model and tokenizer for inference
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()

    return {"model": model, "tokenizer": tokenizer}


def predict_fn(input_data, model_dict):
    """
    Make a prediction with the model
    """
    text = input_data.pop("inputs")
    parameters_list = input_data.pop("parameters_list", None)

    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]

    # Parameters may or may not be passed
    input_ids = tokenizer(
        text, truncation=True, padding="longest", return_tensors="pt"
    ).input_ids.to(device)

    if parameters_list:
        predictions = []
        for parameters in parameters_list:
            output = model.generate(input_ids, **parameters)
            predictions.append(tokenizer.batch_decode(output, skip_special_tokens=True))
    else:
        output = model.generate(input_ids)
        predictions = tokenizer.batch_decode(output, skip_special_tokens=True)

    return predictions


def input_fn(request_body, request_content_type):
    """
    Transform the input request to a dictionary
    """
    return json.loads(request_body)
