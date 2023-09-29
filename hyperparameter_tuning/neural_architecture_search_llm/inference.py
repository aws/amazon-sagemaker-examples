# This is the script that will be used in the inference container
import os
import json
import torch
import logging
logging.basicConfig(level=logging.INFO)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.bert.modeling_bert import BertConfig

from extract_subnetworks import get_final_bert_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    """
    Load the model and tokenizer for inference
    """

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    
    architecture_definition = json.loads(os.environ['SM_HPS'])
    config = BertConfig(vocab_size=model.config.vocab_size,
                        num_hidden_layers=architecture_definition['num-layers'],
                        num_attention_heads=architecture_definition['num-heads'],
                        intermediate_size=architecture_definition['num-units'],
                        )
    config.attention_head_size = int(config.hidden_size / model.config.num_attention_heads)

    sub_network = get_final_bert_model(original_model=model, new_model_config=config).to(device).eval()

    return {"model": sub_network, "tokenizer": tokenizer}


def predict_fn(input_data, model_dict):
    """
    Make a prediction with the model
    """
    text = input_data.pop("inputs")

    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]

    # Parameters may or may not be passed
    input_ids = tokenizer(
        text, truncation=True, padding="max_length", return_tensors="pt"
    ).input_ids.to(device)

    predictions = model.forward(input_ids)

    return predictions


def input_fn(request_body, request_content_type):
    """
    Transform the input request to a dictionary
    """
    return json.loads(request_body)