import os
from djl_python import Input, Output
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

def load_model(properties):
    global model, tokenizer
    global max_dynamic_batch_size
    
    logging.info("Enter: load_model")
    logging.info(f"properties: {properties}")
    
    # model location on the serving host
    model_location = properties.get("model_id")
    max_dynamic_batch_size = max(int(properties.get("max_dynamic_batch_size", 1)), 1)
    assert (max_dynamic_batch_size & (max_dynamic_batch_size-1) == 0), "max_dynamic_batch_size must be power of 2"
    
    logging.info(f"Creating model and tokenizer using: {model_location}")
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = AutoModelForSequenceClassification.from_pretrained(model_location)
    
    logging.info(f"Move model to device")
    model.to(device='cuda')
    model.eval()
    
    logging.info("Exit: load_model")

        
def run_inference(pairs: list, params: dict={}):
    with torch.no_grad():
        inputs = tokenizer(pairs, padding="max_length", truncation=True, return_tensors='pt', max_length=512)
        inputs.to(device='cuda')
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = scores.detach().cpu().numpy().tolist()

    scores = str(scores)
    return scores

def handle(inputs: Input):
    """
    inputs: Contains the configurations from serving.properties
    """

    if os.getenv("MODEL_LOADED", None) != "true":
        load_model(inputs.get_properties())
        os.environ["MODEL_LOADED"] = "true"

    if inputs.is_empty():
        logging.info(f"handle: input is empty")
        return None

    data = inputs.get_as_json()
    pairs = data["inputs"]
    params = data.get("parameters", {})

    scores = run_inference(pairs, params)
    result = {"output": scores}
    return Output().add_as_json(result)