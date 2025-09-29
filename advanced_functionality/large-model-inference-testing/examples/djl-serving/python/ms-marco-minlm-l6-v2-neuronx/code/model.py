import os
from djl_python import Input, Output
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
import torch_xla.core.xla_model as xm
import math

def powers_of_2(n):
    return [2**i for i in range(int(math.log2(n))+1)]

def min_power_of_2(n):
    return 2**math.ceil(math.log2(n))

def compile_model():
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    bucket_list = powers_of_2(max_dynamic_batch_size)
    for batch_size in bucket_list:
        print(f"Compiling model for batch size: {batch_size}")
        pairs = [ example_pair for _ in range(batch_size) ]
        _bucket_batch_inference(pairs)

def load_model(properties):
    global model, tokenizer
    global max_dynamic_batch_size
    global example_pair
    
    logging.info("Enter: load_model")
    logging.info(f"properties: {properties}")
    
    
    example_pair = ['what is panda?', 
              'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
    
    # model location on the serving host
    model_location = properties.get("model_id")
    max_dynamic_batch_size = max(int(properties.get("max_dynamic_batch_size", 1)), 1)
    assert (max_dynamic_batch_size & (max_dynamic_batch_size-1) == 0), "max_dynamic_batch_size must be power of 2"
    
    logging.info(f"Creating model and tokenizer using: {model_location}")
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = AutoModelForSequenceClassification.from_pretrained(model_location)
    model.eval()

    logging.info(f"Move model to device")
    path = os.getcwd()
    os.chdir("/tmp")
    
    model.to(xm.xla_device())
    compile_model()
    
    os.chdir(path)
    logging.info("Exit: load_model")

def _bucket_batch_inference(pairs:list, params: dict={}) -> list:
    
    # params are ignored for our case
    # we want 'padding' to be always 'max_length', max_length to be always 512
    # and 'truncation' to be always 'True'
    
    with torch.no_grad():
        inputs = tokenizer(pairs, padding="max_length", truncation=True, return_tensors='pt', max_length=512)
        inputs.to(xm.xla_device())
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = scores.detach().cpu().numpy().tolist()
        return scores
        
def run_inference(pairs: list, params: dict={}):
    batch_size = len(pairs)
    
    bucket_batch_size = min(min_power_of_2(batch_size), max_dynamic_batch_size)
    pairs.extend([ example_pair for _ in range(bucket_batch_size - batch_size) ] )
    scores  = _bucket_batch_inference(pairs, params)
    scores = scores[:batch_size]
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