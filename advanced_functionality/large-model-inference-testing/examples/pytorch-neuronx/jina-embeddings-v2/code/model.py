import os
from djl_python import Input, Output
from transformers import AutoModel, AutoTokenizer
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
        texts = [ example_text for _ in range(batch_size) ]
        _bucket_batch_inference(texts)

def load_model(properties):
    global model, tokenizer
    global max_dynamic_batch_size
    global example_text
    global max_model_len
    
    logging.info("Enter: load_model")
    logging.info(f"properties: {properties}")
    
    example_text = "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    
    # model location on the serving host
    model_location = properties.get("model_id")
    max_dynamic_batch_size = max(int(properties.get("max_dynamic_batch_size", 1)), 1)
    assert (max_dynamic_batch_size & (max_dynamic_batch_size-1) == 0), "max_dynamic_batch_size must be power of 2"
    
    max_model_len = int(properties.get("max_model_len", 8192))
    assert max_model_len > 0, "max_model_len must be greater than 0"

    logging.info(f"Creating model and tokenizer using: {model_location}")
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = AutoModel.from_pretrained(model_location)
    model.eval()

    logging.info(f"Move model to device")
    path = os.getcwd()
    os.chdir("/tmp")
    
    model.to(xm.xla_device())
    compile_model()
    
    os.chdir(path)
    logging.info("Exit: load_model")

def _bucket_batch_inference(texts:list, params: dict={}) -> list:
    
    # params are ignored for our case
    # we want 'padding' to be always 'max_length', max_length to be always 512
    # and 'truncation' to be always 'True'
    
    with torch.no_grad():
        inputs = tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt', max_length=max_model_len)
        inputs.to(xm.xla_device())
        embeddings = model(**inputs, return_dict=True)
        embeddings = embeddings['pooler_output'].detach().cpu().tolist()
        return embeddings
        
def run_inference(texts: list, params: dict={}):
    batch_size = len(texts)
    
    bucket_batch_size = min(min_power_of_2(batch_size), max_dynamic_batch_size)
    texts.extend([ example_text for _ in range(bucket_batch_size - batch_size) ] )
    embeddings  = _bucket_batch_inference(texts, params)
    embeddings = embeddings[:batch_size]
    return embeddings

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
    texts = data["inputs"]

    embeddings = run_inference(texts)
    result = {"output": embeddings}
    return Output().add_as_json(result)