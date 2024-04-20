import os
from djl_python import Input, Output
from transformers_neuronx import LlamaForSampling
from transformers import AutoTokenizer
import torch
import logging

def load_model(properties):
    global n_positions

    logging.info("Enter: load_model")
    logging.info(f"properties: {properties}")
    
    # model location on the serving host
    model_location = properties["model_id"]
    amp=properties.get("amp", properties.get("dtype", "f16"))
    tp_degree = int(properties.get("tensor_parallel_degree", 8))
    n_positions = int(properties.get("n_positions", 2048))
    logging.info(f"Creating model and tokenizer using: {model_location}")
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = LlamaForSampling.from_pretrained(model_location,
                                                       n_positions=n_positions, 
                                                       tp_degree=tp_degree, 
                                                       amp=amp)
      
    logging.info(f"Move model to Neuron device")
    path = os.getcwd()
    os.chdir("/tmp")
    model.to_neuron()
    os.chdir(path)
    
    logging.info("Exit: load_model")
    return model, tokenizer


def run_inference(model, tokenizer, prompt, params):
    tokenizer.pad_token = tokenizer.eos_token
    encoded_inputs = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    
    sequence_length = params.pop("sequence_length", n_positions)
    with torch.inference_mode():
        generated_token_seqs = model.sample(encoded_inputs['input_ids'], 
                                 sequence_length=sequence_length, **params)
        generated_text_seqs = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_token_seqs]
        
    return generated_text_seqs


def handle(inputs: Input):
    """
    inputs: Contains the configurations from serving.properties
    """

    global model, tokenizer

    if os.getenv("MODEL_LOADED", None) != "true":
        model, tokenizer = load_model(inputs.get_properties())
        os.environ["MODEL_LOADED"] = "true"

    if inputs.is_empty():
        logging.info(f"handle: input is empty")
        return None

    data = inputs.get_as_json()

    prompt = data["inputs"]
    params = data["parameters"]

    outputs = run_inference(model, tokenizer, prompt, params)
    result = {"outputs": outputs}
    return Output().add_as_json(result)