# This is the script that will be used in the inference container
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
    print(model_dir)
    logging.info('asdf')

    import tarfile
    tar = tarfile.open(model_dir)
    tar.extractall()
    tar.close()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    config = BertConfig(vocab_size=model.config.vocab_size,
                        num_hidden_layers=architecture_definition['num_layers'],
                        num_attention_heads=architecture_definition['num_heads'],
                        intermediate_size=architecture_definition['num_units'],
                        )
    config.attention_head_size = int(config.hidden_size / model.config.num_attention_heads)

    sub_network = get_final_bert_model(original_model=model, new_model_config=config)

    return {"model": sub_network, "tokenizer": tokenizer}


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