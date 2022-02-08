import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(model_dir):

    tokenizer_init = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased').eval().to(device)
    
    return (model, tokenizer_init)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return
    

def predict_fn(input_data, models):

    model_bert, tokenizer = models
    sequence_0 = input_data[0] 
    sequence_1 = input_data[1]
    
    max_length = 512
    tokenized_sequence_pair = tokenizer.encode_plus(sequence_0,
                                                    sequence_1,
                                                    max_length=max_length,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors="pt").to(device)
    
    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs = tokenized_sequence_pair['input_ids'], tokenized_sequence_pair['attention_mask']
    
    with torch.no_grad():
        paraphrase_classification_logits = model_bert(*example_inputs)
    
    classes = ['not paraphrase', 'paraphrase']
    paraphrase_prediction = paraphrase_classification_logits[0][0].argmax().item()
    out_str = 'BERT predicts that "{}" and "{}" are {}'.format(sequence_0, sequence_1, classes[paraphrase_prediction])
    
    return out_str


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    
