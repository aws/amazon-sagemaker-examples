import json
from transformers import pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)
import os

JSON_CONTENT_TYPE = 'application/json'

def model_fn(model_dir):
    print("LOADING MODEL")
    mapping_file_path = os.path.join(model_dir, "index_to_name.json")

    setup_config_path = os.path.join(model_dir, "setup_config.json")

    with open(setup_config_path) as f:
        setup_config = json.load(f)

    model_name = setup_config.get("model_name")

    model =  AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer =  AutoTokenizer.from_pretrained(model_name ,do_lower_case= True)
    return (model, tokenizer, mapping_file_path)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    print("INPUT1")
    if content_type == JSON_CONTENT_TYPE:
        print("INPUT2")
        input_data = json.loads(serialized_input_data)
        return input_data

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model_pack):
    
    print('Got input Data: {}'.format(input_data))
    model = model_pack[0]
    tokenizer = model_pack[1]
    mapping_file_path = model_pack[2]
    
    with open(mapping_file_path) as f:
        mapping = json.load(f)
   
    inputs = tokenizer.encode_plus(input_data, max_length=128, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt') 
    output = model(inputs['input_ids'])
    
    print("PRED", output)
    
    inferences = []
    num_rows, num_cols = output[0].shape
    
    for i in range(num_rows):
        out = output[0][i].unsqueeze(0)
        y_hat = out.argmax(1).item()
        predicted_idx = str(y_hat)
        inferences.append(mapping.get(predicted_idx))

    output = inferences

    return output


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    print("PREDICTION", prediction_output)
    
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
