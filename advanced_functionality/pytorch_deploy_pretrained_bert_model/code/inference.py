import json
from transformers import pipeline

JSON_CONTENT_TYPE = 'application/json'


def model_fn(model_dir):
    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model=model_dir,
        tokenizer=model_dir,
        return_all_scores=True
    )

    return sentiment_analysis


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model):
    print('Got input Data: {}'.format(input_data))
    return model(input_data)


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept

    raise Exception('Requested unsupported ContentType in Accept: ' + accept)