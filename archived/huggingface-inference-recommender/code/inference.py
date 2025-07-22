from transformers import pipeline

CSV_CONTENT_TYPE = 'text/csv'


def model_fn(model_dir):
    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model=model_dir,
        tokenizer=model_dir,
        return_all_scores=True
    )
    return sentiment_analysis


def input_fn(serialized_input_data, content_type=CSV_CONTENT_TYPE):
    if content_type == CSV_CONTENT_TYPE:
        input_data = serialized_input_data.splitlines()
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return


def predict_fn(input_data, model):
    return model(input_data)
