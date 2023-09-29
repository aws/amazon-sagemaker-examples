import os
import json
from pathlib import Path
import torch
import logging
import numpy as np

from sagemaker_inference import encoder
from transformers import AutoTokenizer, AutoModelForTokenClassification


def keystoint(x):
    return {int(k): v for k, v in x.items()}


def model_fn(model_dir):
    """Create our inference task as a delegate to the model.

    This runs only once per one worker.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if not tokenizer:
        raise ValueError("tokenizer not found.")

    with open(os.path.join(model_dir, "integer_to_label.json"), 'r') as fp:
        integer_to_label = json.load(fp, object_hook=keystoint)

    try:
        model = AutoModelForTokenClassification.from_pretrained(model_dir, num_labels=len(integer_to_label))
        if torch.cuda.is_available():
            model.to("cuda:0")
        model.eval()
        return model, tokenizer, integer_to_label
    except Exception:
        logging.exception("Failed to load model from checkpoint")
        raise


def transform_fn(task, input_data, content_type, accept="application/json"):
    """Make predictions against the model and return a serialized response.

    The function signature conforms to the SM contract.

    Args:
        task (obj): model loaded by model_fn, in our case is one of the Task.
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.

    Returns:
        obj: the serialized prediction result or a tuple of the form
            (response_data, content_type)

    """
    if content_type == "application/list-text":
        test_data = json.loads(input_data.decode("utf-8"))
        model, tokenizer, integer_to_label = task
        try:
            tokens = tokenizer(test_data, truncation=True, padding=True, is_split_into_words=True)

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_input_ids = torch.tensor(tokens['input_ids'], device=device)
            torch_input_attention_mask = torch.tensor(tokens['attention_mask'], device=device)

            predictions = model.forward(input_ids=torch_input_ids, attention_mask=torch_input_attention_mask)

            predictions_probabilites = predictions.logits.detach().cpu().numpy()
            predictions_integers = torch.argmax(predictions.logits, axis=-1).detach().cpu().numpy().tolist()
            sum_attention_mask = np.array(tokens["attention_mask"]).sum(axis=1)

            res_token_idxes = []
            res_tokens = []
            res_labels = []
            for idx, i in enumerate(predictions_integers):
                tmp = []
                num_real_tokens = sum_attention_mask[idx]
                for count, j in enumerate(i):
                    if count > num_real_tokens - 1:
                        break
                    tmp.append(integer_to_label[j])

                tmp_tokens = tokens.tokens(idx)[:num_real_tokens]
                tmp_token_idx = tokens.word_ids(idx)[:num_real_tokens]
                assert len(tmp_tokens) == len(tmp_token_idx) == len(tmp)
                res_labels.append(tmp)
                res_tokens.append(tmp_tokens)
                res_token_idxes.append(tmp_token_idx)

            return encoder.encode(
                {
                    "predict_label": np.array(res_labels),
                    "predict_probabilites": predictions_probabilites,
                    "token": np.array(res_tokens),
                    "word_id": np.array(res_token_idxes),
                },
                accept
            )
        except Exception:
            logging.exception("Failed to do transform")
            raise
    raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
