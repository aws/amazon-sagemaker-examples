from io import StringIO
import numpy as np
import os
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Any, Dict, List


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load the model for inference
    """
    model_path = os.path.join(model_dir, "model")

    # Load HuggingFace tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load HuggingFace model from disk.
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model_dict = {"model": model, "tokenizer": tokenizer}
    return model_dict


def predict_fn(input_data: List, model: Dict) -> np.ndarray:
    """
    Apply model to the incoming request
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = model["tokenizer"]
    huggingface_model = model["model"]

    encoded_input = tokenizer(input_data, truncation=True, padding=True, max_length=128, return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = huggingface_model(input_ids=encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])
        res = torch.nn.Softmax(dim=1)(output.logits).detach().cpu().numpy()[:, 1]
        return res


def input_fn(request_body: str, request_content_type: str) -> List[str]:
    """
    Deserialize and prepare the prediction input
    """
    if request_content_type == "application/json":
        sentences = [json.loads(request_body)]
        
    elif request_content_type == "text/csv":
        # We have a single column with the text.
        sentences = list(pd.read_csv(StringIO(request_body), header=None).values[:, 0].astype(str))
    else:
        sentences = request_body
    return sentences
