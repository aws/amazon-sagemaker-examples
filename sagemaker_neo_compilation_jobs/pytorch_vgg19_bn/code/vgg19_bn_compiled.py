import io
import json
import logging
import os
import pickle

import neopytorch
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image  # Training container doesn't have this package

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def transform_fn(model, payload, request_content_type, response_content_type):

    logger.info("Invoking user-defined transform function")

    if request_content_type != "application/octet-stream":
        raise RuntimeError(
            "Content type must be application/octet-stream. Provided: {0}".format(
                request_content_type
            )
        )

    # preprocess
    decoded = Image.open(io.BytesIO(payload))
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    normalized = preprocess(decoded)
    batchified = normalized.unsqueeze(0)

    # predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchified = batchified.to(device)
    result = model.forward(batchified)

    # Softmax (assumes batch size 1)
    result = np.squeeze(result.cpu().detach().numpy())
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())
    content_type = "application/json"

    return response_body, content_type


def model_fn(model_dir):

    logger.info("model_fn")
    neopytorch.config(model_dir=model_dir, neo_runtime=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The compiled model is saved as "compiled.pt"
    model = torch.jit.load(os.path.join(model_dir, "compiled.pt"), map_location=device)

    # It is recommended to run warm-up inference during model load
    sample_input_path = os.path.join(model_dir, "sample_input.pkl")
    with open(sample_input_path, "rb") as input_file:
        model_input = pickle.load(input_file)
    if torch.is_tensor(model_input):
        model_input = model_input.to(device)
        model(model_input)
    elif isinstance(model_input, tuple):
        model_input = (inp.to(device) for inp in model_input if torch.is_tensor(inp))
        model(*model_input)
    else:
        print("Only supports a torch tensor or a tuple of torch tensors")

    return model
