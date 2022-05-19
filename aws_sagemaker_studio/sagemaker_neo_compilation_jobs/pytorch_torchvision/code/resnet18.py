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
    result = np.squeeze(result.cpu().numpy())
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())
    content_type = "application/json"

    return response_body, content_type


# Defining a model_fn is unnecessary here, because the model can be loaded by default.
# If you must load your model with a custom function, you can define a model_fn to
# load the model as follows:
# def model_fn(model_dir):
#     ...
