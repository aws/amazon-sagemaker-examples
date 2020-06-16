import os

import torch

from model import Net


def model_fn(model_dir):
    """A function required by SageMaker to deserialize models."""
    model = Net()
    file_path = os.path.join(model_dir, "model.pth")
    state_dict = torch.load(file_path, map_location="cpu")
    model.load_state_dict(state_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)
