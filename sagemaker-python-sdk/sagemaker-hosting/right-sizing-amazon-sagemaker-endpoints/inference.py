import json
import torch
from PIL import Image
from torchvision import transforms
import io
import numpy as np


# defines the model
def model_fn(model_dir):
    model_path = f"{model_dir}/infer-pytorch-ic-resnet50.pt"
    model = torch.load(model_path)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""

    if request_content_type == "application/json":

        input_data = json.loads(request_body)

        input_image = Image.open(io.BytesIO(eval(input_data["data"])))

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        # torch.load(BytesIO(request_body))

        return input_batch
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device))
