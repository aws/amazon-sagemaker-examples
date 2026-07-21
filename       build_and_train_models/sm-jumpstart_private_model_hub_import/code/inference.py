import torch
import torch.nn as nn
import os
import json


# Define the ANN model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def model_fn(model_dir):
    """Load the PyTorch model from the model_dir."""
    model = ANN()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    model.eval()
    return model


def input_fn(request_body, content_type):
    """Process the incoming request body."""
    if content_type == "application/json":
        data = json.loads(request_body)
        return torch.tensor(data["features"], dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def output_fn(prediction, accept):
    """Format the model's prediction for the response."""
    if accept == "application/json":
        return {"output": prediction.item()}
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


def predict_fn(input_data, model):
    """Perform the prediction using the loaded model."""
    with torch.inference_mode():
        output = model(input_data)
        y_pred = (output > 0.5).float()
    return y_pred
