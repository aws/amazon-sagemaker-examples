# flake8: noqa
import argparse
import numpy as np
import os
import sys
import logging
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_model_def import get_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "/opt/ml/input/data"


def get_train_data(train_dir):
    """
    Get the training data and convert to tensors
    """

    x_train = np.load(os.path.join(train_dir, "x_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    logger.info(f"x train: {x_train.shape}, y train: {y_train.shape}")

    return torch.from_numpy(x_train), torch.from_numpy(y_train)


def get_test_data(test_dir):
    """
    Get the testing data and convert to tensors
    """

    x_test = np.load(os.path.join(test_dir, "x_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))
    logger.info(f"x test: {x_test.shape}, y test: {y_test.shape}")

    return torch.from_numpy(x_test), torch.from_numpy(y_test)


def model_fn(model_dir):
    """
    Load the model for inference
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model.load_state_dict(torch.load(model_dir + "/model.pth"))
    model.eval()
    return model.to(device)


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    if request_content_type == "application/json":
        request = json.loads(request_body)
        train_inputs = torch.tensor(request)
        return train_inputs


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.float()).numpy()[0]


def train():
    """
    Train the PyTorch model
    """
    # Directories: train, test and model
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    model_dir = os.environ.get("SM_MODEL_DIR", os.path.join(current_dir, "data/model"))

    # Load the training and testing data
    x_train, y_train = get_train_data(train_dir)
    x_test, y_test = get_test_data(test_dir)
    train_ds = TensorDataset(x_train, y_train)

    # Training parameters - used to configure the training loop
    batch_size = 64
    epochs = 1
    learning_rate = 0.1
    logger.info(
        "batch_size = {}, epochs = {}, learning rate = {}".format(batch_size, epochs, learning_rate)
    )

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Define the model, loss function and optimizer
    model = get_model()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        for x_train_batch, y_train_batch in train_dl:
            y = model(x_train_batch.float())
            loss = criterion(y.flatten(), y_train_batch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch += 1
        logger.info(f"epoch: {epoch} -> loss: {loss}")

    # Test the model
    with torch.no_grad():
        y = model(x_test.float()).flatten()
        mse = ((y - y_test) ** 2).sum() / y_test.shape[0]
    print("\nTest MSE:", mse.numpy())

    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_dir + "/model.pth")
    inference_code_path = model_dir + "/code/"

    if not os.path.exists(inference_code_path):
        os.mkdir(inference_code_path)
        logger.info("Created a folder at {}!".format(inference_code_path))

    code_dir = os.environ.get("SM_CHANNEL_CODE", current_dir)
    shutil.copy(os.path.join(code_dir, "custom_script.py"), inference_code_path)
    shutil.copy(os.path.join(code_dir, "pytorch_model_def.py"), inference_code_path)
    logger.info("Saving models files to {}".format(inference_code_path))


if __name__ == "__main__":
    print("Running the training job ...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train()
