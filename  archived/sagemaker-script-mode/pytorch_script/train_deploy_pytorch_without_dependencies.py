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


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    # Data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


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

    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)
    train_ds = TensorDataset(x_train, y_train)

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    logger.info(
        "batch_size = {}, epochs = {}, learning rate = {}".format(batch_size, epochs, learning_rate)
    )

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    model = get_model()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for x_train_batch, y_train_batch in train_dl:
            y = model(x_train_batch.float())
            loss = criterion(y.flatten(), y_train_batch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch += 1
        logger.info(f"epoch: {epoch} -> loss: {loss}")

    # evalutate on test set
    with torch.no_grad():
        y = model(x_test.float()).flatten()
        mse = ((y - y_test) ** 2).sum() / y_test.shape[0]
    print("\nTest MSE:", mse.numpy())

    torch.save(model.state_dict(), args.model_dir + "/model.pth")
    # PyTorch requires that the inference script must
    # be in the .tar.gz model file and Step Functions SDK doesn't do this.
    inference_code_path = args.model_dir + "/code/"

    if not os.path.exists(inference_code_path):
        os.mkdir(inference_code_path)
        logger.info("Created a folder at {}!".format(inference_code_path))

    shutil.copy("train_deploy_pytorch_without_dependencies.py", inference_code_path)
    shutil.copy("pytorch_model_def.py", inference_code_path)
    logger.info("Saving models files to {}".format(inference_code_path))


if __name__ == "__main__":

    args, _ = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train()
