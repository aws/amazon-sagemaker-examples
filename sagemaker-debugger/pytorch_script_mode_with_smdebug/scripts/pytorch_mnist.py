"""
This script is a simple MNIST training script which uses PyTorch.
It has been orchestrated with Amazon SageMaker Debugger hooks to allow saving tensors during training.
These hooks have been instrumented to read from json configuration that SageMaker will put in the training container.
Configuration provided to the SageMaker python SDK when creating a job will be passed on to the hook.
This allows you to use the same script with differing configurations across different runs.
If you use an official SageMaker Framework container (i.e. AWS Deep Learning Container), then
you do not have to orchestrate your script as below. Hooks will automatically be added in those environments.
For more information, please refer to https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md
"""


from __future__ import absolute_import
import argparse
import logging
import os
import sys

import cv2 as cv
import sagemaker_containers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

# SageMaker Debugger: Import the package
import smdebug.pytorch as smd

import numpy as np
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class Net(nn.Module):
    def __init__(self):
        logger.info("Create neural network module")

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def parse_args():
    env = sagemaker_containers.training_env()
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument('--data_dir', type=str)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of Epochs")

    # SageMaker Debugger: Mention the path where you would like the tensors to be
    # saved.
    parser.add_argument(
        "--smdebug_path",
        type=str,
        default=None,
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Reduce the number of training "
             "and evaluation steps to the give number if desired."
             "If this is not passed, trains for one epoch "
             "of training and validation data",
    )
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    opt = parser.parse_args()
    return opt


def _get_train_data_loader(batch_size, training_dir):
    logger.info("Get train data loader")
    dataset = datasets.MNIST(training_dir, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=4)


def _get_test_data_loader(test_batch_size, training_dir):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=False, num_workers=4)


# SageMaker Debugger: This function created the debug hook required to log tensors.
# In this example, weight, gradients and losses will be logged at steps 1,2, and 3,
# and saved to the output directory specified in hyperparameters.
def create_smdebug_hook():
    # This allows you to create the hook from the configuration you pass to the SageMaker pySDK
    hook = smd.Hook.create_from_json_file()
    return hook


def train(model, device, optimizer, hook, epochs, log_interval, training_dir):
    criterion = nn.CrossEntropyLoss()
    # SageMaker Debugger: If you are using a Loss module and would like to save the
    # values as we are doing in this example, then add a call to register loss.
    hook.register_loss(criterion)

    trainloader = _get_train_data_loader(4, training_dir)
    validloader = _get_test_data_loader(4, training_dir)

    for epoch in range(epochs):
        model.train()
        hook.set_mode(smd.modes.TRAIN)
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(trainloader.sampler),
                           100. * i / len(trainloader), loss.item()))

        test(model, hook, validloader, device, criterion)


def test(model, hook, test_loader, device, loss_fn):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.debug('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    opt = parse_args()

    if opt.random_seed:
        torch.manual_seed(128)
        random.seed(12)
        np.random.seed(2)

    training_dir = opt.data_dir

    device = torch.device("cpu")
    model = Net().to(device)

    # SageMaker Debugger: Create the debug hook,
    # and register the hook to save tensors.
    hook = create_smdebug_hook()
    hook.register_hook(model)

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum)
    train(model, device, optimizer, hook, opt.epochs, opt.log_interval, training_dir)
    print("Training is complete")

if __name__ == "__main__":
    main()
