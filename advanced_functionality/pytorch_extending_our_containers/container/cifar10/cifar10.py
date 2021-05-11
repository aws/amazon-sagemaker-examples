# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import argparse
import ast
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.models
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py#L118
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _train(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.dist_backend, dist.get_world_size()
            )
            + "Current host rank is {}. Using cuda: {}. Number of gpus: {}".format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus
            )
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    logger.info("Loading Cifar10 dataset")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    logger.info("Model loaded")
    model = Net()

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(0, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")
    return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def model_fn(model_dir):
    logger.info("model_fn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, metavar="BS", help="batch size (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--dist-backend", type=str, default="gloo", help="distributed backend (default: gloo)"
    )

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument("--hosts", type=str, default=ast.literal_eval(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    _train(parser.parse_args())
