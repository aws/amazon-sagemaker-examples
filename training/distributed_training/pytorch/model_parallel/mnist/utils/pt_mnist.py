# Future
from __future__ import print_function

import argparse
import math

# Standard Library
import os
import random
import time

# Third Party
import numpy as np

# First Party
import smdistributed.modelparallel.torch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import StepLR
from torchnet.dataset import SplitDataset
from torchvision import datasets, transforms

# SM Distributed: import scaler from smdistributed.modelparallel.torch.amp, instead of torch.cuda.amp

# Make cudnn deterministic in order to get the same losses across runs.
# The following two lines can be removed if they cause a performance impact.
# For more details, see:
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def aws_s3_sync(source, destination):

    """aws s3 sync in quiet mode and time profile"""
    import subprocess
    import time

    cmd = ["aws", "s3", "sync", "--quiet", source, destination]
    print(f"Syncing files from {source} to {destination}")
    start_time = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    end_time = time.time()
    print("Time Taken to Sync: ", (end_time - start_time))
    return


def sync_local_checkpoints_to_s3(
    local_path="/opt/ml/checkpoints",
    s3_path=os.path.dirname(os.path.dirname(os.getenv("SM_MODULE_DIR", ""))) + "/checkpoints",
):

    """sample function to sync checkpoints from local path to s3"""

    import boto3
    import botocore

    # check if local path exists
    if not os.path.exists(local_path):
        raise RuntimeError("Provided local path {local_path} does not exist. Please check")

    # check if s3 bucket exists
    s3 = boto3.resource("s3")
    if "s3://" not in s3_path:
        raise ValueError("Provided s3 path {s3_path} is not valid. Please check")

    s3_bucket = s3_path.replace("s3://", "").split("/")[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            raise RuntimeError("S3 bucket does not exist. Please check")
    aws_s3_sync(local_path, s3_path)
    return


def sync_s3_checkpoints_to_local(
    local_path="/opt/ml/checkpoints",
    s3_path=os.path.dirname(os.path.dirname(os.getenv("SM_MODULE_DIR", ""))) + "/checkpoints",
):

    """sample function to sync checkpoints from s3 to local path"""

    import boto3
    import botocore

    # creat if local path does not exists
    if not os.path.exists(local_path):
        print(f"Provided local path {local_path} does not exist. Creating...")
        try:
            os.makedirs(local_path)
        except Exception as e:
            raise RuntimeError(f"failed to create {local_path}")

    # check if s3 bucket exists
    s3 = boto3.resource("s3")
    if "s3://" not in s3_path:
        raise ValueError("Provided s3 path {s3_path} is not valid. Please check")

    s3_bucket = s3_path.replace("s3://", "").split("/")[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except botocore.exceptions.ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            raise RuntimeError("S3 bucket does not exist. Please check")
    aws_s3_sync(s3_path, local_path)
    return


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, 1)
        return output


class GroupedNet(nn.Module):
    def __init__(self):
        super(GroupedNet, self).__init__()
        self.net1 = Net1()
        self.net2 = Net2()

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


# SM Distributed: Define smp.step. Return any tensors needed outside.
@smp.step
def train_step(model, scaler, data, target):
    with autocast(1 > 0):
        output = model(data)

    loss = F.nll_loss(output, target, reduction="mean")

    scaled_loss = loss
    model.backward(scaled_loss)
    return output, loss


def train(model, scaler, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # SM Distributed: Move input tensors to the GPU ID used by the current process,
        # based on the set_device call.
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Return value, loss_mb is a StepOutput object
        _, loss_mb = train_step(model, scaler, data, target)

        # SM Distributed: Average the loss across microbatches.
        loss = loss_mb.reduce_mean()

        optimizer.step()

        if smp.rank() == 0 and batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


# SM Distributed: Define smp.step for evaluation.
@smp.step
def test_step(model, data, target):
    output = model(data)
    loss = F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, correct


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # SM Distributed: Moves input tensors to the GPU ID used by the current process
            # based on the set_device call.
            data, target = data.to(device), target.to(device)

            # Since test_step returns scalars instead of tensors,
            # test_step decorated with smp.step will return lists instead of StepOutput objects.
            loss_batch, correct_batch = test_step(model, data, target)
            test_loss += sum(loss_batch)
            correct += sum(correct_batch)

    test_loss /= len(test_loader.dataset)
    if smp.mp_rank() == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return test_loss


def main():
    if not torch.cuda.is_available():
        raise ValueError("The script requires CUDA support, but CUDA not available")
    use_ddp = True
    use_horovod = False

    # Fix seeds in order to get the same losses across runs
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    smp.init()

    # SM Distributed: Set the device to the GPU ID used by the current process.
    # Input tensors should be transferred to this device.
    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda")
    kwargs = {"batch_size": 64}
    kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": False})

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # SM Distributed: Download only on a single process per instance.
    # When this is not present, the file is corrupted by multiple processes trying
    # to download and extract at the same time
    if smp.local_rank() == 0:
        dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    smp.barrier()
    dataset1 = datasets.MNIST("../data", train=True, download=False, transform=transform)

    if (use_ddp or use_horovod) and smp.dp_size() > 1:
        partitions_dict = {f"{i}": 1 / smp.dp_size() for i in range(smp.dp_size())}
        dataset1 = SplitDataset(dataset1, partitions=partitions_dict)
        dataset1.select(f"{smp.dp_rank()}")

    # Download and create dataloaders for train and test dataset
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = GroupedNet()

    # SMP handles the transfer of parameters to the right device
    # and the user doesn't need to call 'model.to' explicitly.
    # model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=4.0)

    # SM Distributed: Use the DistributedModel container to provide the model
    # to be partitioned across different ranks. For the rest of the script,
    # the returned DistributedModel object should be used in place of
    # the model provided for DistributedModel class instantiation.
    model = smp.DistributedModel(model)
    scaler = smp.amp.GradScaler()
    optimizer = smp.DistributedOptimizer(optimizer)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 2):
        train(model, scaler, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        scheduler.step()

    if smp.rank() == 0:
        if os.path.exists("/opt/ml/local_checkpoints"):
            print("-INFO- PATH DO EXIST")
        else:
            os.makedirs("/opt/ml/local_checkpoints")
            print("-INFO- PATH DO NOT EXIST")

    # Waiting the save checkpoint to be finished before run another allgather_object
    smp.barrier()

    if smp.dp_rank() == 0:
        model_dict = model.local_state_dict()
        opt_dict = optimizer.local_state_dict()
        smp.save(
            {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
            f"/opt/ml/local_checkpoints/pt_mnist_checkpoint.pt",
            partial=True,
        )
    smp.barrier()

    if smp.local_rank() == 0:
        print("Start syncing")
        base_s3_path = os.path.dirname(os.path.dirname(os.getenv("SM_MODULE_DIR", "")))
        curr_host = os.getenv("SM_CURRENT_HOST")
        full_s3_path = f"{base_s3_path}/checkpoints/{curr_host}/"
        sync_local_checkpoints_to_s3(local_path="/opt/ml/local_checkpoints", s3_path=full_s3_path)
        print("Finished syncing")


if __name__ == "__main__":
    main()
