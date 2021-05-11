# Python Built-Ins:
import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
from distutils.dir_util import copy_tree
from tempfile import TemporaryDirectory

# External Dependencies:
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import MNISTNet
from packaging import version as pkgversion
from sagemaker_pytorch_serving_container import handler_service as default_handler_service
from torch.utils.data import DataLoader, Dataset

# Local Dependencies:
from inference import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def enable_sm_oneclick_deploy(model_dir):
    """Copy current running source code folder to model_dir, to enable Estimator.deploy()
    PyTorch framework containers will load custom inference code if:
    - The code exists in a top-level code/ folder in the model.tar.gz
    - The entry point argument matches an existing file
    ...So to make one-click estimator.deploy() work (without creating a PyTorchModel first), we need
    to:
    - Copy the current working directory to model_dir/code
    - `from inference import *` because "train.py" will still be the entry point (same as the training job)
    """
    code_path = os.path.join(model_dir, "code")
    logger.info(f"Copying working folder to {code_path}")
    for currpath, dirs, files in os.walk("."):
        for file in files:
            # Skip any filenames starting with dot:
            if file.startswith("."):
                continue
            filepath = os.path.join(currpath, file)
            # Skip any pycache or dot folders:
            if ((os.path.sep + ".") in filepath) or ("__pycache__" in filepath):
                continue
            relpath = filepath[len(".") :]
            if relpath.startswith(os.path.sep):
                relpath = relpath[1:]
            outpath = os.path.join(code_path, relpath)
            logger.info(f"Copying {filepath} to {outpath}")
            os.makedirs(outpath.rpartition(os.path.sep)[0], exist_ok=True)
            shutil.copy2(filepath, outpath)
    return code_path


def enable_torchserve_multi_model(model_dir, handler_service_file=default_handler_service.__file__):
    """Package the contents of model_dir as a TorchServe model archive
    SageMaker framework serving containers for PyTorch versions >=1.6 use TorchServe, for consistency with
    the PyTorch ecosystem. TorchServe expects particular 'model archive' packaging around models.

    On single-model endpoints, the SageMaker container can transparently package your model.tar.gz for
    TorchServe at start-up. On multi-model endpoints though, as models are loaded and unloaded dynamically,
    this is not (currently?) supported.

    ...So to make your training jobs produce model.tar.gz's which are already compatible with TorchServe
    (and therefore SageMaker Multi-Model-Endpoints, on PyTorch >=1.6), you can do something like this.

    Check out the PyTorch Inference Toolkit (used by SageMaker PyTorch containers) for more details:
    https://github.com/aws/sagemaker-pytorch-inference-toolkit

    For running single-model endpoints, or MMEs on PyTorch<1.6, this function is not necessary.

    If you use the SageMaker PyTorch framework containers, you won't need to change `handler_service_file`
    unless you already know about the topic :-)  The default handler will already support `model_fn`, etc.
    """
    if pkgversion.parse(torch.__version__) >= pkgversion.parse("1.6"):
        logger.info(f"Packaging {model_dir} for use with TorchServe")
        # torch-model-archiver creates a subdirectory per `model-name` within `export-path`, but we want the
        # contents to end up in `model_dir`'s root - so will use a temp dir and copy back:
        with TemporaryDirectory() as temp_dir:
            ts_model_name = "model"  # Just a placeholder, doesn't really matter for our purposes
            subprocess.check_call(
                [
                    "torch-model-archiver",
                    "--model-name",
                    ts_model_name,
                    "--version",
                    "1",
                    "--handler",
                    handler_service_file,
                    "--extra-files",
                    model_dir,
                    "--archive-format",
                    "no-archive",
                    "--export-path",
                    temp_dir,
                ]
            )
            copy_tree(os.path.join(temp_dir, ts_model_name), model_dir)
    else:
        logger.info(f"Skipping TorchServe repackage: PyTorch version {torch.__version__} < 1.6")


def normalize(x, axis):
    eps = np.finfo(float).eps
    mean = np.mean(x, axis=axis, keepdims=True)
    # avoid division by zero
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std


def convert_to_tensor(data_dir, images_file, labels_file):
    """Byte string to torch tensor"""
    with gzip.open(os.path.join(data_dir, images_file), "rb") as f:
        images = (
            np.frombuffer(
                f.read(),
                np.uint8,
                offset=16,
            )
            .reshape(-1, 28, 28)
            .astype(np.float32)
        )

    with gzip.open(os.path.join(data_dir, labels_file), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    # normalize the images
    images = normalize(images, axis=(1, 2))

    # add channel dimension (depth-major)
    images = np.expand_dims(images, axis=1)

    # to torch tensor
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)
    return images, labels


class MNIST(Dataset):
    def __init__(self, data_dir, train=True):
        """PyTorch Dataset for example MNIST files

        Loads and decodes the expected gzip file names from data_dir
        """
        if train:
            images_file = "train-images-idx3-ubyte.gz"
            labels_file = "train-labels-idx1-ubyte.gz"
        else:
            images_file = "t10k-images-idx3-ubyte.gz"
            labels_file = "t10k-labels-idx1-ubyte.gz"

        self.images, self.labels = convert_to_tensor(data_dir, images_file, labels_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda > 0 else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = DataLoader(
        MNIST(args.train, train=True), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        MNIST(args.test, train=False), batch_size=args.test_batch_size, shuffle=False
    )

    net = MNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        betas=(args.beta_1, args.beta_2),
        weight_decay=args.weight_decay,
    )

    logger.info("Start training ...")
    for epoch in range(1, args.epochs + 1):
        net.train()
        for batch_idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            output = net(imgs)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(imgs),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        # test the model
        test(net, test_loader, device)

    # save model checkpoint
    save_model(net, args.model_dir)
    return


def test(model, test_loader, device):
    """Evaluate `model` on the test set and log metrics to console"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            test_loss += F.cross_entropy(output, labels, reduction="sum").item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{}, {})\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return


def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    logger.info(f"Saving model to {path}")
    torch.save(model.cpu().state_dict(), path)
    enable_sm_oneclick_deploy(model_dir)
    enable_torchserve_multi_model(model_dir)
    return


def parse_args():
    """Load SageMaker training job (hyper)-parameters from CLI and environment variables"""
    parser = argparse.ArgumentParser()

    # Training procedure parameters:
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--beta_1",
        type=float,
        default=0.9,
        metavar="BETA1",
        help="beta1 (default: 0.9)",
    )
    parser.add_argument(
        "--beta_2",
        type=float,
        default=0.999,
        metavar="BETA2",
        help="beta2 (default: 0.999)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="WD",
        help="L2 weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    #     parser.add_argument("--backend", type=str, default=None,
    #         help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    #     )

    # I/O folders:
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    # Container environment:
    #     parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    #     parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
