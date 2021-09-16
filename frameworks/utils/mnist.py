import gzip
import json
import os
from urllib import request

import boto3
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(dirname, "config.json"), "r") as f:
    CONFIG = json.load(f)


def mnist_to_numpy(data_dir="/tmp/data", train=True):
    """Download MNIST dataset and convert it to numpy array

    Args:
        data_dir (str): directory to save the data
        train (bool): download training set

    Returns:
        tuple of images and labels as numpy arrays
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if train:
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"

    # download objects
    s3 = boto3.client("s3")
    bucket = CONFIG["public_bucket"]
    for obj in [images_file, labels_file]:
        key = os.path.join("datasets/image/MNIST", obj)
        dest = os.path.join(data_dir, obj)
        if not os.path.exists(dest):
            s3.download_file(bucket, key, dest)

    return _convert_to_numpy(data_dir, images_file, labels_file)


def _convert_to_numpy(data_dir, images_file, labels_file):
    """Byte string to numpy arrays"""
    with gzip.open(os.path.join(data_dir, images_file), "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(data_dir, labels_file), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return (images, labels)


def normalize(x, axis):
    eps = np.finfo(float).eps

    mean = np.mean(x, axis=axis, keepdims=True)
    # avoid division by zero
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std


def adjust_to_framework(x, framework="pytorch"):
    """Adjust a ``numpy.ndarray`` to be used as input for specified framework

    Args:
        x (numpy.ndarray): Batch of images to be adjusted
            to follow the convention in pytorch / tensorflow / mxnet

        framework (str): Framework to use. Takes value in
            ``pytorch``, ``tensorflow`` or ``mxnet``
    Return:
        numpy.ndarray following the convention of tensors in the given
        framework
    """

    if x.ndim == 3:
        # input is gray-scale
        x = np.expand_dims(x, 1)

    if framework in ["pytorch", "mxnet"]:
        # depth-major
        return x
    elif framework == "tensorlfow":
        # depth-minor
        return np.transpose(x, (0, 2, 3, 1))
    elif framework == "mxnet":
        return x
    else:
        raise ValueError(
            "framework must be one of " + "[pytorch, tensorflow, mxnet], got {}".format(framework)
        )


if __name__ == "__main__":
    X, Y = mnist_to_numpy()
    X, Y = X.astype(np.float32), Y.astype(np.int8)
