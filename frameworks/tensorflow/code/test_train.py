import json
import os
import sys

import boto3
from train import parse_args, train

dirname = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(dirname, "config.json"), "r") as f:
    CONFIG = json.load(f)


def download_from_s3(data_dir="/tmp/data", train=True):
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
    return


class Env:
    def __init__(self):
        # simulate container env
        os.environ["SM_MODEL_DIR"] = "/tmp/tf/model"
        os.environ["SM_CHANNEL_TRAINING"] = "/tmp/data"
        os.environ["SM_CHANNEL_TESTING"] = "/tmp/data"
        os.environ["SM_HOSTS"] = '["algo-1"]'
        os.environ["SM_CURRENT_HOST"] = "algo-1"
        os.environ["SM_NUM_GPUS"] = "0"


if __name__ == "__main__":
    Env()
    download_from_s3()
    download_from_s3(train=False)
    args = parse_args()
    train(args)
