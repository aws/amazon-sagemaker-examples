import bz2
import random
import tempfile
import urllib.request

import boto3


def load_mnist(train_split=0.8, seed=42):

    if not (0 < train_split <= 1):
        raise ValueError("'train_split' must be between 0 and 1.")

    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2"

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as mnist_bz2:
        urllib.request.urlretrieve(url, mnist_bz2.name)

    with bz2.open(mnist_bz2.name, "r") as fin:
        content = fin.read().decode("utf-8")
        lines = content.strip().split('\n')
        n = sum(1 for line in lines)
        indices = list(range(n))
        random.seed(seed)
        random.shuffle(indices)
        train_indices = set(indices[:int(n * 0.8)])

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as train_file:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as valid_file:
                for idx, line in enumerate(lines):
                    if idx in train_indices:
                        train_file.write(line + '\n')
                    else:
                        valid_file.write(line + '\n')

    return train_file.name, valid_file.name


def write_to_s3(fobj, bucket, key):
    return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fobj)


def upload_to_s3(filename, bucket, key):
    url = f"s3://{bucket}/{key}"
    print(f"Writing to {url}")
    with open(filename, "rb") as fobj:
        write_to_s3(fobj, bucket, key)
