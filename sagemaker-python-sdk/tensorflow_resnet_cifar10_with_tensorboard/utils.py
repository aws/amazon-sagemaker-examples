import os
import sys
import tarfile
import boto3
from six.moves import urllib
from ipywidgets import FloatProgress
from IPython.display import display


def cifar10_download(data_dir='/tmp/cifar10_data', print_progress=True):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(os.path.join(data_dir, 'cifar-10-batches-bin')):
        print('cifar dataset already downloaded')
        return

    filename = 'cifar-10-binary.tar.gz'
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        region = boto3.Session().region_name
        boto3.Session().resource('s3', region_name=region).Bucket('sagemaker-sample-data-{}'.format(region)).download_file('tensorflow/cifar10/cifar-10-binary.tar.gz', '/tmp/cifar10_data/cifar-10-binary.tar.gz')

    tarfile.open(filepath, 'r:gz').extractall(data_dir)
