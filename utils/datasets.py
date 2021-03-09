import boto3
import os


PUBLIC_BUCKET = "sagemaker-sample-files"

def download_mnist(data_dir='/tmp/data', train=True):
    """Download MNIST dataset from a public S3 bucket
    
    Args:
        data_dir (str): directory to save the data
        train (bool): download training set
    
    Returns:
        None
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
    s3 = boto3.client('s3')
    for obj in [images_file, labels_file]:
        key = os.path.join("datasets/image/MNIST", obj)
        dest = os.path.join(data_dir, obj)
        if not os.path.exists(dest):
            s3.download_file(PUBLIC_BUCKET, key, dest)
    return 

