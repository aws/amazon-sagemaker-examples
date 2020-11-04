import numpy as np
from urllib import request
import gzip 
import os

def mnist_to_numpy(data_dir='data', train=True):
    """Download MNIST dataset and convert it to numpy array
    
    Args:
        data_dir (str): directory to save the data
        train (bool): download training set
    
    Returns:
        tuple of images and labels as numpy arrays
    """
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    base_url = "http://yann.lecun.com/exdb/mnist/"
    
    if train:
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"
    
    # download images and labels
    if not os.path.exists(os.path.join(data_dir, images_file)):
        request.urlretrieve(
            os.path.join(base_url, images_file),
            os.path.join(data_dir, images_file))
     
    if not os.path.exists(os.path.join(data_dir, labels_file)):
        request.urlretrieve(
            os.path.join(base_url, labels_file),
            os.path.join(data_dir, labels_file))
    
    return _convert_to_numpy(data_dir, images_file, labels_file)


def _convert_to_numpy(data_dir, images_file, labels_file):
    """Byte string to numpy arrays"""
    with gzip.open(os.path.join(data_dir, images_file), 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    
    with gzip.open(os.path.join(data_dir, labels_file), 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return (images, labels)

def normalize(x, axis):
    eps = np.finfo(float).eps

    mean = np.mean(x, axis=axis, keepdims=True)
    # avoid division by zero
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std

def adjust_to_framework(x, framework='pytorch'):
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
    
    if framework in ['pytorch', 'mxnet']:
        # depth-major
        return x
    elif framework == 'tensorlfow':
        # depth-minor
        return np.transpose(x, (0, 2, 3, 1))
    elif framework == 'mxnet':
        return x
    else:
        raise ValueError('framework must be one of ' + \
                        '[pytorch, tensorflow, mxnet], got {}'.format(framework))

if __name__ == '__main__':
    X, Y = mnist_to_numpy()
    X, Y = X.astype(np.float32), Y.astype(np.int8)

    
    
    
    
