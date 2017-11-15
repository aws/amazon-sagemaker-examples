import numpy as np
import os
import zipfile
from skimage import io
from mxnet.test_utils import download


def download_training_data():
    print('downloading training data...')
    if not os.path.isdir("data"):
        os.makedirs('data')
    if (not os.path.exists('data/train.rec')) or \
            (not os.path.exists('data/test.rec')) or \
            (not os.path.exists('data/train.lst')) or \
            (not os.path.exists('data/test.lst')):
        zip_file_path = download('http://data.mxnet.io/mxnet/data/cifar10.zip')
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall()
        os.rename('cifar', 'data')
    print('done')


def read_image(filename):
    img = io.imread(filename)
    img = np.array(img).transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img


def read_images(filenames):
    return [read_image(f) for f in filenames]
