import os

import chainer
import numpy as np

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300


def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.
    
    Here, we load the pre-trained model's weights. `voc_bbox_label_names` contains
    label names, and `SSD300` defines the network architecture. We pass in the
    number of labels and the path to the model for `SSD300` to load.

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model

    For more on `model_fn` and `save`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    # Loads a pretrained SSD model.
    chainer.config.train = False
    path = os.path.join(model_dir, 'ssd_model.npz')
    model = SSD300(n_fg_class=len(voc_bbox_label_names), pretrained_model=path)
    return model
    

def predict_fn(input_data, model):
    """
    This function receives a NumPy array and makes a prediction on it using the model returned
    by `model_fn`.
    
    The default predictor used by `Chainer` serializes input data to the 'npy' format:
    https://docs.scipy.org/doc/numpy-1.14.0/neps/npy-format.html

    The Chainer container provides an overridable pre-processing function `input_fn`
    that accepts the serialized input data and deserializes it into a NumPy array.
    `input_fn` is invoked before `predict_fn` and passes its return value to this function
    (as `input_data`)
    
    The Chainer container provides an overridable post-processing function `output_fn`
    that accepts this function's return value and serializes it back into `npy` format, which
    the Chainer predictor can deserialize back into a NumPy array on the client.

    Args:
        input_data (bytes): a NumPy array containing the data serialized by the Chainer predictor
        model: the return value of `model_fn`
    Returns:
        a NumPy array containing predictions which will be returned to the client

    For more on `input_fn`, `predict_fn` and `output_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        bboxes, labels, scores = model.predict([input_data])
        bbox, label, score = bboxes[0], labels[0], scores[0]
        return np.array([bbox.tolist(), label, score])
