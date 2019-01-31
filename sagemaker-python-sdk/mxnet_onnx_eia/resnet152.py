from __future__ import print_function

import bisect
import json
import logging
import time
import random
import re
from collections import Counter, namedtuple
from itertools import chain, islice

import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import numpy as np
from mxnet import gluon, autograd, nd
from mxnet.io import DataIter, DataBatch, DataDesc
from mxnet.gluon import nn

logging.basicConfig(level=logging.DEBUG)


def model_fn(model_dir):
    """
    Load the onnx model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model
    """
    sym, arg_params, aux_params = onnx_mxnet.import_model('%s/resnet152v1.onnx' % model_dir) 
    # create module
    mod = mx.mod.Module(symbol=sym, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', [1, 3, 224, 224])], label_shapes=mod._label_shapes)
    mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)
    return mod


def transform_fn(mod, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param mod: The resnet152 model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    input_data = json.loads(data)
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch([mx.nd.array(input_data)]))
    scores = mx.ndarray.softmax(mod.get_outputs()[0]).asnumpy()
    scores = np.squeeze(scores)
    return json.dumps(scores.tolist()), output_content_type
