import argparse
import ast
import logging
import os
import json
from gluoncv import model_zoo, data
import mxnet as mx
from mxnet import nd, gluon
import numpy as np
from io import BytesIO
from timeit import default_timer as timer


def get_ctx():
    "function to get machine hardware context"
    try:
        _ = mx.nd.array([0], ctx=mx.gpu())
        ctx = mx.gpu()
    except:
        try:
            _ = mx.nd.array([0], ctx=mx.eia())
            ctx = mx.eia()
        except:
            ctx = mx.cpu()
    return ctx


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)

    assumes that the parameters artifact is {model_name}.params
    """
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    ctx = get_ctx()
    logging.info("Using ctx {}".format(ctx))
    logging.info("Dir content {}".format(os.listdir()))

    # instantiate net and reset to classes of interest
    net = gluon.nn.SymbolBlock.imports(
        symbol_file=[f for f in os.listdir() if f.endswith("json")][0],
        input_names=["data"],
        param_file=[f for f in os.listdir() if f.endswith("params")][0],
        ctx=ctx,
    )

    return net


def transform_fn(model, input_data, content_type, accept):

    start = timer()
    # Expecting input_data as numpy serialized object
    images_array = np.load(BytesIO(input_data), allow_pickle=True)
    if images_array.ndim == 3:
        images_array = np.expand_dims(images_array, axis=0)
    images_nd_array_list = [mx.nd.array(image_array) for image_array in images_array]
    logging.info(f"Decoded images in {(timer()-start):.4f} seconds")

    start = timer()
    tensors, origs = data.transforms.presets.yolo.transform_test(images_nd_array_list)
    if type(tensors) != list:
        tensors = [tensors]
        origs = [origs]

    logging.info(f"Transformed images in {(timer()-start):.4f} seconds")

    ctx = get_ctx()
    logging.info("Using ctx {}".format(ctx))

    start = timer()
    results = []
    # forward pass and display
    for i in range(len(tensors)):
        h = origs[i].shape[0]
        w = origs[i].shape[1]

        box_ids, scores, bboxes = model(tensors[i].as_in_context(ctx))

        result = nd.concat(box_ids, scores, bboxes, dim=2).asnumpy()
        # Filter out -1 value rows
        result = np.array([res[res[:, 0] != -1] for res in result])
        result[:, :, 2] /= w
        result[:, :, 3] /= h
        result[:, :, 4] /= w
        result[:, :, 5] /= h

        result[:, :, 2:] = np.where(result[:, :, 2:] > 1, 1, result[:, :, 2:])
        result[:, :, 2:] = np.where(result[:, :, 2:] < 0, 0, result[:, :, 2:])

        results.append(result[0].tolist())
    logging.info(f"Ran model inference on images in {(timer()-start):.4f} seconds")
    logging.info(f"Returning predictions: {results}")
    return results  # return a single tensor
