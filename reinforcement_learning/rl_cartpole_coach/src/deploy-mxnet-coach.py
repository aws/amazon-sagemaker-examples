import os
import json
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
from mxnet import gluon, nd
import numpy as np
    

def model_fn(model_dir):
    """
    Load the onnx model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model
    """
    onnx_path = os.path.join(model_dir, "model.onnx")
    ctx = mx.cpu() # todo: pass into function
    # load onnx model symbol and parameters
    sym, arg_params, aux_params = onnx_mxnet.import_model(onnx_path)
    model_metadata = onnx_mxnet.get_model_metadata(onnx_path)
    # first index is name, second index is shape
    input_names = [inputs[0] for inputs in model_metadata.get('input_tensor_data')]
    input_symbols = [mx.sym.var(i) for i in input_names]
    net = gluon.nn.SymbolBlock(outputs=sym, inputs=input_symbols)
    net_params = net.collect_params()
    # set parameters (on correct context)
    for param in arg_params:
        if param in net_params:
            net_params[param]._load_init(arg_params[param], ctx=ctx)
    for param in aux_params:
        if param in net_params:
            net_params[param]._load_init(aux_params[param], ctx=ctx)
    # hybridize for increase performance
    net.hybridize()
    return net
    

def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: Gluon Model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    input_list = json.loads(data)
    input_nd = mx.nd.array(input_list).expand_dims(0)
    output = net(input_nd)
    if type(output) == list:
        output_list = [o.asnumpy().tolist() for o in output]
    elif type(output) == mx.nd.NDArray:
        output_list = [output.asnumpy().tolist()]
    return json.dumps(output_list), output_content_type
