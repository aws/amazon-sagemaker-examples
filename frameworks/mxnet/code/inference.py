# coding=utf-8
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import logging

from mxnet import gluon
import mxnet as mx

import numpy as np
import json
import os

logging.basicConfig(level=logging.DEBUG)

def model_fn(model_dir):
    """Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    net = gluon.SymbolBlock.imports(
            symbol_file=os.path.join(model_dir, 'model-symbol.json'),
            input_names=['data'],
            param_file=os.path.join(model_dir, 'model-0000.params'))
    return net

def transform_fn(net, data, input_content_type, output_content_type):
    assert input_content_type=='application/json'
    assert output_content_type=='application/json' 

    # parsed should be a 1d array of length 728
    parsed = json.loads(data)
    parsed = parsed['inputs'] 
    
    # convert to numpy array
    arr = np.array(parsed).reshape(-1, 1, 28, 28)
    
    # convert to mxnet ndarray
    nda = mx.nd.array(arr)

    output = net(nda)
    
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist())

    return response_body, output_content_type


if __name__ == '__main__':
    model_dir = '/home/ubuntu/models/mxnet-gluon-mnist'
    net = model_fn(model_dir)

    import json
    import random
    data = {'inputs': [random.random() for _ in range(784)]}
    data = json.dumps(data)
    
    content_type = 'application/json'
    a, b = transform_fn(net, data, content_type, content_type)
    print(a, b)


