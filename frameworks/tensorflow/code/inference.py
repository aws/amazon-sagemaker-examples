# from __future__ import print_function

import logging

#import numpy as np
import json
import os

import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)

def _model_fn(model_dir):
    """Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    # TF serving container requires model to have a version number
    return tf.keras.models.load_model(model_dir)

def _transform_fn(net, data, input_content_type, output_content_type):
    response_body = json.dumps([1, 2, 3, 4, 5])
    output_content_type = 'application/json'

    return response_body, output_content_type

def _transform_fn(net, data, input_content_type, output_content_type):
    assert input_content_type=='application/json'
    assert output_content_type=='application/json' 

    # parsed should be a 1d array of length 728
    parsed = json.loads(data)
    parsed = parsed['inputs'] 
    
    # convert to numpy array
    arr = np.array(parsed).astype(np.float32)
    
    # convert to an eager tensor
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    
    output = net.predict(tensor)
    
    prediction = np.argmax(output, axis=1)
    response_body = json.dumps(prediction.tolist())

    return response_body, output_content_type




