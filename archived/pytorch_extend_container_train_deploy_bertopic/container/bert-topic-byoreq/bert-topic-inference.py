# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import argparse
import ast
import logging
import os
import torch
import json
from bertopic import BERTopic

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):    
    logger.info(f"inside model_fn, model_dir= {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))
    model = BERTopic.load(os.path.join(model_dir, 'my_model'))
    return model

def predict_fn(data, model):
    logger.info(f'Got input Data: {data}')
    return model.transform(data)

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info(f"serialized_input_data object: {serialized_input_data}")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        logger.info(f"input_data object: {input_data}")
        return input_data

    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return
    
def output_fn(prediction, content_type):
    logger.info(f"prediction object before: {prediction}, type: {type(prediction)}")
    prediction = list(prediction)
    logger.info(f"prediction object after: {prediction}, type: {type(prediction)}")
    prediction[0] = [int(res_class) for res_class in prediction[0]]
    logger.info(f"prediction[0] object after: {prediction[0]}, type: {type(prediction[0])}")
    prediction[1] = [float(res_class) for res_class in prediction[1]]
    logger.info(f"prediction[1] object after: {prediction[0]}, type: {type(prediction[1])}")    
    prediction_result = { "predictions": prediction[0],
                         "scores": prediction[1],
                        }
    prediction_result = json.dumps(prediction)
    logger.info(f"prediction_result object: {prediction_result}")
    return prediction_result
