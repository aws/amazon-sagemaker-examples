#     Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License").
#     You may not use this file except in compliance with the License.
#     A copy of the License is located at
#
#         https://aws.amazon.com/apache-2-0/
#
#     or in the "license" file accompanying this file. This file is distributed
#     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#     express or implied. See the License for the specific language governing
#     permissions and limitations under the License.

from keras.models import load_model
import logging
import numpy as np
from sagemaker_sklearn_container import serving

logging.getLogger().setLevel(logging.INFO)

def model_fn(model_dir):
    logging.info(model_dir)
    model = load_model(model_dir + '/model.h5')
    model._make_predict_function()
    return model

def predict_fn(input_data, model):
    logging.info(input_data)
    return serving.default_predict_fn(input_data, model)
    