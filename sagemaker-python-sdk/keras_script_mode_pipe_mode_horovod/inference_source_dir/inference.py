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
    