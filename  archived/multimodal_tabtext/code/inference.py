import csv
import json
import numpy as np
import pandas as pd
import sys
from autogluon.tabular import TabularPredictor


csv.field_size_limit(int(sys.maxsize/10))  # Avoid csv.Error: field larger than field limit (131072)


def model_fn(model_dir):
    """Load the AutoGluon model. Called when the hosting service starts.

    :param model_dir: The directory where model files are stored.

    :return: AutoGluon model.
    """
    model = TabularPredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()
    return model


def _predict_fn(input_data, model):
    """Apply model to the incoming request.
    
    :param input_data: The request data in a Pandas Dataframe format.
    :param model: The loaded model for prediction.
    
    :return: Predicted results.
    """
    return model.predict(input_data).values[:]


def _input_fn(request_body, request_content_type):
    """Deserialize and prepare the prediction input.
    
    :param request_body: The request payload.
    :param request_content_type: The request content type.
    
    :return: Pandas Dataframe.
    """
    if request_content_type == "text/csv":
        reader = csv.reader(request_body.split('\n'), delimiter=',', quotechar='"')
        data = [line for line in reader]
        data = np.array(data, dtype=object)
    else:
        raise Exception(f"{request_content_type} content type not supported")
    return pd.DataFrame(data=data, columns=column_names)


def transform_fn(model, data, input_content_type, output_content_type):
    """
    Transform a request using the loaded model. Called once per inference request.
    
    :param model: The inference model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    
    :return: Response payload and content type.
    """
    input_data = _input_fn(data, input_content_type)
    result = _predict_fn(input_data, model)
    return json.dumps(result.tolist()), output_content_type
