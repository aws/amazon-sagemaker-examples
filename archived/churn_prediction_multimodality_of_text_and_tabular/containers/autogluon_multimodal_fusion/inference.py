import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from sagemaker_inference import encoder


def model_fn(model_dir: str) -> MultiModalPredictor:
    """Read model saved in model_dir and return an object of autogluon tabular model.

    Args:
        model_dir (str): directory that saves the model artifact.

    Returns:
        obj (TabularPredictor): autogluon model.
    """
    try:
        model = MultiModalPredictor.load(model_dir)
        
        col_names = []
        if "numerical" in model._data_processors:
            col_names += model._data_processors["numerical"][0].numerical_column_names
        if "categorical" in model._data_processors:
            col_names += model._data_processors["categorical"][0].categorical_column_names
        if "text" in model._data_processors:
            col_names += model._data_processors["text"][0].text_column_names            
        
        globals()["column_names"] = col_names
        logging.warning(f"columns names: {column_names}")
        return model
    except Exception:
        logging.exception("Failed to load model from checkpoint")
        raise


def transform_fn(task: MultiModalPredictor, input_data: Any, content_type: str, accept: str) -> np.array:
    """Make predictions against the model and return a serialized response.

    The function signature conforms to the SM contract.
    Args:
        task (TabularPredictor): model loaded by model_fn.
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.

    Returns:
        obj: the serialized prediction result or a tuple of the form (response_data, content_type)
    """
    if content_type == "text/csv":
        data = pd.read_csv(io.StringIO(input_data), sep=",", header=None)
        data.columns = column_names
        
        try:
            model_output = task.predict_proba(data).values
            output = {"probabilities": model_output}
            return encoder.encode(output, accept)
        except Exception:
            logging.exception("Failed to do transform")
            raise
    raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
