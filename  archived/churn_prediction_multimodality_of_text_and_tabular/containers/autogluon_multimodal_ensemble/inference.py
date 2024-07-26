import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sagemaker_inference import encoder


def model_fn(model_dir: str) -> TabularPredictor:
    """Read model saved in model_dir and return an object of autogluon tabular model.

    Args:
        model_dir (str): directory that saves the model artifact.

    Returns:
        obj (TabularPredictor): autogluon model.
    """
    try:
        model = TabularPredictor.load(model_dir)
        globals()["column_names"] = model.feature_metadata_in.get_features()
        return model
    except Exception:
        logging.exception("Failed to load model from checkpoint")
        raise


def transform_fn(task: TabularPredictor, input_data: Any, content_type: str, accept: str) -> np.array:
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
