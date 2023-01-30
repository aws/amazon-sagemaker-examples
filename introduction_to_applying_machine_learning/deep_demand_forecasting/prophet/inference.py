# mypy: ignore-errors
from typing import Any, List, Dict, Union
import os
import os.path as osp
from pathlib import Path
import json

import numpy as np

from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions

from utils import get_logger, evaluate


LOG_CONFIG = os.getenv("LOG_CONFIG_PATH", Path(osp.abspath(__file__)).parent / "log.ini")

logger = get_logger(LOG_CONFIG)


def model_fn(model_dir: str) -> Predictor:
    predictor = Predictor.deserialize(Path(model_dir))
    logger.info("model was loaded successfully")
    return predictor


def transform_fn(model: Predictor, request_body: Any, content_type: Any, accept_type: Any):
    request_data = json.loads(request_body)
    # TODO: customize serde
    request_list_data = ListDataset(
        [
            {
                FieldName.TARGET: x, 
                FieldName.START: request_data[FieldName.START]
            } 
            for x in request_data[FieldName.TARGET]
        ],
        freq=request_data["freq"],
    )
      
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=request_list_data, predictor=model, num_samples=1
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    response_body = {}
    response_body["forecasts"] = {}
    response_body["forecasts"]["samples"] = np.array([x.samples for x in forecasts]).tolist()
    response_body["forecasts"]["start_date"] = str(forecasts[0].start_date)
    response_body["forecasts"]["freq"] = forecasts[0].freq

    return response_body, content_type
