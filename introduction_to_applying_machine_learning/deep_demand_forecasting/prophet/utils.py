from typing import List, Dict, Tuple
import logging
from logging.config import fileConfig

import pandas as pd

from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import Forecast


def get_logger(config_path) -> logging.Logger:
    fileConfig(config_path)
    logger = logging.getLogger()
    return logger


def evaluate(
    model: Predictor, test_data: ListDataset, num_samples: int
) -> Tuple[List[Forecast], List[pd.Series], Dict[str, float], pd.DataFrame]:
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data, predictor=model, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator()
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
    
    return forecasts, tss, agg_metrics, item_metrics
