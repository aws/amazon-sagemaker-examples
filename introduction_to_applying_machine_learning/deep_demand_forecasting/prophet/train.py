import os
import os.path as osp
from pathlib import Path
import json

import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset, load_datasets
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import MetaData

from data import load_train_and_validation_datasets
from metrics import rrse
from utils import get_logger, evaluate


LOG_CONFIG = os.getenv("LOG_CONFIG_PATH", Path(osp.abspath(__file__)).parent / "log.ini")

logger = get_logger(LOG_CONFIG)

def save(model: Predictor, model_dir: str) -> None:
    model.serialize(Path(model_dir))
    return
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa(
        "--dataset_path",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"],
        help="path to the dataset",
    )
    aa(
        "--output_dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"],
        help="output directory",
    )
    aa(
        "--model_dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="model directory",
    )
    aa("--prediction_length", type=int, help="future prediction length")
    aa("--multiple_prediction_length_for_training", type=int, help="length of training data")
    aa("--changepoint_prior_scale", type=float, help="determines the flexibility of the trendh")
    aa("--seasonality_prior_scale", type=float, help="controls the flexibility of the seasonality")
    aa("--holidays_prior_scale", type=float, help="controls flexibility to fit holiday effects")
    aa("--seasonality_mode", type=str, help="additive or multiplicative")
    aa("--changepoint_range", type=float, help="the proportion of the history in which the trend is allowed to change")

    args = parser.parse_args()
    logger.info(f"Passed arguments: {args}")
    
    meta = MetaData.parse_file(Path(args.dataset_path) / "metadata.json")
    dataset = load_train_and_validation_datasets(
        path=Path(args.dataset_path), 
        prediction_length=args.prediction_length,
        multiple_prediction_length_for_training=args.multiple_prediction_length_for_training
    )
    
    logger.info(f"The length of each train data is: {next(iter(dataset[0]))['target'].shape}")

    # define prophet predictor
    predictor = ProphetPredictor(
        meta.freq, 
        args.prediction_length,
        prophet_params={
            "changepoint_prior_scale": args.changepoint_prior_scale,
            "seasonality_prior_scale": args.seasonality_prior_scale,
            "holidays_prior_scale": args.holidays_prior_scale,
            "seasonality_mode": args.seasonality_mode,
            "changepoint_range": args.changepoint_range
        }
    )

    # store serialized model artifacts
    save(predictor, args.model_dir)
    logger.info(f"Model serialized in {args.model_dir}")
    
    # evaluate the model on the generated rolling-based data where we call evaluate on each set of data seperately
    agg_metrics_all = []
    item_metrics_all = pd.DataFrame()
    for data in dataset:
        forecasts, tss, agg_metrics, item_metrics = evaluate(predictor, data, num_samples=1)
        agg_metrics["RRSE"] = rrse(agg_metrics, data)
        agg_metrics_all.append(agg_metrics)
        item_metrics_all = item_metrics_all.append(item_metrics)
        
    # average each metric 
    agg_metrics_average = {}
    for each_key in agg_metrics_all[0].keys():
        tmp_sum = 0
        for each_agg in agg_metrics_all:
            tmp_sum += each_agg[each_key]
        agg_metrics_average[each_key] = tmp_sum / len(agg_metrics_all)
    
    logger.info(f"Root Relative Squared Error (RRSE): {agg_metrics_average['RRSE']}")

    with open(osp.join(args.output_dir, "train_agg_metrics.json"), "w", encoding="utf-8") as fout:
        json.dump(agg_metrics_average, fout)

    item_metrics_all.to_csv(
        osp.join(args.output_dir, "item_metrics.csv.gz"),
        index=False,
        encoding="utf-8",
        compression="gzip",
    )
