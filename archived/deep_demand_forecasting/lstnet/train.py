import os
import os.path as osp
from pathlib import Path
import json

import numpy as np
import mxnet as mx
import pandas as pd

from gluonts.model.lstnet import LSTNetEstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import TrainDatasets
from gluonts.model.predictor import Predictor
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset

from data import load_multivariate_datasets
from metrics import rrse
from utils import get_logger, evaluate, str2bool


LOG_CONFIG = os.getenv("LOG_CONFIG_PATH", Path(osp.abspath(__file__)).parent / "log.ini")

logger = get_logger(LOG_CONFIG)


def train(
    dataset: TrainDatasets,
    output_dir: str,
    model_dir: str,
    context_length: int,
    prediction_length: int,
    skip_size: int,
    ar_window: int,
    channels: int,
    scaling: bool,
    output_activation: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float,
    rnn_cell_type: str,
    rnn_num_layers: int,
    rnn_num_cells: int, 
    skip_rnn_cell_type: str,
    skip_rnn_num_layers: int,
    skip_rnn_num_cells: int,
    lead_time: int,
    kernel_size: int,    
    seed: int,
) -> Predictor:
    np.random.seed(seed)
    mx.random.seed(seed)
    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    logger.info(f"Using the context: {ctx}")
    trainer_hyperparameters = {
        "ctx": ctx,
        "epochs": epochs,
        "hybridize": True,
        "patience": 10,
        "learning_rate_decay_factor": 0.5,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
    }
    model_hyperparameters = {
        "freq": dataset.metadata.freq,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "skip_size": skip_size,
        "ar_window": ar_window,
        "num_series": dataset.metadata.feat_static_cat[0].cardinality,
        "channels": channels,
        "output_activation": output_activation,
        "scaling": scaling,
        "dropout_rate": dropout_rate,
        "rnn_cell_type": rnn_cell_type,
        "rnn_num_layers": rnn_num_layers,
        "rnn_num_cells": rnn_num_cells, 
        "skip_rnn_cell_type": skip_rnn_cell_type,
        "skip_rnn_num_layers": skip_rnn_num_layers,
        "skip_rnn_num_cells": skip_rnn_num_cells,
        "lead_time": lead_time,
        "kernel_size": kernel_size,  
        "trainer": Trainer(**trainer_hyperparameters),
    }
    estimator = LSTNetEstimator(**model_hyperparameters)
    predictor = estimator.train(dataset.train)
    return predictor


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
    aa("--context_length", type=int, help="past context length")
    aa("--prediction_length", type=int, help="future prediction length")
    aa("--skip_size", type=int, help="LSTNet skip size")
    aa("--ar_window", type=int, help="LSTNet AR window linear part")
    aa("--channels", type=int, help="number of channels for first conv1d layer")
    aa("--scaling", type=str, help="whether to mean scale normalize the data")
    aa(
        "--output_activation",
        type=str,
        help="the activation function for the output, either `None`, `sigmoid` or `tanh`",
    )
    aa("--epochs", type=int, default=1, help="number of epochs to train")
    aa("--batch_size", type=int, default=32, help="batch size")
    aa("--learning_rate", type=float, default=1e-2, help="learning rate")
    aa("--dropout_rate", type=float, default=0.2, help="dropout rate")
    aa("--rnn_cell_type", type=str, help="Type of the RNN cell. Either lstm or gru")
    aa("--rnn_num_layers", type=int, default=3, help="Number of RNN layers to be used")    
    aa("--rnn_num_cells", type=int, default=100, help="Number of RNN cells for each layer")        
    aa("--skip_rnn_cell_type", type=str, help="Type of the RNN cell for the skip layer. Either lstm or gru")            
    aa("--skip_rnn_num_layers", type=int, default=1, help="Number of RNN layers to be used for skip part")                
    aa("--skip_rnn_num_cells", type=int, default=10, help="Number of RNN cells for each layer for skip part")                    
    aa("--lead_time", type=int, default=0, help="Lead time")                    
    aa("--kernel_size", type=int, default=6, help="kernel_size")                        
    aa("--seed", type=int, default=12, help="RNG seed")
    args = parser.parse_args()
    logger.info(f"Passed arguments: {args}")

    dataset = load_multivariate_datasets(path=Path(args.dataset_path), is_validation=True, prediction_length=args.prediction_length, load_raw=True)

    logger.info(f"Train data shape: {next(iter(dataset.train))['target'].shape}")
    len_val = len(dataset.test)
    val_list = list(iter(dataset.test))
    logger.info(f"{len_val} rolling-based val data are created with multiple starting dates. The shape of these val data are:")
    for idx in range(len_val):
        logger.info(f"val data index {idx} shape: {val_list[idx]['target'].shape}")

    predictor = train(
        dataset,
        args.output_dir,
        args.model_dir,
        args.context_length,
        args.prediction_length,
        args.skip_size,
        args.ar_window,
        args.channels,
        str2bool(args.scaling),
        args.output_activation,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.dropout_rate,
        args.rnn_cell_type,
        args.rnn_num_layers,
        args.rnn_num_cells, 
        args.skip_rnn_cell_type,
        args.skip_rnn_num_layers,
        args.skip_rnn_num_cells,
        args.lead_time,
        args.kernel_size,          
        args.seed,
    )
    # store serialized model artifacts
    save(predictor, args.model_dir)
    logger.info(f"Model serialized in {args.model_dir}")
    
    # evaluate the model on the validation data
    forecasts, tss, agg_metrics, item_metrics = evaluate(predictor, dataset.test, num_samples=1)

    logger.info(f"Root Relative Squared Error (RRSE): {rrse(agg_metrics, dataset.test)}")

    with open(osp.join(args.output_dir, "train_agg_metrics.json"), "w", encoding="utf-8") as fout:
        json.dump(agg_metrics, fout)

    item_metrics.to_csv(
        osp.join(args.output_dir, "item_metrics.csv.gz"),
        index=False,
        encoding="utf-8",
        compression="gzip",
    )
