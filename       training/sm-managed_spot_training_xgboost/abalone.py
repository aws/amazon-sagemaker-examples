"""XGBoost script-mode entry point for the Managed Spot Training example.

This script is executed inside the SageMaker XGBoost (framework/script mode)
container. It reads libsvm-formatted training and validation data from the
SageMaker channel directories, trains an XGBoost model, and writes the model
artifact to the model directory.

Managed Spot Training checkpointing is enabled through the
``sagemaker_xgboost_container.checkpointing`` helpers so the job can resume from
the last saved checkpoint after a Spot interruption. Checkpoints are written to
``/opt/ml/checkpoints`` (the container-local path that SageMaker syncs with the
``checkpoint_s3_uri`` you configure on the trainer).
"""
import argparse
import logging
import os
import pickle as pkl

import xgboost as xgb
from sagemaker_xgboost_container import checkpointing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default local checkpoint directory used by managed spot training.
CHECKPOINTS_DIR = "/opt/ml/checkpoints"


def _parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters are passed to the script as command-line arguments.
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--min_child_weight", type=float, default=6)
    parser.add_argument("--subsample", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    parser.add_argument("--num_round", type=int, default=50)
    parser.add_argument("--verbosity", type=int, default=2)

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )

    # Use parse_known_args so the script tolerates extra hyperparameters that
    # SageMaker injects during Automatic Model Tuning (e.g.
    # ``_tuning_objective_metric``) as well as any additional tunable
    # hyperparameters not explicitly declared above.
    args, _ = parser.parse_known_args()
    return args


def _find_data_file(channel_dir):
    """Return the path to the single data file inside a channel directory.

    ``channel_dir`` is a SageMaker-managed channel path, but we still resolve
    each entry and confirm it stays within the channel directory so that no
    symlink or unexpected entry can escape it via ``..`` sequences.
    """
    base = os.path.realpath(channel_dir)
    files = []
    for name in os.listdir(base):
        candidate = os.path.realpath(os.path.join(base, name))
        if os.path.commonpath([base, candidate]) != base:
            # Entry resolves outside the channel directory; skip it.
            continue
        if os.path.isfile(candidate):
            files.append(candidate)
    if not files:
        raise ValueError(f"No data files found in channel directory: {channel_dir}")
    return files[0]


def main():
    args = _parse_args()

    # Load the libsvm data into XGBoost DMatrix objects.
    dtrain = xgb.DMatrix(f"{_find_data_file(args.train)}?format=libsvm")

    watchlist = [(dtrain, "train")]
    if args.validation:
        dval = xgb.DMatrix(f"{_find_data_file(args.validation)}?format=libsvm")
        watchlist.append((dval, "validation"))

    train_hp = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "alpha": args.alpha,
        "objective": args.objective,
        "verbosity": args.verbosity,
    }

    # Enable managed-spot-training checkpointing. checkpointing.train() loads any
    # existing checkpoint, resumes from the last completed iteration, and saves a
    # checkpoint per round so the job can survive Spot interruptions.
    train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
    )
    bst = checkpointing.train(train_args, checkpoint_dir=CHECKPOINTS_DIR)

    # Persist the trained model to the model directory so SageMaker uploads it.
    os.makedirs(args.model_dir, exist_ok=True)
    model_location = os.path.join(args.model_dir, "xgboost-model")
    with open(model_location, "wb") as model_file:
        pkl.dump(bst, model_file)
    logger.info("Stored trained model at %s", model_location)


if __name__ == "__main__":
    main()
