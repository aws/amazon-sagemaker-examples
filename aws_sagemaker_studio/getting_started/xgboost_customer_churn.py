import argparse
import json
import os
import pickle
import random
import tempfile
import urllib.request

import xgboost
from smdebug import SaveConfig
from smdebug.xgboost import Hook


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--silent", type=int, default=0)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int, default=50)
    parser.add_argument("--smdebug_path", type=str, default=None)
    parser.add_argument("--smdebug_frequency", type=int, default=1)
    parser.add_argument("--smdebug_collections", type=str, default='metrics')
    parser.add_argument("--output_uri", type=str, default="/opt/ml/output/tensors",
                        help="S3 URI of the bucket where tensor data will be stored.")

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()

    return args


def create_smdebug_hook(out_dir, train_data=None, validation_data=None, frequency=1, collections=None,):

    save_config = SaveConfig(save_interval=frequency)
    hook = Hook(
        out_dir=out_dir,
        train_data=train_data,
        validation_data=validation_data,
        save_config=save_config,
        include_collections=collections,
    )

    return hook


def main():
    
    args = parse_args()

    train, validation = args.train, args.validation
    parse_csv = "?format=csv&label_column=0"
    dtrain = xgboost.DMatrix(train+parse_csv)
    dval = xgboost.DMatrix(validation+parse_csv)

    watchlist = [(dtrain, "train"), (dval, "validation")]

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "silent": args.silent,
        "objective": args.objective}

    # The output_uri is a the URI for the s3 bucket where the metrics will be
    # saved.
    output_uri = (
        args.smdebug_path
        if args.smdebug_path is not None
        else args.output_uri
    )

    collections = (
        args.smdebug_collections.split(',')
        if args.smdebug_collections is not None
        else None
    )

    hook = create_smdebug_hook(
        out_dir=output_uri,
        frequency=args.smdebug_frequency,
        collections=collections,
        train_data=dtrain,
        validation_data=dval,
    )

    bst = xgboost.train(
        params=params,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        callbacks=[hook])
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model_location = os.path.join(args.model_dir, 'xgboost-model')
    pickle.dump(bst, open(model_location, 'wb'))


if __name__ == "__main__":

    main()


def model_fn(model_dir):
    """Load a model. For XGBoost Framework, a default function to load a model is not provided.
    Users should provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns:
        A XGBoost model.
        XGBoost model format type.
    """
    model_files = (file for file in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, file)))
    model_file = next(model_files)
    try:
        booster = pickle.load(open(os.path.join(model_dir, model_file), 'rb'))
        format = 'pkl_format'
    except Exception as exp_pkl:
        try:
            booster = xgboost.Booster()
            booster.load_model(os.path.join(model_dir, model_file))
            format = 'xgb_format'
        except Exception as exp_xgb:
            raise ModelLoadInferenceError("Unable to load model: {} {}".format(str(exp_pkl), str(exp_xgb)))
    booster.set_param('nthread', 1)
    return booster, format
