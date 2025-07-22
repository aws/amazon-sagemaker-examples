import argparse
import json
import os
import random
import pandas as pd
import glob
import pickle as pkl

import xgboost


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--silent", type=int, default=0)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--eval_metric", type=str, default="auc")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    train_files_path, validation_files_path = args.train, args.validation

    train_features_path = os.path.join(args.train, "train_features.csv")
    train_labels_path = os.path.join(args.train, "train_labels.csv")

    val_features_path = os.path.join(args.validation, "val_features.csv")
    val_labels_path = os.path.join(args.validation, "val_labels.csv")

    print("Loading training dataframes...")
    df_train_features = pd.read_csv(train_features_path)
    df_train_labels = pd.read_csv(train_labels_path)

    print("Loading validation dataframes...")
    df_val_features = pd.read_csv(val_features_path)
    df_val_labels = pd.read_csv(val_labels_path)

    X = df_train_features.values
    y = df_train_labels.values

    val_X = df_val_features.values
    val_y = df_val_labels.values

    dtrain = xgboost.DMatrix(X, label=y)
    dval = xgboost.DMatrix(val_X, label=val_y)

    watchlist = [(dtrain, "train"), (dval, "validation")]

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "silent": args.silent,
        "objective": args.objective,
        "subsample": args.subsample,
        "eval_metric": args.eval_metric,
        "early_stopping_rounds": args.early_stopping_rounds,
    }

    bst = xgboost.train(
        params=params, dtrain=dtrain, evals=watchlist, num_boost_round=args.num_round
    )

    model_dir = os.environ.get("SM_MODEL_DIR")
    pkl.dump(bst, open(model_dir + "/model.bin", "wb"))


if __name__ == "__main__":
    main()
