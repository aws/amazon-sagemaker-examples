import argparse
import json
import logging
import os
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor


logging.basicConfig(level=logging.WARNING)  # Use logging.WARNING since logging.INFO is ignored by AutoGluon DLC


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation-dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--n_gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--problem_type", type=str, default="classification")
    parser.add_argument("--eval_metric", type=str, default="roc_auc")
    parser.add_argument("--presets", type=str, default="medium_quality")
    parser.add_argument("--text_nn_presets", type=str, default="medium_quality_faster_train")
    parser.add_argument("--auto_stack", type=str, default="False")
    parser.add_argument("--num_bag_folds", type=int, default=0)
    parser.add_argument("--num_bag_sets", type=int, default=1)
    parser.add_argument("--num_stack_levels", type=int, default=0)
    parser.add_argument("--refit_full", type=str, default="False")
    parser.add_argument("--set_best_to_refit_full", type=str, default="False")
    parser.add_argument("--save_space", type=str, default="True")
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--numerical-feature-names", type=str)
    parser.add_argument("--categorical-feature-names", type=str)
    parser.add_argument("--textual-feature-names", type=str)
    parser.add_argument("--label-name", type=str)
    parser.add_argument("--pretrained-transformer", type=str, default="google/electra-small-discriminator")  

    return parser.parse_known_args()


def find_filepath(path):
    jsonl_filepaths = [f for f in Path(path).glob('**/*.jsonl')]
    assert len(jsonl_filepaths) == 1, "Single JSON Lines file expected."
    jsonl_filepath = jsonl_filepaths[0]
    return jsonl_filepath

def load_jsonl(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data

def run_with_args(args):
    """Run training."""

    if args.n_gpus:
        logging.warning(f"Running training job with the number of gpu: {args.n_gpus}")

    logging.warning('loading training data')
    train_data = load_jsonl(
        find_filepath(args.train_dir)
    )
    logging.warning(f'length of training data: {len(train_data)}')
    train_data = pd.DataFrame(train_data)

    logging.warning('loading test data')
    validation_data = load_jsonl(
        find_filepath(args.validation_dir)
    )
    logging.warning(f'length of validation data: {len(validation_data)}')
    validation_data = pd.DataFrame(validation_data)
    
    # parse feature names
    logging.warning('parsing feature names')
    numerical_feature_names = args.numerical_feature_names.split(',')
    logging.warning(f'numerical features are: {numerical_feature_names}')
    categorical_feature_names = args.categorical_feature_names.split(',')
    logging.warning(f'categorical features are: {categorical_feature_names}')
    textual_feature_names = args.textual_feature_names.split(',')
    logging.warning(f'text features are: {textual_feature_names}')
    
    label_name = args.label_name
    
    train_data = train_data[numerical_feature_names + categorical_feature_names + textual_feature_names + [label_name]]
    validation_data = validation_data[numerical_feature_names + categorical_feature_names + textual_feature_names + [label_name]]    

    if args.problem_type == "classification":
        num_classes_y = len(np.unique(train_data[label_name].values))

        if num_classes_y >= 2:
            if num_classes_y == 2:
                problem_type = "binary"
            else:
                problem_type = "multiclass"
    else:
        assert args.problem_type == "regression", "problem_type has to be one of {classificaiton, regression}."
        problem_type = args.problem_type
    logging.warning(f"problem type is {problem_type}")
    
    ag_predictor_args = {
        "label": label_name,
        "path": args.model_dir,
        "eval_metric": args.eval_metric,
        "problem_type": problem_type,
    }

    kwargs = {
        "auto_stack": args.auto_stack == "True",
        "num_bag_folds": args.num_bag_folds,
        "num_bag_sets": args.num_bag_sets,
        "num_stack_levels": args.num_stack_levels,
        "refit_full": args.refit_full == "True",
        "set_best_to_refit_full": args.set_best_to_refit_full == "True",
        "save_space": args.save_space == "True",
        "verbosity": args.verbosity,
    }
    
    hyperparameters_multimodal = {
        'NN_TORCH': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
        "AG_TEXT_NN": {
            'presets': args.text_nn_presets,
            "model.hf_text.checkpoint_name": args.pretrained_transformer,
        },
        'AG_IMAGE_NN': {},
        'VW': {},        

    }
    
    if int(args.num_bag_folds) == 0:
        TabularPredictor(**ag_predictor_args).fit(
            train_data=train_data, tuning_data=validation_data, hyperparameters=hyperparameters_multimodal, presets=args.presets, **kwargs
        )
    else:
        logging.warning(
            f"bagged mode was specified with num_bag_folds as {int(args.num_bag_folds)} but "
            f"validation data is not None. Concatenating validation data with training data "
            f"and passing it as `train_data` into `TabularPredictor.fit()`."
        )

        train_data = pd.concat([train_data, validation_data], axis=0, ignore_index=True)
        TabularPredictor(**ag_predictor_args).fit(train_data=train_data, hyperparameters=hyperparameters_multimodal, presets=args.presets, **kwargs)


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
