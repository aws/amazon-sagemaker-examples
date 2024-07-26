import os
import json
import argparse
import yaml
import logging
from pprint import pprint

from autogluon.tabular import TabularDataset, TabularPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        logger.warn("More than one file is found in %s directory", path)
    logger.info("Using %s", file)
    filename = f"{path}/{file}"
    return filename


def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
    return result


if __name__ == "__main__":
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ---------------------------- Args parsing --------------------------------
    logger.info("Starting AutoGluon modelling")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=get_env_if_present("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=get_env_if_present("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=str, default=get_env_if_present("SM_NUM_GPUS"))
    parser.add_argument("--training_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TEST"))
    parser.add_argument("--ag_config", type=str, default=get_env_if_present("SM_CHANNEL_CONFIG"))

    args, _ = parser.parse_known_args()
    logger.info("Args: %s", args)

    # See SageMaker-specific environment variables: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    os.makedirs(args.output_data_dir, mode=0o777, exist_ok=True)

    config_file = get_input_path(args.ag_config)
    with open(config_file) as f:
        config = yaml.safe_load(f)  # AutoGluon-specific config

    if args.n_gpus:
        config['num_gpus'] = int(args.n_gpus)

    logger.info("Running training job with the config:")
    pprint(config)

    # ----------------------------- Training -----------------------------------

    train_file = get_input_path(args.training_dir)
    train_data = TabularDataset(train_file)
    test_file = get_input_path(args.test_dir)
    test_data = TabularDataset(test_file)

    ag_predictor_args = config["ag_predictor_args"]
    ag_predictor_args["path"] = args.model_dir
    ag_fit_args = config["ag_fit_args"]

    predictor = TabularPredictor(**ag_predictor_args).fit(train_data, **ag_fit_args)
    logger.info("Best model: %s", predictor.get_model_best())
    
    # Leaderboard
    lb = predictor.leaderboard()
    lb.to_csv(f'{args.output_data_dir}/leaderboard.csv', index=False)
    logger.info("Saved leaderboard to output.")
    
    # Feature importance
    feature_importance = predictor.feature_importance(test_data)
    feature_importance.to_csv(f'{args.output_data_dir}/feature_importance.csv')
    logger.info("Saved feature importance to output.")
    
    # Evaluation
    evaluation = predictor.evaluate(test_data)
    with open(f'{args.output_data_dir}/evaluation.json', 'w') as f:
        json.dump(evaluation, f)
    logger.info("Saved evaluation to output.")
    
    predictor.save_space()

    # ---------------------------- Inference -----------------------------------

    test_data_nolabel = test_data.drop(labels=ag_predictor_args['label'], axis=1)
    y_pred = predictor.predict(test_data_nolabel)
    y_pred.to_csv(f'{args.output_data_dir}/predictions.csv', index=False)