import argparse
import os
import shutil
from pprint import pprint

import yaml
import pandas as pd

from autogluon.multimodal import MultiModalPredictor


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        print(f"WARN: more than one file is found in {path} directory")
    print(f"Using {file}")
    filename = f"{path}/{file}"
    return filename


if __name__ == "__main__":
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ------------------------------------------------------------ Args parsing
    print("Starting AG")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=str, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--test_dir", type=str, required=False, default=os.environ.get("SM_CHANNEL_TEST")
    )
    parser.add_argument("--images_dir", type=str, default=os.environ.get("SM_CHANNEL_IMAGES"))
    parser.add_argument("--ag_config", type=str, default=os.environ.get("SM_CHANNEL_CONFIG"))

    args, _ = parser.parse_known_args()

    print(f"Args: {args}")

    # See SageMaker-specific environment variables: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    os.makedirs(args.output_data_dir, mode=0o755, exist_ok=True)

    config_file = get_input_path(args.ag_config)
    with open(config_file) as f:
        config = yaml.safe_load(f)  # AutoGluon-specific config

    if args.n_gpus:
        config["num_gpus"] = int(args.n_gpus)

    print("Running training job with the config:")
    pprint(config)

    # ---------------------------------------------------------------- Training

    save_path = os.path.normpath(args.model_dir)

    train_file = get_input_path(args.training_dir)
    train_data = pd.read_csv(train_file, index_col=0)

    ag_predictor_args = config["ag_predictor_args"]
    ag_predictor_args["path"] = save_path
    ag_fit_args = config["ag_fit_args"]

    if args.images_dir:
        image_compressed_file = get_input_path(args.images_dir)
        shutil.unpack_archive(image_compressed_file)

    predictor = MultiModalPredictor(**ag_predictor_args).fit(train_data, **ag_fit_args)
    predictor.save(path=save_path + os.path.sep, standalone=True)

    # --------------------------------------------------------------- Inference

    if args.test_dir:
        test_file = get_input_path(args.test_dir)
        test_data = pd.read_csv(test_file)

        # Predictions
        y_pred_proba = predictor.predict_proba(test_data)
        y_pred_proba.to_csv(f"{args.output_data_dir}/predictions.csv")
