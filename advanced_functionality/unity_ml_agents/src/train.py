import argparse
import json
import os
import subprocess

import mlagents
import numpy as np
import tensorflow as tf
import yaml
from mlagents_envs.environment import UnityEnvironment


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--env_name", type=str, default=os.environ.get("SM_HP_ENV_NAME"))
    parser.add_argument("--yaml_file", type=str, default=os.environ.get("SM_HP_YAML_FILE"))
    # parser.add_argument('--train_config', type=str, default=os.environ.get('SM_HP_TRAIN_CONFIG'))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, _ = parse_args()
    subprocess.call(f"chmod 755 {args.train}/{args.env_name}".split())
    subprocess.call(
        f"mlagents-learn --env={args.train}/{args.env_name} --train /opt/ml/code/{args.yaml_file}".split()
    )
    subprocess.call(f"cp -arf ./models {args.model_dir}".split())
    subprocess.call(f"cp -arf ./summaries {args.model_dir}".split())
