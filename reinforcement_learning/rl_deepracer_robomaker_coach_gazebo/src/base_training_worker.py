import argparse
import os
import subprocess
import time

from rl_coach.logger import screen

PRETRAINED_MODEL_DIR = "./pretrained_checkpoint"


def start_redis_server():
    process = subprocess.Popen(
        "redis-server /etc/redis/redis.conf", shell=True, stderr=subprocess.STDOUT
    )
    time.sleep(5)
    if process.poll() is not None:
        raise RuntimeError("Could not start Redis server.")
    else:
        print("Redis server started successfully!")
    return process


def get_args():
    screen.set_use_colors(False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pk",
        "--preset_s3_key",
        help="(string) Name of a preset to download from S3",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-ek",
        "--environment_s3_key",
        help="(string) Name of an environment file to download from S3",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model_metadata_s3_key",
        help="(string) Model Metadata File S3 Key",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        help="(string) Path to a folder containing a checkpoint to write the model to.",
        type=str,
        default="./checkpoint",
    )
    parser.add_argument(
        "--pretrained_checkpoint_dir",
        help="(string) Path to a folder for downloading a pre-trained model",
        type=str,
        default=PRETRAINED_MODEL_DIR,
    )
    parser.add_argument(
        "--s3_bucket",
        help="(string) S3 bucket",
        type=str,
        default=os.environ.get("SAGEMAKER_SHARED_S3_BUCKET_PATH", "gsaur-test"),
    )
    parser.add_argument("--s3_prefix", help="(string) S3 prefix", type=str, default="sagemaker")
    parser.add_argument(
        "--framework", help="(string) tensorflow or mxnet", type=str, default="tensorflow"
    )
    parser.add_argument(
        "--pretrained_s3_bucket", help="(string) S3 bucket for pre-trained model", type=str
    )
    parser.add_argument(
        "--pretrained_s3_prefix",
        help="(string) S3 prefix for pre-trained model",
        type=str,
        default="sagemaker",
    )
    parser.add_argument(
        "--aws_region",
        help="(string) AWS region",
        type=str,
        default=os.environ.get("AWS_REGION", "us-east-1"),
    )

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    start_redis_server()

    # Pass the same argument to the training_worker.py
    passing_arg_list = list()
    for key, value in vars(args).items():
        if value:
            passing_arg_list.append("--{} {}".format(key, value))
    os.system("{} {}".format("./markov/training_worker.py", " ".join(passing_arg_list)))


if __name__ == "__main__":
    main()
