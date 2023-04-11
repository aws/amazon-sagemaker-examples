import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train-args', type=str, help='Train arguments')
parser.add_argument('--sd-models-s3uri', default='', type=str, help='SD Models S3Uri')
parser.add_argument('--db-models-s3uri', default='', type=str, help='DB Models S3Uri')
parser.add_argument('--lora-models-s3uri', default='', type=str, help='Lora Models S3Uri')
parser.add_argument('--dreambooth-config-id', default='', type=str, help='Dreambooth config ID')

args = parser.parse_args()

cmd = "LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH ACCELERATE=true bash webui.sh --port 8080 --listen --xformers --train --lora-models-path /opt/ml/input/data/lora --dreambooth-models-path /opt/ml/input/data/dreambooth  --ckpt-dir /opt/ml/input/data/models --train-args '{0}' --sd-models-s3uri {1} --db-models-s3uri {2} --lora-models-s3uri {3} --dreambooth-config-id {4}".format(args.train_args, args.sd_models_s3uri, args.db_models_s3uri, args.lora_models_s3uri, args.dreambooth_config_id)

os.system('mkdir -p /opt/ml/input/data/lora')
os.system('mkdir -p /opt/ml/input/data/dreambooth')
os.system('mkdir -p /opt/ml/input/data/models')

os.system(cmd)
