from __future__ import absolute_import, print_function

import argparse
import csv
import decimal
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from io import StringIO
from string import Template

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from joblib import dump, load
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from utils import (
    ExitSignalHandler,
    load_json_object,
    print_files_in_path,
    print_json_object,
    save_model_artifacts,
    write_failure_file,
)

hyperparameters_file_path = "/opt/ml/input/config/hyperparameters.json"
inputdataconfig_file_path = "/opt/ml/input/config/inputdataconfig.json"
resource_file_path = "/opt/ml/input/config/resourceconfig.json"
data_files_path = "/opt/ml/input/data/"
failure_file_path = "/opt/ml/output/failure"
model_artifacts_path = "/opt/ml/model/"

feature_column = "words"
label_column = "label"

cpu_count = multiprocessing.cpu_count()

model_server_timeout = os.environ.get("MODEL_SERVER_TIMEOUT", 60)
model_server_workers = int(os.environ.get("MODEL_SERVER_WORKERS", cpu_count))


def sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)


def serve():
    print("Starting the inference server with {} workers.".format(model_server_workers))

    port = os.environ.get("SAGEMAKER_BIND_TO_PORT", 8080)
    print("using port: ", port)
    with open("/nginx.conf.template") as nginx_template:
        template = Template(nginx_template.read())
        print("nginx.conf:", template.substitute(port=port))
        nginx_conf = open("/nginx.conf", "w")
        nginx_conf.write(template.substitute(port=port))
        nginx_conf.close()

    # link the log streams to stdout/err so they will be logged to the container logs
    subprocess.check_call(["ln", "-sf", "/dev/stdout", "/var/log/nginx/access.log"])
    subprocess.check_call(["ln", "-sf", "/dev/stderr", "/var/log/nginx/error.log"])

    nginx = subprocess.Popen(["nginx", "-c", "/nginx.conf"])
    gunicorn = subprocess.Popen(
        [
            "gunicorn",
            "--timeout",
            str(model_server_timeout),
            "-k",
            "gevent",
            "-b",
            "unix:/tmp/gunicorn.sock",
            "-w",
            str(model_server_workers),
            "wsgi:app",
        ]
    )

    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print("Inference server exiting")


def load_data(raw, columns, skip_first_row=True):
    recs = [(row[0], set(row[1:])) for row in csv.reader(raw)]
    if skip_first_row:
        return pd.DataFrame.from_records(recs[1:], columns=columns)
    else:
        return pd.DataFrame.from_records(recs, columns=columns)


def load_raw(files, columns, skip_first_row=True):
    raw_data = []
    for file in files:
        raw_data.append(load_data(open(file), columns, skip_first_row))

    return pd.concat(raw_data)


def train():
    try:
        print("starting training...")
        hyperparameters = load_json_object(hyperparameters_file_path)
        print("\nHyperparameters configuration:")
        print_json_object(hyperparameters)

        input_data_config = load_json_object(inputdataconfig_file_path)
        print("\nInput data configuration:")
        print_json_object(input_data_config)

        for key in input_data_config:
            print("\nList of files in {0} channel: ".format(key))
            channel_path = data_files_path + key + "/"
            print_files_in_path(channel_path)

        if os.path.exists(resource_file_path):
            resource_config = load_json_object(resource_file_path)
            print("\nResource configuration:")
            print_json_object(resource_config)

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [
            os.path.join(data_files_path + "train/", file)
            for file in os.listdir(data_files_path + "train/")
        ]
        if len(input_files) == 0:
            raise ValueError(
                (
                    "There are no files in {}.\n"
                    + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                    + "the data specification in S3 was incorrectly specified or the role specified\n"
                    + "does not have permission to access the data."
                ).format(data_files_path + "train/", "train")
            )

        concat_data = load_raw(input_files, [label_column, feature_column])

        print(concat_data.info())

        preprocessor = CountVectorizer(analyzer=set)
        print("fitting...")
        preprocessor.fit(concat_data[feature_column])
        print("finished fitting...")

        feature_column_names = preprocessor.get_feature_names()
        print(feature_column_names)

        le = LabelEncoder()
        le.fit(concat_data[label_column])
        print("le classes: ", le.classes_)

        dump(preprocessor, os.path.join(model_artifacts_path, "model.joblib"))
        dump(le, os.path.join(model_artifacts_path, "label.joblib"))

        print("saved model!")
    except Exception as e:
        write_failure_file(failure_file_path, str(e))
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    print("Starting script")
    print("arguments:", sys.argv)
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "serve":
        serve()
    else:
        print("Launch argument: ", sys.argv[1])
        print("Missing required argument 'train'.", file=sys.stderr)
        sys.exit(1)
