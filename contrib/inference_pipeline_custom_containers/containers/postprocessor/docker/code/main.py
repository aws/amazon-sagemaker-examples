from __future__ import absolute_import
from __future__ import print_function
from string import Template

import sys
import time
import os
import multiprocessing
import signal
import subprocess
from urllib.parse import urlparse

from utils import ExitSignalHandler
from utils import write_failure_file, print_json_object, load_json_object, save_model_artifacts, print_files_in_path

import traceback
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd

from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

import boto3
import decimal

from botocore.exceptions import ClientError

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

cpu_count = multiprocessing.cpu_count()

model_server_timeout = os.environ.get('MODEL_SERVER_TIMEOUT', 60)
model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', cpu_count))


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
    print('Starting the inference server with {} workers.'.format(
        model_server_workers))

    port = os.environ.get('SAGEMAKER_BIND_TO_PORT', 8080)
    print("using port: ", port)
    with open("/nginx.conf.template") as nginx_template:
        template = Template(nginx_template.read())
        print("nginx.conf:", template.substitute(port=port))
        nginx_conf = open('/nginx.conf', 'w')
        nginx_conf.write(template.substitute(port=port))
        nginx_conf.close()

    # link the log streams to stdout/err so they will be logged to the container logs
    subprocess.check_call(
        ['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(
        ['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '/nginx.conf'])
    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers),
                                 'wsgi:app'])

    signal.signal(signal.SIGTERM, lambda a,
                  b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print('Inference server exiting')


if __name__ == "__main__":
    print("Starting script")
    print("arguments:", sys.argv)
    if (sys.argv[1] == "serve"):
        serve()
    else:
        print("Launch argument: ", sys.argv[1])
        print("Missing required argument 'train'.", file=sys.stderr)
        sys.exit(1)
