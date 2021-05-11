import json
import os
import pprint
import signal
from os import path


class ExitSignalHandler:
    def __init__(self):
        self.exit_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.exit_now = True


def write_failure_file(failure_file_path, failure_reason):
    failure_file = open(failure_file_path, "w")
    failure_file.write(failure_reason)
    failure_file.close()


def save_model_artifacts(model_artifacts_path, net):
    if path.exists(model_artifacts_path):
        model_file = open(model_artifacts_path + "model.dummy", "w")
        model_file.write("Dummy model.")
        model_file.close()


def print_json_object(json_object):
    pprint.pprint(json_object)


def load_json_object(json_file_path):
    with open(json_file_path) as json_file:
        return json.load(json_file)


def print_files_in_path(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        print(f)
