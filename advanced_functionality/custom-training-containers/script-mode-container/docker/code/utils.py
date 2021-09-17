import os
from os import path


def save_model_artifacts(model_artifacts_path, net):
    if path.exists(model_artifacts_path):
        model_file = open(model_artifacts_path + "model.dummy", "w")
        model_file.write("Dummy model.")
        model_file.close()


def print_files_in_path(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        print(f)
