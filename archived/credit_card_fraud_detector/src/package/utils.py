from pathlib import Path
import os
import json


def get_notebook_name():
    with open('/opt/ml/metadata/resource-metadata.json') as openfile:
        data = json.load(openfile)
    notebook_name = data['ResourceName']
    return notebook_name


def get_current_folder(global_variables):
    # if calling from a file
    if "__file__" in global_variables:
        current_file = Path(global_variables["__file__"])
        current_folder = current_file.parent.resolve()
    # if calling from a notebook
    else:
        current_folder = Path(os.getcwd())
    return current_folder
