import json
import os
from pathlib import Path

def get_current_folder(global_variables):
    # if calling from a file
    if "__file__" in global_variables:
        current_file = Path(global_variables["__file__"])
        current_folder = current_file.parent.resolve()
    # if calling from a notebook
    else:
        current_folder = Path(os.getcwd())
    return current_folder

current_folder = get_current_folder(globals())


DATASETS_S3_PREFIX = 'datasets'
OUTPUTS_S3_PREFIX_RF = 'outputs_rf'
OUTPUTS_S3_PREFIX_AG_ENSEMBLE = 'outputs_ag_ensemble'
OUTPUTS_S3_PREFIX_AG_FUSION = 'outputs_ag_fusion'


TRAINING_INSTANCE_TYPE = "ml.p3.2xlarge"
HOSTING_INSTANCE_TYPE = "ml.g4dn.2xlarge"

TAG_KEY = 'sagemaker-soln'
SOLUTION_PREFIX = 'sagemaker-soln-churn-js-'
