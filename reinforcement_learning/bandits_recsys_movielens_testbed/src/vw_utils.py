import os
import json

TRAIN_CHANNEL = "training"
EVAL_CHANNEL = "evaluation"
MODEL_CHANNEL = "pretrained_model"
MODEL_OUTPUT_DIR = os.environ.get('SM_MODEL_DIR', "/opt/ml/model")
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, "vw.model")
DATA_OUTPUT_DIR = "/opt/ml/output/data"

def save_vw_metadata(meta):
    """
    Save metadata of a Vowpal Wabbit model.
    """
    file_location = os.path.join(MODEL_OUTPUT_DIR, "vw.metadata")
    with open(file_location, "w") as f:
        f.write(meta)


def save_vw_model(model=None, meta=None):
    """
    Save a Vowpal Wabbit model.
    """
    if model:
        model.save(MODEL_OUTPUT_PATH)
    save_vw_metadata(meta)


def transform_to_vw(x):
    """
    Transform context(feature) to VW format.
    """
    x = json.loads(x)
    # feature:feature_value
    return " ".join(["%s:%s" % (i + 1, j) for i, j in enumerate(x)])
