import glob
import os
import re
import shutil
from rl_coach.logger import screen

def save_onnx_model():
    from .onnx_utils import fix_onnx_model
    ckpt_dir = '/opt/ml/output/data/checkpoint'
    model_dir = '/opt/ml/model'
    # find latest onnx file
    # currently done by name, expected to be changed in future release of coach.
    glob_pattern = os.path.join(ckpt_dir, '*.onnx')
    onnx_files = [file for file in glob.iglob(glob_pattern, recursive=True)]
    if len(onnx_files) > 0:
        extract_step = lambda string: int(re.search('/(\d*)_Step.*', string, re.IGNORECASE).group(1))
        onnx_files.sort(key=extract_step)
        latest_onnx_file = onnx_files[-1]
        # move to model directory
        filepath_from = os.path.abspath(latest_onnx_file)
        filepath_to = os.path.join(model_dir, "model.onnx")
        shutil.move(filepath_from, filepath_to)
        fix_onnx_model(filepath_to)
    else:
        screen.warning("No ONNX files found in {}".format(ckpt_dir))
