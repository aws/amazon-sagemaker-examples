import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "mxnet"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])

import os
import pathlib

import cv2
import h5py
import mxnet as mx
import numpy as np
from sklearn.model_selection import train_test_split


def write_to_recordio(X: np.ndarray, y: np.ndarray, prefix: str):
    record = mx.recordio.MXIndexedRecordIO(idx_path=f"{prefix}.idx", uri=f"{prefix}.rec", flag="w")
    for idx, arr in enumerate(X):
        header = mx.recordio.IRHeader(0, y[idx], idx, 0)
        s = mx.recordio.pack_img(
            header,
            arr,
            quality=95,
            img_fmt=".jpg",
        )
        record.write_idx(idx, s)
    record.close()


if __name__ == "__main__":

    #     input_path = pathlib.Path('/opt/ml/processing/input')
    #     input_file = list(input_path.glob('*.h5'))[0]

    input_file = "/opt/ml/processing/input/camelyon16_tiles.h5"
    f = h5py.File(input_file, "r")
    X = f["x"]
    y = f["y"]

    X_numpy = X[:]
    y_numpy = y[:]

    X_train, X_test, y_train, y_test = train_test_split(
        X_numpy, y_numpy, test_size=1000, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=2000, random_state=1
    )

    output_dir = "/opt/ml/processing/output/data"

    write_to_recordio(X_train, y_train, prefix=f"{output_dir}/train/train")
    write_to_recordio(X_val, y_val, prefix=f"{output_dir}/val/val")
    write_to_recordio(X_test, y_test, prefix=f"{output_dir}/test/test")

    # we do not need the idx files so we remove them to prevent them from becoming part of the output
    os.remove(f"{output_dir}/train/train.idx")
    os.remove(f"{output_dir}/val/val.idx")
    os.remove(f"{output_dir}/test/test.idx")
