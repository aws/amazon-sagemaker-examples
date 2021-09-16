# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import gzip
import os

import numpy as np


def load_train_dataset(droot):
    return _load_dataset(
        xfile=os.path.join(droot, "train-images-idx3-ubyte.gz"),
        yfile=os.path.join(droot, "train-labels-idx1-ubyte.gz"),
    )


def load_test_dataset(root):
    return _load_dataset(
        xfile=os.path.join(droot, "t10k-images-idx3-ubyte.gz"),
        yfile=os.path.join(droot, "t10k-labels-idx1-ubyte.gz"),
    )


def _load_dataset(xfile, yfile):
    with gzip.open(yfile, "rb") as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(xfile, "rb") as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)

    return x_test, y_test
