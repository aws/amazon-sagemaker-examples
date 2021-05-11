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

import csv
import glob
import io
import os
import shutil
import tarfile
import tempfile

import chainer
import numpy
from src.nlp_utils import make_vocab, normalize_text, split_text, transform_to_array

URL_STSA_BASE = "https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/"


def download_dataset(name):
    files = [name + suff for suff in [".train", ".test"]]
    file_paths = []
    for f_name in files:
        url = os.path.join(URL_STSA_BASE, f_name)
        path = chainer.dataset.cached_download(url)
        file_paths.append(path)
    return file_paths


def get_stsa_dataset(file_paths, vocab=None, shrink=1, char_based=False, seed=777):
    train = read_dataset(file_paths[0], shrink=shrink, char_based=char_based)
    if len(file_paths) == 2:
        test = read_dataset(file_paths[1], shrink=shrink, char_based=char_based)
    else:
        numpy.random.seed(seed)
        alldata = numpy.random.permutation(train)
        train = alldata[: -len(alldata) // 10]
        test = alldata[-len(alldata) // 10 :]

    if vocab is None:
        vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab


def read_dataset(path, shrink=1, char_based=False):
    dataset = []
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for i, l in enumerate(f):
            if i % shrink != 0 or not len(l.strip()) >= 3:
                continue
            label, text = l.strip().split(None, 1)
            label = int(label)
            tokens = split_text(normalize_text(text), char_based)
            dataset.append((tokens, label))
    return dataset
