# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     https://aws.amazon.com/apache-2-0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import argparse
import os
import shutil
import sys
import tarfile

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import xrange

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'


def download_and_extract(data_dir):
    # download CIFAR-10 if not already downloaded.
    tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir, CIFAR_DOWNLOAD_URL)
    tarfile.open(os.path.join(data_dir, CIFAR_FILENAME), 'r:gz').extractall(data_dir)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    return {
        'train': ['data_batch_%d' % i for i in xrange(1, 5)],
        'validation': ['data_batch_5'],
        'eval': ['test_batch'],
    }


def read_pickle_from_file(filename):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        if sys.version_info.major >= 3:
            return pickle.load(f, encoding='bytes')
        else:
            return pickle.load(f)


def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b'data']
            labels = data_dict[b'labels']

            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(data[i].tobytes()),
                        'label': _int64_feature(labels[i])
                    }))
            record_writer.write(example.SerializeToString())


def main(data_dir):
    print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
    download_and_extract(data_dir)

    file_names = _get_file_names()
    input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]

        mode_dir = os.path.join(data_dir, mode)
        output_file = os.path.join(mode_dir, mode + '.tfrecords')
        if not os.path.exists(mode_dir):
            os.makedirs(mode_dir)
        try:
            os.remove(output_file)
        except OSError:
            pass

        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file)

    print('Done!')
    shutil.rmtree(os.path.join(data_dir, 'cifar-10-batches-py'))
    os.remove(os.path.join(data_dir, 'cifar-10-python.tar.gz'))  # Remove the original .tzr.gz files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='',
        help='Directory to download and extract CIFAR-10 to.'
    )

    args = parser.parse_args()
    main(args.data_dir)
