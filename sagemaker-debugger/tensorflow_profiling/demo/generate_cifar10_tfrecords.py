#     Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License").
#     You may not use this file except in compliance with the License.
#     A copy of the License is located at
#
#         https://aws.amazon.com/apache-2-0/
#
#     or in the "license" file accompanying this file. This file is distributed
#     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#     express or implied. See the License for the specific language governing
#     permissions and limitations under the License.

import argparse
import os
import sys
import tarfile

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin

# import tensorflow_datasets as tfds

CIFAR_FILENAME = "cifar-10-python.tar.gz"
CIFAR_DOWNLOAD_URL = "https://www.cs.toronto.edu/~kriz/" + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = "cifar-10-batches-py"


def download_and_extract(data_dir):
    import tensorflow_datasets as tfds

    dm = tfds.download.DownloadManager(download_dir=data_dir + "/tmp")
    extract_dir = dm.download_and_extract(CIFAR_DOWNLOAD_URL)

    return extract_dir


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names["train"] = ["data_batch_%d" % i for i in xrange(1, 5)]
    file_names["validation"] = ["data_batch_5"]
    file_names["eval"] = ["test_batch"]
    return file_names


def read_pickle_from_file(filename):
    # with open(filename, 'rb') as f:
    with tf.io.gfile.GFile(filename, "rb") as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding="bytes")
        else:
            data_dict = pickle.load(f)
    return data_dict


def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print("Generating %s" % output_file)
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b"data"]
            labels = data_dict[b"labels"]

            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": _bytes_feature(data[i].tobytes()),
                            "label": _int64_feature(labels[i]),
                        }
                    )
                )
                record_writer.write(example.SerializeToString())


def install_dependencies():
    from subprocess import call

    call(["pip", "install", "--upgrade", "pip"])
    call(["pip", "install", "tensorflow_datasets==4.1.0"])


def main(data_dir):
    print("Download from {} and extract.".format(CIFAR_DOWNLOAD_URL))

    extract_dir = download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join(extract_dir, CIFAR_LOCAL_FOLDER)

    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir + "/" + mode, mode + ".tfrecords")
        if not os.path.exists(data_dir + "/" + mode):
            os.makedirs(data_dir + "/" + mode)
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file)
    print("Done!")
    import shutil

    shutil.rmtree(data_dir + "/tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to download and extract CIFAR-10 to.",
    )

    args = parser.parse_args()

    install_dependencies()

    main(args.data_dir)
