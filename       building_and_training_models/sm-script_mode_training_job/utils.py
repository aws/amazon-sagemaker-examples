"""Converts MNIST data to TFRecords file format with Example protos."""
import os

import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name, directory):
    """Converts a dataset to tfrecords."""
    num_examples = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(directory, name + ".tfrecords")
    print("Writing", filename)
    writer = tf.compat.v1.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "height": _int64_feature(rows),
                    "width": _int64_feature(cols),
                    "depth": _int64_feature(depth),
                    "label": _int64_feature(int(labels[index])),
                    "image_raw": _bytes_feature(image_raw),
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()
