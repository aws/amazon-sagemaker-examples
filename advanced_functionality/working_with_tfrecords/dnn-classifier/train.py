#!/usr/bin/env python

import argparse
import os
import tensorflow as tf
import time

from os.path import isfile, join


parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parser.add_argument('--data_dir', type=str, default='/opt/ml/input',
                    help='directory containing dataset')
parser.add_argument('--model_dir', type=str, default='save',
                    help='directory to store checkpointed models')
args = parser.parse_args()


def parser(serialized_example):
    """
    Parses a single tf.Example into image and label tensors.
    Should return a (data, labels) tuple where data is a features dict
    """
    features={
          'sepal_length': tf.FixedLenFeature([], tf.float32),
          'sepal_width': tf.FixedLenFeature([], tf.float32),
          'petal_length': tf.FixedLenFeature([], tf.float32),
          'petal_width': tf.FixedLenFeature([], tf.float32),
          'label': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(serialized_example, features)
    features = dict({
        'sepal_length': tf.cast(parsed_features['sepal_length'], tf.float32),
        'sepal_width': tf.cast(parsed_features['sepal_width'], tf.float32),
        'petal_length': tf.cast(parsed_features['petal_length'], tf.float32),
        'petal_width': tf.cast(parsed_features['petal_width'], tf.float32),
    })
    label = tf.cast(parsed_features['label'], tf.int64)
    return features, label


def input_fn_train():
    """Training input function that iterates over TFRecord files in data_dir"""
    dataset_files = [f for f in os.listdir(args.data_dir) if isfile(join(args.data_dir, f))]

    dataset = tf.data.TFRecordDataset(dataset_files)
    dataset = dataset.map(parser)
    dataset = dataset.batch(1)
    dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def serving_input_receiver_fn():
    """A serving input receiver that expects features encoded as JSON """
    features = {
        'sepal_length': tf.placeholder(tf.float32, [None, 1]),
        'sepal_width': tf.placeholder(tf.float32, [None, 1]),
        'petal_length': tf.placeholder(tf.float32, [None, 1]),
        'petal_width': tf.placeholder(tf.float32, [None, 1]),
    }
    return tf.estimator.export.ServingInputReceiver(features, features)


def train(args):
    feature_columns = [tf.feature_column.numeric_column(key="sepal_length", dtype=tf.float32),
                          tf.feature_column.numeric_column(key="sepal_width", dtype=tf.float32),
                          tf.feature_column.numeric_column(key="petal_length", dtype=tf.float32),
                          tf.feature_column.numeric_column(key="petal_width", dtype=tf.float32)]

    estimator =  tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3)

    estimator.train(input_fn=input_fn_train, steps=1000)

    print(f'Saving model to {args.model_dir}')
    estimator.export_saved_model(args.model_dir, serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == '__main__':
    train(args)