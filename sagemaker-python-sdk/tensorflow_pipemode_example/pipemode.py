import numpy as np
import os
import tensorflow as tf
from sagemaker_tensorflow import PipeModeDataset

from tensorflow.contrib.data import map_and_batch

PREFETCH_SIZE = 10
BATCH_SIZE = 64
NUM_PARALLEL_BATCHES = 2
DIMENSION = 1024
EPOCHS = 1

def estimator_fn(run_config, params):
    column = tf.feature_column.numeric_column('data', shape=(DIMENSION, ))
    return tf.estimator.LinearClassifier(feature_columns=[column], config=run_config)

def train_input_fn(training_dir, params):
    """Returns input function that would feed the model during training"""
    return _input_fn('train')

def eval_input_fn(training_dir, params):
    """Returns input function that would feed the model during evaluation"""
    return _input_fn('eval')

def _input_fn(channel):
    """Returns a Dataset for reading from a SageMaker PipeMode channel."""
    features = {
        'data': tf.FixedLenFeature([], tf.string),
        'labels': tf.FixedLenFeature([], tf.int64),
    }

    def parse(record):
        parsed = tf.parse_single_example(record, features)
        return ({
            'data': tf.decode_raw(parsed['data'], tf.float64)
        }, parsed['labels'])

    ds = PipeModeDataset(channel)
    if EPOCHS > 1:
        ds = ds.repeat(EPOCHS)
    ds = ds.prefetch(PREFETCH_SIZE)
    ds = ds.apply(map_and_batch(parse, batch_size=BATCH_SIZE,
                                num_parallel_batches=NUM_PARALLEL_BATCHES))

    return ds