import argparse
import json
import os
import tensorflow as tf
from sagemaker_tensorflow import PipeModeDataset

from tensorflow.contrib.data import map_and_batch

PREFETCH_SIZE = 10
BATCH_SIZE = 64
NUM_PARALLEL_BATCHES = 2
DIMENSION = 1024
EPOCHS = 1


def train_input_fn():
    """Returns input function that would feed the model during training"""
    return _input_fn('train')


def eval_input_fn():
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


def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))


    return parser.parse_known_args()


def serving_input_fn():
    inputs = {'data': tf.placeholder(tf.string)}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


if __name__ == "__main__":
    args, unknown = _parse_args()

    column = tf.feature_column.numeric_column('data', shape=(DIMENSION, ))
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=3000)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=[column], model_dir=args.model_dir)
    tf.estimator.train_and_evaluate(linear_classifier, train_spec, eval_spec)

    if args.current_host == args.hosts[0]:
        linear_classifier.export_savedmodel(args.sm_model_dir, serving_input_fn)
