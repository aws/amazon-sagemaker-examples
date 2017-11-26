import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001


def model_fn(features, labels, mode, params):
    """Model function for Estimator.
     # Logic to do the following:
     # 1. Configure the model via Keras functional api
     # 2. Define the loss function for training/evaluation using Tensorflow.
     # 3. Define the training operation/optimizer using Tensorflow operation/optimizer.
     # 4. Generate predictions as Tensorflow tensors.
     # 5. Generate necessary evaluation metrics.
     # 6. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object"""

    # 1. Configure the model via Keras functional api

    first_hidden_layer = tf.keras.layers.Dense(10, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])
    second_hidden_layer = tf.keras.layers.Dense(10, activation='relu')(first_hidden_layer)
    output_layer = tf.keras.layers.Dense(1, activation='linear')(second_hidden_layer)

    predictions = tf.reshape(output_layer, [-1])

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"ages": predictions},
            export_outputs={SIGNATURE_NAME: PredictOutput({"ages": predictions})})

    # 2. Define the loss function for training/evaluation using Tensorflow.
    loss = tf.losses.mean_squared_error(labels, predictions)

    # 3. Define the training operation/optimizer using Tensorflow operation/optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="SGD")

    # 4. Generate predictions as Tensorflow tensors.
    predictions_dict = {"ages": predictions}

    # 5. Generate necessary evaluation metrics.
    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32), predictions)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=[1, 7])
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()


params = {"learning_rate": LEARNING_RATE}


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'abalone_train.csv')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'abalone_test.csv')


def _input_fn(training_dir, training_filename):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()
