"""
This script is a simple MNIST training script which uses Tensorflow's Estimator interface.
It has been orchestrated with SageMaker Debugger hooks to allow saving tensors during training.
These hooks have been instrumented to read from json configuration that SageMaker will put in the training container.
Configuration provided to the SageMaker python SDK when creating a job will be passed on to the hook.
This allows you to use the same script with differing configurations across different runs. 
If you use an official SageMaker Framework container (i.e. AWS Deep Learning Container), then 
you do not have to orchestrate your script as below. Hooks will automatically be added in those environments.
For more information, please refer to https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md 
"""

# Standard Library
import argparse
import logging
import random

# Third Party
import numpy as np
import smdebug.tensorflow as smd
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--random_seed", type=bool, default=False)
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for")
parser.add_argument(
    "--num_steps",
    type=int,
    help="Number of steps to train for. If this" "is passed, it overrides num_epochs",
)
parser.add_argument(
    "--num_eval_steps",
    type=int,
    help="Number of steps to evaluate for. If this"
    "is passed, it doesnt evaluate over the full eval set",
)
parser.add_argument("--model_dir", type=str, default="/tmp/mnist_model")
args = parser.parse_args()

if args.random_seed:
    tf.set_random_seed(2)
    np.random.seed(2)
    random.seed(12)

# This allows you to create the hook from the configuration you pass to the SageMaker pySDK
hook = smd.SessionHook.create_from_json_file()


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)

        # SMD: Wrap your optimizer as follows to help SageMaker Debugger identify gradients
        # This does not change your optimization logic, it returns back the same optimizer
        optimizer = hook.wrap_optimizer(optimizer)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Load training and eval data
((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data / np.float32(255)
train_labels = train_labels.astype(np.int32)  # not required

eval_data = eval_data / np.float32(255)
eval_labels = eval_labels.astype(np.int32)  # not required

mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=args.model_dir)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data}, y=train_labels, batch_size=128, num_epochs=args.num_epochs, shuffle=True
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False
)

# Set training mode so SMDebug can classify the steps into training mode
hook.set_mode(smd.modes.TRAIN)
mnist_classifier.train(input_fn=train_input_fn, steps=args.num_steps, hooks=[hook])

# Set eval mode so SMDebug can classify the steps into eval mode
hook.set_mode(smd.modes.EVAL)
mnist_classifier.evaluate(input_fn=eval_input_fn, steps=args.num_eval_steps, hooks=[hook])
