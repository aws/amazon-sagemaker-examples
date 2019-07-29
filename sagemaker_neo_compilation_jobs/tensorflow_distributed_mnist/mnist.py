import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = 'predictions'

LEARNING_RATE = 0.001


def model_fn(features, labels, mode, params):
    # Input Layer
    input_layer = tf.reshape(features[INPUT_TENSOR_NAME], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == Modes.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(label_indices, depth=10), logits=logits)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 784])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'train.tfrecords', batch_size=100)


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'test.tfrecords', batch_size=100)


def _input_fn(training_dir, training_filename, batch_size=100):
    test_file = os.path.join(training_dir, training_filename)
    filename_queue = tf.train.string_input_producer([test_file])

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        capacity=1000 + 3 * batch_size)

    return {INPUT_TENSOR_NAME: images}, labels

def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io

    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'application/x-image' and content_type != 'application/vnd+python.numpy+binary':
        raise RuntimeError('Content type must be application/x-image or application/vnd+python.numpy+binary')
    
    f = io.BytesIO(payload)
    image = np.load(f)*255

    return image

### NOTE: this function cannot use MXNet
def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    logging.info('Invoking user-defined post-processing function')
    
    # Softmax (assumes batch size 1)
    result = np.squeeze(result)
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())
    content_type = 'application/json'

    return response_body, content_type
