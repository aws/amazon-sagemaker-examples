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
import io
import itertools
import json
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

logging.getLogger().setLevel(logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

HP_EPOCHS = hp.HParam("epochs", hp.IntInterval(1, 100))
HP_BATCH_SIZE = hp.HParam("batch-size", hp.Discrete([64, 128, 256, 512]))
HP_LR = hp.HParam("learning-rate", hp.RealInterval(0.0, 1.0))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["sgd", "adam", "rmsprop"]))

METRIC_ACCURACY = "accuracy"


def keras_model_fn(learning_rate, optimizer):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", name="inputs", input_shape=(HEIGHT, WIDTH, DEPTH)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, kernel_constraint=max_norm(2.0)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation("softmax"))

    if optimizer == "sgd":
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    else:
        raise Exception("Unknown optimizer", optimizer)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def read_dataset(epochs, batch_size, channel, channel_name):
    mode = args.data_config[channel_name]["TrainingInputMode"]

    logging.info("Running {} in {} mode".format(channel_name, mode))
    if mode == "Pipe":
        from sagemaker_tensorflow import PipeModeDataset

        dataset = PipeModeDataset(channel=channel_name, record_format="TFRecord")
    else:
        filenames = [os.path.join(channel, channel_name + ".tfrecords")]
        dataset = tf.data.TFRecordDataset(filenames)

    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        features = tf.io.parse_single_example(example_proto, image_feature_description)
        image = tf.io.decode_raw(features["image"], tf.uint8)
        image.set_shape([3 * 32 * 32])
        image = tf.reshape(image, [32, 32, 3])

        label = tf.cast(features["label"], tf.int32)
        label = tf.one_hot(label, 10)

        return image, label

    dataset = dataset.map(_parse_image_function, num_parallel_calls=10)
    dataset = dataset.prefetch(10)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def tb_plot_confusion_matrix(y_true, y_pred):
    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    classes_cnt = cm.shape[0]

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(classes_cnt)
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    cm_image = tf.expand_dims(image, 0)

    return cm_image


def main(args):
    # Initializing TensorFlow summary writer
    job_name = json.loads(os.environ.get("SM_TRAINING_ENV"))["job_name"]
    logs_dir = "{}/{}".format(args.tf_logs_path, job_name)
    logging.info("Writing TensorBoard logs to {}".format(logs_dir))
    tf_writer = tf.summary.create_file_writer(logs_dir)
    tf_writer.set_as_default()

    # Configuration of hyperparameters to visualize in TensorBoard
    hp.hparams_config(
        hparams=[HP_EPOCHS, HP_BATCH_SIZE, HP_LR, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name="Accuracy")],
    )

    hparams = {
        HP_EPOCHS: args.epochs,
        HP_BATCH_SIZE: args.batch_size,
        HP_LR: args.learning_rate,
        HP_OPTIMIZER: args.optimizer,
    }

    # Importing datasets
    train_dataset = read_dataset(args.epochs, args.batch_size, args.train, "train")
    validation_dataset = read_dataset(args.epochs, args.batch_size, args.validation, "validation")

    # Initializing and compiling hte model
    model = keras_model_fn(args.learning_rate, args.optimizer)

    callbacks = []
    # TensorBoard callback to collect standard metrics, profiling informationg, and compute activation and weight histograms for the layers
    callbacks.append(
        TensorBoard(log_dir=logs_dir, update_freq="epoch", histogram_freq=1, profile_batch="5,35")
    )
    # TensorBoard logging hyperparameter
    callbacks.append(hp.KerasCallback(writer=logs_dir, hparams=hparams, trial_id=job_name))

    # Train the model
    model.fit(
        x=train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

    # Saving trained model
    model.save(args.model_output + "/1")

    # Converting validation dataset to numpy array
    validation_array = np.array(list(validation_dataset.unbatch().take(-1).as_numpy_iterator()))
    test_x = np.stack(validation_array[:, 0])
    test_y = np.stack(validation_array[:, 1])

    # Use the model to predict the labels
    test_predictions = model.predict(test_x)
    test_y_pred = np.argmax(test_predictions, axis=1)
    test_y_true = np.argmax(test_y, axis=1)

    # Evaluating model accuracy and logging it as a scalar for TensorBoard hyperparameter visualization.
    accuracy = sklearn.metrics.accuracy_score(test_y_true, test_y_pred)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    logging.info("Test accuracy:{}".format(accuracy))

    # Calculating confusion matrix and logging it as an image for TensorBoard visualization.
    cm_image = tb_plot_confusion_matrix(test_y_true, test_y_pred)
    tf.summary.image("Confusion Matrix", cm_image, step=1)
    tf_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
        help="The directory where the CIFAR-10 input data is stored.",
    )
    parser.add_argument(
        "--validation",
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_VALIDATION"),
        help="The directory where the CIFAR-10 input data is stored.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
        help="The directory where the trained model will be stored.",
    )
    parser.add_argument(
        "--tf-logs-path",
        type=str,
        required=True,
        help="Path used for writing TensorFlow logs. Can be S3 bucket.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="The number of steps to use for training."
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument(
        "--data-config", type=json.loads, default=os.environ.get("SM_INPUT_DATA_CONFIG")
    )
    parser.add_argument("--optimizer", type=str.lower, default="adam")
    parser.add_argument("--model_dir", type=str)

    args = parser.parse_args()
    main(args)
