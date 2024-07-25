from __future__ import print_function

import argparse
import gzip
import json
import logging
import os
import traceback

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

logging.basicConfig(level=logging.DEBUG)

# Define the model object


class SmallConv(Model):
    def __init__(self):
        super(SmallConv, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# Decode and preprocess data
def convert_to_numpy(data_dir, images_file, labels_file):
    """Byte string to numpy arrays"""
    with gzip.open(os.path.join(data_dir, images_file), "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(os.path.join(data_dir, labels_file), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return (images, labels)


def mnist_to_numpy(data_dir, train):
    """Load raw MNIST data into numpy array

    Args:
        data_dir (str): directory of MNIST raw data.
            This argument can be accessed via SM_CHANNEL_TRAINING

        train (bool): use training data

    Returns:
        tuple of images and labels as numpy array
    """

    if train:
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"

    return convert_to_numpy(data_dir, images_file, labels_file)


def normalize(x, axis):
    eps = np.finfo(float).eps

    mean = np.mean(x, axis=axis, keepdims=True)
    # avoid division by zero
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std


# Training logic


def train(args):
    # create data loader from the train / test channels
    x_train, y_train = mnist_to_numpy(data_dir=args.train, train=True)
    x_test, y_test = mnist_to_numpy(data_dir=args.test, train=False)

    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

    # normalize the inputs to mean 0 and std 1
    x_train, x_test = normalize(x_train, (1, 2)), normalize(x_test, (1, 2))

    # expand channel axis
    # tf uses depth minor convention
    x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

    # normalize the data to mean 0 and std 1
    train_loader = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train))
        .batch(args.batch_size)
    )

    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)

    model = SmallConv()
    model.compile()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        return

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        return

    print("Training starts ...")
    for epoch in range(args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for batch, (images, labels) in enumerate(train_loader):
            train_step(images, labels)

        for images, labels in test_loader:
            test_step(images, labels)

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {train_loss.result()}, "
            f"Accuracy: {train_accuracy.result() * 100}, "
            f"Test Loss: {test_loss.result()}, "
            f"Test Accuracy: {test_accuracy.result() * 100}"
        )

    # Save the model
    # A version number is needed for the serving container
    # to load the model
    version = "00000000"
    ckpt_dir = os.path.join(args.model_dir, version)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save(ckpt_dir)
    return


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)

    # Environment variables given by the training image
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])

    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
