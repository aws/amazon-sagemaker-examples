
import os

os.system("pip install -U sagemaker")

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import argparse

from sagemaker.session import Session
from sagemaker.experiments import load_run

import boto3

boto_session = boto3.session.Session(region_name=os.environ["REGION"])
sagemaker_session = Session(boto_session=boto_session)
s3 = boto3.client("s3")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.01)

    return parser.parse_known_args()


class ExperimentCallback(keras.callbacks.Callback):
    """ """

    def __init__(self, run, model, x_test, y_test):
        """Save params in constructor"""
        self.run = run
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        """ """
        keys = list(logs.keys())
        for key in keys:
            self.run.log_metric(name=key, value=logs[key], step=epoch)
            print("{} -> {}".format(key, logs[key]))


def load_data():
    num_classes = 10
    input_shape = (28, 28, 1)

    train_path = "input_train.npy"
    test_path = "input_test.npy"
    train_labels_path = "input_train_labels.npy"
    test_labels_path = "input_test_labels.npy"

    # Load the data and split it between train and test sets
    s3.download_file(
        f"sagemaker-example-files-prod-{os.environ['REGION']}", "datasets/image/MNIST/numpy/input_train.npy", train_path
    )
    s3.download_file(
        f"sagemaker-example-files-prod-{os.environ['REGION']}", "datasets/image/MNIST/numpy/input_test.npy", test_path
    )
    s3.download_file(
        f"sagemaker-example-files-prod-{os.environ['REGION']}",
        "datasets/image/MNIST/numpy/input_train_labels.npy",
        train_labels_path,
    )
    s3.download_file(
        f"sagemaker-example-files-prod-{os.environ['REGION']}",
        "datasets/image/MNIST/numpy/input_test_labels.npy",
        test_labels_path,
    )

    x_train = np.load(train_path)
    x_test = np.load(test_path)
    y_train = np.load(train_labels_path)
    y_test = np.load(test_labels_path)

    # Reshape the arrays
    x_train = np.reshape(x_train, (60000, 28, 28))
    x_test = np.reshape(x_test, (10000, 28, 28))
    y_train = np.reshape(y_train, (60000,))
    y_test = np.reshape(y_test, (10000,))

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


def main():
    """ """
    args, _ = parse_args()
    print("Args are : ", args)
    num_classes = 10
    input_shape = (28, 28, 1)
    x_train, x_test, y_train, y_test = load_data()

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(args.dropout),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = args.batch_size
    epochs = args.epochs

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    ###
    # `load_run` will use the run defined when calling the estimator
    ###
    with load_run(sagemaker_session=sagemaker_session) as run:
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[ExperimentCallback(run, model, x_test, y_test)],
        )

        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        run.log_metric(name="Final Test Loss", value=score[0])
        run.log_metric(name="Final Test Accuracy", value=score[1])

        model.save("/opt/ml/model")


if __name__ == "__main__":
    main()
