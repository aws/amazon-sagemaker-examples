# Standard Library
import argparse
import os
import time

# Third Party
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10

# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def train(batch_size, epoch, model):
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()

    Y_train = to_categorical(y_train, 10)
    Y_valid = to_categorical(y_valid, 10)

    X_train = X_train.astype("float32")
    X_valid = X_valid.astype("float32")

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_valid -= mean_image
    X_train /= 128.0
    X_valid /= 128.0

    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(X_valid, Y_valid),
        shuffle=True,
        verbose=0,
    )


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="./output_keras_resnet")
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")

    opt = parser.parse_args()

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # start the training.
    train(opt.batch_size, opt.epoch, model)


if __name__ == "__main__":
    main()
