# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import argparse
import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from data_util import load_test_dataset, load_train_dataset
from model import get_model
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

#  Copy inference pre/post-processing script so it will be included in the model package
os.system("mkdir /opt/ml/model/code")
os.system("cp inference.py /opt/ml/model/code")
os.system("cp requirements.txt /opt/ml/model/code")


def save_model(model, path):
    tf.contrib.saved_model.save_keras_model(model, f"{path}/SavedModel")
    logging.info("Model successfully saved at: {}".format(path))


def main(args):
    model = get_model(
        filters=args.filter_sizes,
        hidden_units=args.hidden_size,
        dropouts=args.dropout_sizes,
        num_class=args.num_classes,
    )

    # load training data
    x, y = load_train_dataset(droot=args.train_dir)
    # one-hot encode label
    one_hot_y = np.zeros((y.shape[0], args.num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    # split x and y into train and val set
    X_train, X_val, y_train, y_val = train_test_split(
        x, one_hot_y, test_size=args.test_size, random_state=42, shuffle=True
    )

    # normalize the x image
    X_train = X_train / 255
    X_val = X_val / 255

    opt = tf.keras.optimizers.Adam(args.learning_rate, args.momentum)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["categorical_crossentropy", "accuracy"],
    )

    # a callback to save model ckpt after each epoch if better model is found
    model_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        args.output_data_dir + "/checkpoint-{epoch}.h5",
        monitor="val_accuracy",
    )

    logging.info("Start training ...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[model_ckpt_callback],
        verbose=2,
    )

    save_model(model, args.model_output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filter-sizes", nargs=2, type=int, default=[64, 32], help="Filter size with length of 2"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Feed-forward layer hidden unit size."
    )
    parser.add_argument(
        "--dropout-sizes",
        nargs=3,
        type=float,
        default=[0.3, 0.3, 0.5],
        help="Dropout layer size with length of 2",
    )
    parser.add_argument(
        "--num-classes", type=int, default=10, help="Num of class in classification task."
    )
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model-output-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))

    args = parser.parse_args()

    main(args)
