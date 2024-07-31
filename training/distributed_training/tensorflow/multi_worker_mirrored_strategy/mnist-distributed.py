# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
# language governing permissions and limitations under the License.import tensorflow as tf

import argparse
import json
import os

import numpy as np
import tensorflow as tf


def model(x_train, y_train, x_test, y_test, strategy):
    """Generate a simple model"""
    with strategy.scope():

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )

        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test)

    return model


def _load_training_data(base_dir):
    """Load MNIST training data"""
    x_train = np.load(os.path.join(base_dir, "input_train.npy"))
    y_train = np.load(os.path.join(base_dir, "input_train_labels.npy"))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load MNIST testing data"""
    x_test = np.load(os.path.join(base_dir, "input_test.npy"))
    y_test = np.load(os.path.join(base_dir, "input_test_labels.npy"))
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.train)

    print("Tensorflow version: ", tf.__version__)
    print("TF_CONFIG", os.environ.get("TF_CONFIG"))

    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    )
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=communication_options
    )

    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    mnist_classifier = model(train_data, train_labels, eval_data, eval_labels, strategy)

    task_type, task_id = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)

    print("Task type: ", task_type)
    print("Task id: ", task_id)

    # Save the model on chief worker
    if strategy.cluster_resolver.task_id == 0:
        print("Saving model on chief")
        mnist_classifier.save(os.path.join(args.sm_model_dir, "000000001"))
    else:
        print("Saving model in /tmp on worker")
        mnist_classifier.save(f"/tmp/{strategy.cluster_resolver.task_id}")
