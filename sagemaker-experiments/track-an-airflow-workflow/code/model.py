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

import tensorflow as tf


def get_model(filters=[64, 32], hidden_units=256, dropouts=[0.3, 0.3, 0.5], num_class=10):
    if len(filters) != 2:
        raise ValueError("Please provide 2 filter size.")

    if len(dropouts) != 3:
        raise ValueError("Please provide 3 dropout layer size")

    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(
        tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=2,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(dropouts[0]))
    model.add(
        tf.keras.layers.Conv2D(filters=filters[1], kernel_size=2, padding="same", activation="relu")
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(dropouts[1]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hidden_units, activation="relu"))
    model.add(tf.keras.layers.Dropout(dropouts[2]))
    model.add(tf.keras.layers.Dense(num_class, activation="softmax"))

    return model
