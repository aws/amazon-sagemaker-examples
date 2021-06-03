# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""
Train JAX model using purely functional code and serialize as TF SavedModel
"""

import argparse
import time
from itertools import count

from jax import random, grad, jit, numpy as jnp

from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, Relu, LogSoftmax, Flatten
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental import jax2tf


def load_fashion_mnist(split: tfds.Split, batch_size: int):
    ds = tfds.load("fashion_mnist", split=split)

    def _prepare_example(x):
        image = tf.cast(x["image"], tf.float32) / 255.0
        label = tf.one_hot(x["label"], 10)
        return image, label

    ds = ds.map(_prepare_example)
    # drop_remainder=True is important for use with Keras
    ds = ds.cache().shuffle(1000).batch(batch_size, drop_remainder=True)
    return ds


def init_nn():
    """
    Initialize Stax model. This function can be customized as needed to define architecture
    """
    layers = [
        Conv(16, (3, 3)),
        Relu,
        Conv(16, (3, 3)),
        Relu,
        Flatten,
        Dense(10),
        LogSoftmax,
    ]

    return stax.serial(*layers)


def get_acc_loss_and_update_fns(predict_fn, opt_update, get_params):
    def accuracy(params, ds):
        aggregate_mean = 0.0
        for batch in ds.as_numpy_iterator():
            inputs, targets = batch
            target_class = jnp.argmax(targets, axis=1)
            predicted_class = jnp.argmax(predict_fn(params, inputs), axis=1)
            aggregate_mean += jnp.mean(predicted_class == target_class)
        return aggregate_mean / len(ds)

    @jit
    def loss(params, inputs, labels):
        predictions = predict_fn(params, inputs)
        return -jnp.mean(jnp.sum(predictions * labels, axis=1))

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch[0], batch[1]), opt_state)

    return accuracy, loss, update


def train(train_ds, test_ds, num_epochs, step_size):
    """
    The primary training loop is defined here

    - Initialize optimizer
    - Initialize neural network. In Stax, this means get a `predict` function and a pytree of parameters/weights.
    - Initialize parameters according to random seed
    - Create the loss calculation and parameter/optimizer combination update
    - Run in for loop for `num_epochs`

    """
    rng = random.PRNGKey(42)
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)

    init_nn_params, predict_fn = init_nn()
    _, init_params = init_nn_params(rng, (-1, 28, 28, 1))
    opt_state = opt_init(init_params)
    itercount = count()

    accuracy, loss, update = get_acc_loss_and_update_fns(
        predict_fn, opt_update, get_params
    )

    for epoch in range(num_epochs):
        start_time = time.time()
        for train_batch in train_ds.as_numpy_iterator():
            opt_state = update(next(itercount), opt_state, train_batch)

        params = get_params(opt_state)
        train_acc = accuracy(params, train_ds)
        test_acc = accuracy(params, test_ds)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")

    return predict_fn, params


def save_model_tf(prediction_function, params_to_save):
    tf_fun = jax2tf.convert(prediction_function, enable_xla=False)
    param_vars = tf.nest.map_structure(lambda param: tf.Variable(param), params_to_save)

    tf_graph = tf.function(
        lambda inputs: tf_fun(param_vars, inputs),
        autograph=False,
        experimental_compile=True,
    )

    # This signature is needed for TensorFlow Serving use.
    signatures = {}
    signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ] = tf_graph.get_concrete_function(tf.TensorSpec((1, 28, 28, 1), tf.float32))

    wrapper = _ReusableSavedModelWrapper(tf_graph, param_vars)
    model_dir = "/opt/ml/model/1"
    tf.saved_model.save(wrapper, model_dir, signatures=signatures)


class _ReusableSavedModelWrapper(tf.train.Checkpoint):
    """Wraps a function and its parameters for saving to a SavedModel.
    Implements the interface described at
    https://www.tensorflow.org/hub/reusable_saved_models.
    """

    def __init__(self, tf_graph, param_vars):
        """Args:
        tf_graph: a tf.function taking one argument (the inputs), which can be
           be tuples/lists/dictionaries of np.ndarray or tensors. The function
           may have references to the tf.Variables in `param_vars`.
        param_vars: the parameters, as tuples/lists/dictionaries of tf.Variable,
           to be saved as the variables of the SavedModel.
        """
        super().__init__()
        # Implement the interface from https://www.tensorflow.org/hub/reusable_saved_models
        self.variables = tf.nest.flatten(param_vars)
        self.trainable_variables = [v for v in self.variables if v.trainable]
        # If you intend to prescribe regularization terms for users of the model,
        # add them as @tf.functions with no inputs to this list. Else drop this.
        self.regularization_losses = []
        self.__call__ = tf_graph


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    train_ds = load_fashion_mnist(tfds.Split.TRAIN, batch_size=args.batch_size)
    test_ds = load_fashion_mnist(tfds.Split.TEST, batch_size=args.batch_size)

    predict_fn, final_params = train(
        train_ds, test_ds, args.num_epochs, args.learning_rate
    )

    print("finished training")

    save_model_tf(predict_fn, final_params)
