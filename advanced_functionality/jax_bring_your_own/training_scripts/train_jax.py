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
Train JAX model and serialize as TF SavedModel
"""
import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.experimental import jax2tf


def load_fashion_mnist(split: tfds.Split, batch_size: int):
    ds = tfds.load("fashion_mnist", split=split)

    def _prepare_example(x):
        image = tf.cast(x["image"], tf.float32) / 255.0
        label = tf.one_hot(x["label"], 10)
        return (image, label)

    ds = ds.map(_prepare_example)
    # drop_remainder=True is important for use with Keras
    ds = ds.cache().shuffle(1000).batch(batch_size, drop_remainder=True)
    return ds


class PureJaxMNIST:
    name = "mnist_pure_jax"

    @staticmethod
    def predict(params, inputs, with_classifier=True):
        x = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))  # flatten to f32[B, 784]
        for w, b in params[:-1]:
            x = jnp.dot(x, w) + b
            x = jnp.tanh(x)

        if not with_classifier:
            return x
        final_w, final_b = params[-1]
        logits = jnp.dot(x, final_w) + final_b
        return logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)

    @staticmethod
    def loss(params, inputs, labels):
        predictions = PureJaxMNIST.predict(params, inputs, with_classifier=True)
        return -jnp.mean(jnp.sum(predictions * labels, axis=1))

    @staticmethod
    def accuracy(predict, params, dataset):
        @jax.jit
        def _per_batch(inputs, labels):
            target_class = jnp.argmax(labels, axis=1)
            predicted_class = jnp.argmax(predict(params, inputs), axis=1)
            return jnp.mean(predicted_class == target_class)

        batched = [_per_batch(inputs, labels) for inputs, labels in tfds.as_numpy(dataset)]
        return jnp.mean(jnp.stack(batched))

    @staticmethod
    def update(params, step_size, inputs, labels):
        grads = jax.grad(PureJaxMNIST.loss)(params, inputs, labels)
        return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

    @staticmethod
    def train(train_ds, test_ds, num_epochs, step_size, with_classifier=True):
        layer_sizes = [784, 512, 512, 10]

        rng = jax.random.PRNGKey(0)
        params = [
            (0.1 * jax.random.normal(rng, (m, n)), 0.1 * jax.random.normal(rng, (n,)))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        for epoch in range(num_epochs):
            start_time = time.time()
            for inputs, labels in tfds.as_numpy(train_ds):
                params = jax.jit(PureJaxMNIST.update)(params, step_size, inputs, labels)
            epoch_time = time.time() - start_time
            train_acc = PureJaxMNIST.accuracy(PureJaxMNIST.predict, params, train_ds)
            test_acc = PureJaxMNIST.accuracy(PureJaxMNIST.predict, params, test_ds)
            print(f"{PureJaxMNIST.name}: Epoch {epoch} in {epoch_time:0.2f} sec")
            print(f"{PureJaxMNIST.name}: Training set accuracy {train_acc}")
            print(f"{PureJaxMNIST.name}: Test set accuracy {test_acc}")

        return (functools.partial(PureJaxMNIST.predict, with_classifier=with_classifier), params)


def save_model_tf(prediction_function, params_to_save):
    tf_fun = jax2tf.convert(prediction_function, enable_xla=True)
    param_vars = tf.nest.map_structure(lambda param: tf.Variable(param), params_to_save)

    tf_graph = tf.function(
        lambda inputs: tf_fun(param_vars, inputs), autograph=False, experimental_compile=True
    )

    # This signature is needed for TensorFlow Serving use.
    signatures = {}
    signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tf_graph.get_concrete_function(
        tf.TensorSpec((1, 28, 28, 1), tf.float32)
    )

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

    (predict_fn, predict_params) = PureJaxMNIST.train(
        train_ds, test_ds, args.num_epochs, args.learning_rate, with_classifier=True
    )

    save_model_tf(predict_fn, predict_params)
