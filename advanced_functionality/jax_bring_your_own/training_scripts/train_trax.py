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
Train Trax model and serialize as TF SavedModel
"""
import argparse

import tensorflow as tf
import trax
from trax import layers as tl
from trax.supervised import training


def get_model(n_output_classes=10):
    """
    Simple CNN to classify Fashion MNIST
    """
    model = tl.Serial(
        tl.ToFloat(),
        tl.Conv(32, (3, 3), (1, 1), "SAME"),
        tl.LayerNorm(),
        tl.Relu(),
        tl.MaxPool(),
        tl.Conv(64, (3, 3), (1, 1), "SAME"),
        tl.LayerNorm(),
        tl.Relu(),
        tl.MaxPool(),
        tl.Flatten(),
        tl.Dense(n_output_classes),
    )

    return model


def save_model_tf(model_to_save):
    """
    Serialize a TensorFlow graph from trained Trax Model
    :param model_to_save: Trax Model
    """
    keras_layer = trax.AsKeras(model_to_save, batch_size=1)
    inputs = tf.keras.Input(shape=(28, 28, 1))
    hidden = keras_layer(inputs)

    keras_model = tf.keras.Model(inputs=inputs, outputs=hidden)
    keras_model.save("/opt/ml/model/1", save_format="tf")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load Dataset from TensorFlow DataSets
    train_stream = trax.data.TFDS("fashion_mnist", keys=("image", "label"), train=True)()
    eval_stream = trax.data.TFDS("fashion_mnist", keys=("image", "label"), train=False)()

    # Create Data Pipelines
    train_data_pipeline = trax.data.Serial(
        trax.data.Shuffle(),
        trax.data.Batch(8),
    )
    train_batches_stream = train_data_pipeline(train_stream)

    eval_data_pipeline = trax.data.Batch(1)
    eval_batches_stream = eval_data_pipeline(eval_stream)

    # Define Train and Eval tasks using Trax Training
    train_task = training.TrainTask(
        labeled_data=train_batches_stream,
        loss_layer=tl.CategoryCrossEntropy(),
        optimizer=trax.optimizers.Adam(args.learning_rate),
    )

    eval_task = training.EvalTask(
        labeled_data=eval_batches_stream,
        metrics=[tl.CategoryCrossEntropy(), tl.CategoryAccuracy()],
        n_eval_batches=20,
    )

    # Train Model
    model = get_model(n_output_classes=10)
    training_loop = training.Loop(model, train_task, eval_tasks=[eval_task])
    training_loop.run(args.train_steps)

    # Save Model
    save_model_tf(model)
