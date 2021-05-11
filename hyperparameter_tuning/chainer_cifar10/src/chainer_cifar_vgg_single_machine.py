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

from __future__ import absolute_import, print_function

import argparse
import os

import chainer
import chainer.functions as F
import chainer.links as L
import net
import numpy as np
from chainer import serializers, training
from chainer.training import extensions

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # retrieve the hyperparameters we set from the client (with some defaults)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)

    # Data, model, and output directories These are required.
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    num_gpus = int(os.environ["SM_NUM_GPUS"])

    train_data = np.load(os.path.join(args.train, "train.npz"))["data"]
    train_labels = np.load(os.path.join(args.train, "train.npz"))["labels"]

    test_data = np.load(os.path.join(args.test, "test.npz"))["data"]
    test_labels = np.load(os.path.join(args.test, "test.npz"))["labels"]

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    print("# Minibatch-size: {}".format(args.batch_size))
    print("# epoch: {}".format(args.epochs))
    print("# learning rate: {}".format(args.learning_rate))

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(net.VGG(10))

    optimizer = chainer.optimizers.MomentumSGD(args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Set up a trainer
    device = 0 if num_gpus > 0 else -1  # -1 indicates CPU, 0 indicates first GPU device.
    if num_gpus > 1:
        devices = range(num_gpus)
        train_iters = [
            chainer.iterators.MultiprocessIterator(i, args.batch_size, n_processes=4)
            for i in chainer.datasets.split_dataset_n_random(train, len(devices))
        ]
        test_iter = chainer.iterators.MultiprocessIterator(
            test, args.batch_size, repeat=False, n_processes=num_gpus
        )
        updater = training.updaters.MultiprocessParallelUpdater(
            train_iters, optimizer, devices=range(num_gpus)
        )
    else:
        train_iter = chainer.iterators.MultiprocessIterator(train, args.batch_size)
        test_iter = chainer.iterators.MultiprocessIterator(test, args.batch_size, repeat=False)
        updater = training.updater.StandardUpdater(train_iter, optimizer, device=device)

    stop_trigger = (args.epochs, "epoch")

    output_data_dir = os.path.join(args.output_dir, "data")
    trainer = training.Trainer(updater, stop_trigger, out=output_data_dir)
    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift("lr", 0.5), trigger=(25, "epoch"))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph("main/loss"))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ["main/loss", "validation/main/loss"], "epoch", file_name="loss.png"
            )
        )
        trainer.extend(
            extensions.PlotReport(
                ["main/accuracy", "validation/main/accuracy"], "epoch", file_name="accuracy.png"
            )
        )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "main/loss",
                "validation/main/loss",
                "main/accuracy",
                "validation/main/accuracy",
                "elapsed_time",
            ]
        )
    )

    # Run the training
    trainer.run()

    # Save the model to model_dir. It's loaded below in `model_fn`.
    serializers.save_npz(os.path.join(args.model_dir, "model.npz"), model)


def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.

    This function loads models written during training into `model_dir`.

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model

    For more on `model_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    chainer.config.train = False
    model = L.Classifier(net.VGG(10))
    serializers.load_npz(os.path.join(model_dir, "model.npz"), model)
    return model.predictor
