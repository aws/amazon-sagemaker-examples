#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from __future__ import print_function

import argparse
import os

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import serializers, training
from chainer.datasets import tuple_dataset
from chainer.training import extensions


# Define the network to train MNIST
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # retrieve the hyperparameters we set from the client (with some defaults)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)

    # Data, model, and output directories. These are required.
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    num_gpus = int(os.environ["SM_NUM_GPUS"])

    train_data = np.load(os.path.join(args.train, "train.npz"))["images"]
    train_labels = np.load(os.path.join(args.train, "train.npz"))["labels"]

    test_data = np.load(os.path.join(args.test, "test.npz"))["images"]
    test_labels = np.load(os.path.join(args.test, "test.npz"))["labels"]

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    # Create the network
    model = L.Classifier(MLP(1000, 10))

    # Configure gpu if necessary
    if num_gpus > 0:
        chainer.cuda.get_device_from_id(0).use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    # Set up a trainer
    device = 0 if num_gpus > 0 else -1  # -1 indicates CPU, 0 indicates first GPU device.
    if num_gpus > 0:
        updater = training.ParallelUpdater(
            train_iter,
            optimizer,
            # The device of the name 'main' is used as a "master", while others are
            # used as slaves. Names other than 'main' are arbitrary.
            devices={
                ("main" if device == 0 else str(device)): device for device in range(num_gpus)
            },
        )
    else:
        updater = training.StandardUpdater(train_iter, optimizer, device=device)

    # Write output files to output_data_dir. These are zipped and uploaded to S3 output path as output.tar.gz.
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.output_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph("main/loss"))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epochs, "epoch"))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name=None))

    # Save two plot images to the result dir
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
    model = L.Classifier(MLP(1000, 10))
    serializers.load_npz(os.path.join(model_dir, "model.npz"), model)
    return model.predictor
