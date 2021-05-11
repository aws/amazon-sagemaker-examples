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
import chainermn
import net
import numpy as np
from chainer import initializers, serializers, training
from chainer.training import extensions

if __name__ == "__main__":

    num_gpus = int(os.environ["SM_NUM_GPUS"])

    parser = argparse.ArgumentParser()

    # retrieve the hyperparameters we set from the client (with some defaults)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--communicator", type=str, default="pure_nccl" if num_gpus > 0 else "naive"
    )

    # Data, model, and output directories. These are required.
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    train_data = np.load(os.path.join(args.train, "train.npz"))["data"]
    train_labels = np.load(os.path.join(args.train, "train.npz"))["labels"]

    test_data = np.load(os.path.join(args.test, "test.npz"))["data"]
    test_labels = np.load(os.path.join(args.test, "test.npz"))["labels"]

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(net.VGG(10))

    comm = chainermn.create_communicator(args.communicator)

    # comm.inter_rank gives the rank of the node. This should only print on one node.
    if comm.inter_rank == 0:
        print("# Minibatch-size: {}".format(args.batch_size))
        print("# epoch: {}".format(args.epochs))
        print("# communicator: {}".format(args.communicator))

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.

    # comm.intra_rank gives the rank of the process on a given node.
    device = comm.intra_rank if num_gpus > 0 else -1
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(args.learning_rate), comm
    )
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    num_loaders = 2
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batch_size, n_processes=num_loaders
    )
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batch_size, repeat=False, n_processes=num_loaders
    )

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    output_data_dir = os.path.join(args.output_dir, "data")
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=output_data_dir)

    # Evaluate the model with the test dataset for each epoch

    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift("lr", 0.5), trigger=(25, "epoch"))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph("main/loss"))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    if comm.rank == 0:
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

        trainer.extend(extensions.dump_graph("main/loss"))

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

    # Save the model (only on one host).
    if comm.rank == 0:
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
