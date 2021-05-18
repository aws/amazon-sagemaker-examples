# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.:wq
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
import time
import zipfile

import horovod.mxnet as hvd
import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.test_utils import download


def main(args):
    # Function to get mnist iterator given a rank
    def get_mnist_iterator(rank):
        data_dir = "data-%d" % rank
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        zip_file_path = download("http://data.mxnet.io/mxnet/data/mnist.zip", dirname=data_dir)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(data_dir)

        input_shape = (1, 28, 28)
        batch_size = args.batch_size

        train_iter = mx.io.MNISTIter(
            image="%s/train-images-idx3-ubyte" % data_dir,
            label="%s/train-labels-idx1-ubyte" % data_dir,
            input_shape=input_shape,
            batch_size=batch_size,
            shuffle=True,
            flat=False,
            num_parts=hvd.size(),
            part_index=hvd.rank(),
        )

        val_iter = mx.io.MNISTIter(
            image="%s/t10k-images-idx3-ubyte" % data_dir,
            label="%s/t10k-labels-idx1-ubyte" % data_dir,
            input_shape=input_shape,
            batch_size=batch_size,
            flat=False,
        )

        return train_iter, val_iter

    kernel_size = 5
    strides = 2
    pool_size = 2
    hidden_dim = 512
    output_dim = 10
    activation = "relu"

    # Function to define neural network
    def conv_nets():
        net = gluon.nn.HybridSequential()
        with net.name_scope():
            net.add(gluon.nn.Conv2D(channels=20, kernel_size=kernel_size, activation=activation))
            net.add(gluon.nn.MaxPool2D(pool_size=pool_size, strides=strides))
            net.add(gluon.nn.Conv2D(channels=50, kernel_size=kernel_size, activation=activation))
            net.add(gluon.nn.MaxPool2D(pool_size=pool_size, strides=strides))
            net.add(gluon.nn.Flatten())
            net.add(gluon.nn.Dense(hidden_dim, activation=activation))
            net.add(gluon.nn.Dense(output_dim))
        return net

    # Function to evaluate accuracy for a model
    def evaluate(model, data_iter, context):
        data_iter.reset()
        metric = mx.metric.Accuracy()
        for _, batch in enumerate(data_iter):
            data = batch.data[0].as_in_context(context)
            label = batch.label[0].as_in_context(context)
            output = model(data.astype(args.dtype, copy=False))
            metric.update([label], [output])

        return metric.get()

    # Initialize Horovod
    hvd.init()

    # Horovod: pin context to local rank
    context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())
    num_workers = hvd.size()

    # Load training and validation data
    train_data, val_data = get_mnist_iterator(hvd.rank())

    # Build model
    model = conv_nets()
    model.cast(args.dtype)
    model.hybridize()

    # Create optimizer
    optimizer_params = {"momentum": args.momentum, "learning_rate": args.lr * hvd.size()}
    opt = mx.optimizer.create("sgd", **optimizer_params)

    # Initialize parameters
    initializer = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)
    model.initialize(initializer, ctx=context)

    # Horovod: fetch and broadcast parameters
    params = model.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    # Horovod: create DistributedTrainer, a subclass of gluon.Trainer
    trainer = hvd.DistributedTrainer(params, opt)

    # Create loss function and train metric
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()

    # Global training timing
    if hvd.rank() == 0:
        global_tic = time.time()

    # Train model
    for epoch in range(args.epochs):
        tic = time.time()
        train_data.reset()
        metric.reset()
        for nbatch, batch in enumerate(train_data, start=1):
            data = batch.data[0].as_in_context(context)
            label = batch.label[0].as_in_context(context)
            with autograd.record():
                output = model(data.astype(args.dtype, copy=False))
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(args.batch_size)
            metric.update([label], [output])

            if nbatch % 100 == 0:
                name, acc = metric.get()
                logging.info("[Epoch %d Batch %d] Training: %s=%f" % (epoch, nbatch, name, acc))

        if hvd.rank() == 0:
            elapsed = time.time() - tic
            speed = nbatch * args.batch_size * hvd.size() / elapsed
            logging.info("Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f", epoch, speed, elapsed)

        # Evaluate model accuracy
        _, train_acc = metric.get()
        name, val_acc = evaluate(model, val_data, context)
        if hvd.rank() == 0:
            logging.info(
                "Epoch[%d]\tTrain: %s=%f\tValidation: %s=%f", epoch, name, train_acc, name, val_acc
            )

        if hvd.rank() == 0 and epoch == args.epochs - 1:
            assert val_acc > 0.96, (
                "Achieved accuracy (%f) is lower than expected\
                                    (0.96)"
                % val_acc
            )

    if hvd.rank() == 0:
        global_training_time = time.time() - global_tic
        print("Global elpased time on training:{}".format(global_training_time))
        device = context.device_type + str(num_workers)
        logging.info("Device info: %s", device)


def parse_args():
    # Handling script arguments
    parser = argparse.ArgumentParser(description="MXNet MNIST Distributed Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="training batch size (default: 64)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", help="training data type (default: float32)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of training epochs (default: 5)"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument(
        "--no-cuda", type=bool, default=False, help="disable training on GPU (default: False)"
    )

    # Container Environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    if not args.no_cuda:
        # Disable CUDA if there are no GPUs.
        if mx.context.num_gpus() == 0:
            args.no_cuda = True
    return args


if __name__ == "__main__":

    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)
    main(args)
