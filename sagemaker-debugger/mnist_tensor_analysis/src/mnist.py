# Standard Library
import argparse

# Third Party
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Train a mxnet gluon model")
    parser.add_argument(
        "--output-s3-uri",
        type=str,
        default=None,
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument(
        "--smdebug_path",
        type=str,
        default=None,
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument("--initializer", type=int, default=1, help="Variable to change intializer")
    parser.add_argument("--lr", type=float, default=0.001, help="Variable to change learning rate")
    opt = parser.parse_args()
    return opt


def create_gluon_model(initializer):
    net = nn.HybridSequential()
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, activation="relu"),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation="relu"),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10),
    )
    if initializer == 1:
        net.initialize(init=init.Xavier(), ctx=mx.cpu())
    elif initializer == 2:
        # variance will not remain the same across layers
        net.initialize(init=init.Uniform(1), ctx=mx.cpu())
    else:
        # does not break symmetry,so gradients will not differ much
        net.initialize(init=init.Uniform(0.0001), ctx=mx.cpu())
    return net


def train_model(batch_size, net, train_data, lr):
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr})
    for epoch in range(3):
        for data, label in train_data:
            data = data.as_in_context(mx.cpu(0))
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
        print(np.mean(loss.asnumpy()))

def prepare_data(batch_size):
    mnist_train = datasets.FashionMNIST(train=True)
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.286, 0.352)])
    mnist_train = mnist_train.transform_first(transformer)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return train_data


def main():
    opt = parse_args()
    net = create_gluon_model(opt.initializer)
    train_data = prepare_data(128)
    train_model(128, net, train_data, opt.lr)


if __name__ == "__main__":
    main()

