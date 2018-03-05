import logging

import gzip
import mxnet as mx
import numpy as np
import os
import struct


def load_data(path):
    with gzip.open(find_file(path, "labels.gz")) as flbl:
        struct.unpack(">II", flbl.read(8))
        labels = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(find_file(path, "images.gz")) as fimg:
        _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(labels), rows, cols)
        images = images.reshape(images.shape[0], 1, 28, 28).astype(np.float32) / 255
    return labels, images


def find_file(root_path, file_name):
    for root, dirs, files in os.walk(root_path):
        if file_name in files:
            return os.path.join(root, file_name)


def build_graph():
    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)
    return mx.sym.SoftmaxOutput(data=fc3, name='softmax')


def train(current_host, channel_input_dirs, hyperparameters, hosts, num_cpus, num_gpus):
    (train_labels, train_images) = load_data(os.path.join(channel_input_dirs['train']))
    (test_labels, test_images) = load_data(os.path.join(channel_input_dirs['test']))

    # Alternatively to splitting in memory, the data could be pre-split in S3 and use ShardedByS3Key
    # to do parallel training.
    shard_size = len(train_images) // len(hosts)
    for i, host in enumerate(hosts):
        if host == current_host:
            start = shard_size * i
            end = start + shard_size
            break

    batch_size = 100
    train_iter = mx.io.NDArrayIter(train_images[start:end], train_labels[start:end], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(test_images, test_labels, batch_size)
    logging.getLogger().setLevel(logging.DEBUG)
    kvstore = 'local' if len(hosts) == 1 else 'dist_sync'
    mlp_model = mx.mod.Module(
        symbol=build_graph(),
        context=get_train_context(num_cpus, num_gpus))
    mlp_model.fit(train_iter,
                  eval_data=val_iter,
                  kvstore=kvstore,
                  optimizer='sgd',
                  optimizer_params={'learning_rate': float(hyperparameters.get("learning_rate", 0.1))},
                  eval_metric='acc',
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  num_epoch=25)
    return mlp_model


def get_train_context(num_cpus, num_gpus):
    if num_gpus > 0:
        return mx.gpu()
    return mx.cpu()
