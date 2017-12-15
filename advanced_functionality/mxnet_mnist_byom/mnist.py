import logging

import gzip
import mxnet as mx
import numpy as np
import os
import struct


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


def train(data, hyperparameters= {'learning_rate': 0.11}, num_cpus=0, num_gpus =1 , **kwargs):
    train_labels = data['train_label']
    train_images = data['train_data']
    test_labels = data['test_label']
    test_images = data['test_data']
    batch_size = 100
    train_iter = mx.io.NDArrayIter(train_images, train_labels, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(test_images, test_labels, batch_size)
    logging.getLogger().setLevel(logging.DEBUG)
    mlp_model = mx.mod.Module(
        symbol=build_graph(),
        context=get_train_context(num_cpus, num_gpus))
    mlp_model.fit(train_iter,
                  eval_data=val_iter,
                  optimizer='sgd',
                  optimizer_params={'learning_rate': float(hyperparameters.get("learning_rate", 0.1))},
                  eval_metric='acc',
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  num_epoch=10)
    return mlp_model


def get_train_context(num_cpus, num_gpus):
    if num_gpus > 0:
        return mx.gpu()
    return mx.cpu()