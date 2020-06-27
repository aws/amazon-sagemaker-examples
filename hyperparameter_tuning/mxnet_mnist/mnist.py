import argparse
import gzip
import json
import logging
import os
import struct

import mxnet as mx
import numpy as np

from sagemaker_mxnet_container.training_utils import scheduler_host


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


def train(batch_size, num_epoch, learning_rate, optimizer, training_channel,
          testing_channel, hosts, current_host, model_dir):
    (train_labels, train_images) = load_data(training_channel)
    (test_labels, test_images) = load_data(testing_channel)

    # Alternatively to splitting in memory, the data could be pre-split in S3 and use ShardedByS3Key
    # to do parallel training.
    shard_size = len(train_images) // len(hosts)
    for i, host in enumerate(hosts):
        if host == current_host:
            start = shard_size * i
            end = start + shard_size
            break

    train_iter = mx.io.NDArrayIter(train_images[start:end], train_labels[start:end], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(test_images, test_labels, batch_size)
    logging.getLogger().setLevel(logging.DEBUG)
    kvstore = 'local' if len(hosts) == 1 else 'dist_sync'
    mlp_model = mx.mod.Module(
        symbol=build_graph(),
        context=get_train_context())
    mlp_model.fit(train_iter,
                  eval_data=val_iter,
                  kvstore=kvstore,
                  optimizer=optimizer,
                  optimizer_params={'learning_rate': learning_rate},
                  eval_metric='acc',
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  num_epoch=num_epoch)
    return mlp_model

    if current_host == scheduler_host(hosts):
        save(model_dir, mlp_model)


def get_train_context():
    if int(os.environ['SM_NUM_GPUS']) > 0:
        return mx.gpu()
    return mx.cpu()


def save(model_dir, model):
    model.symbol.save(os.path.join(model_dir, 'model-symbol.json'))
    model.save_params(os.path.join(model_dir, 'model-0000.params'))

    signature = [{'name': data_desc.name, 'shape': [dim for dim in data_desc.shape]}
                 for data_desc in model.data_shapes]
    with open(os.path.join(model_dir, 'model-shapes.json'), 'w') as f:
        json.dump(signature, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--optimizer', default='sgd')

    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train(args.batch_size, args.num_epoch, args.learning_rate, args.optimizer,
          args.train, args.test, args.hosts, args.current_host, args.model_dir)
