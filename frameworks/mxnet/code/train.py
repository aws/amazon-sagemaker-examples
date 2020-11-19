from __future__ import print_function

import argparse
import gzip
import logging
import os

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data import Dataset
import numpy as np
import json
import time


logging.basicConfig(level=logging.DEBUG)

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def input_transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255.
    return data, label


class MNIST(Dataset):
    def __init__(self, data_dir, train=True, transform=None):

        if train:
            images_file="train-images-idx3-ubyte.gz"
            labels_file="train-labels-idx1-ubyte.gz"
        else:
            images_files="t10k-images-idx3-ubyte.gz"
            labels_files="t10k-labels-idx1-ubyte.gz"
        
        self.images, self.labels = self._convert_to_numpy(
                data_dir, images_file, labels_file)

        def _id(x, y):
            return x, y
        self.transform = transform or _id 

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        return self.transform(img, label)

    def _convert_to_numpy(self, data_dir, images_file, labels_file):
        """Byte string to numpy arrays 
        """
        with gzip.open(os.path.join(data_dir, images_file), 'rb') as f:
            images = np.frombuffer(f.read(), 
                    np.uint8, offset=16).reshape(-1, 28, 28)

        with gzip.open(os.path.join(data_dir, labels_file), 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return (images, labels)
    
    
    
    
def get_data_loader(data_dir, batch_size, train=True):
    mnist = MNIST(data_dir, train, input_transformer)
    return gluon.data.DataLoader(mnist, batch_size=batch_size,
            shuffle=True, last_batch='rollover')



def train(args):
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.
    ctx = mx.cpu()

    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    log_interval = args.log_interval

    num_gpus = int(os.environ['SM_NUM_GPUS'])
    current_host = args.current_host
    hosts = args.hosts
    model_dir = args.model_dir
    CHECKPOINTS_DIR = '/opt/ml/checkpoints'
    checkpoints_enabled = os.path.exists(CHECKPOINTS_DIR)

    # load training and validation data
    # we use the gluon.data.vision.MNIST class because of its built in mnist pre-processing logic,
    # but point it at the location where SageMaker placed the data files, so it doesn't download them again.
   
    train_data = get_data_loader(args.train, batch_size)
    val_data = get_data_loader(args.test, batch_size)

    # define the network
    net = define_network()

    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # Trainer is for updating parameters with gradient.

    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': learning_rate, 'momentum': momentum},
                            kvstore=kvstore)
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # shard the training data in case we are doing distributed training. Alternatively to splitting in memory,
    # the data could be pre-split in S3 and use ShardedByS3Key to do distributed training.
    if len(hosts) > 1:
        train_data = [x for x in train_data]
        shard_size = len(train_data) // len(hosts)
        for i, host in enumerate(hosts):
            if host == current_host:
                start = shard_size * i
                end = start + shard_size
                break

        train_data = train_data[start:end]

    net.hybridize()

    best_val_score = 0.0
    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        metric.reset()
        btic = time.time()
        for i, (data, label) in enumerate(train_data):
            # Copy data to ctx if necessary
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])
            # update metric at last.
            metric.update([label], [output])

            if i % log_interval == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f, %f samples/s' %
                      (epoch, i, name, acc, batch_size / (time.time() - btic)))

            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] Training: %s=%f' % (epoch, name, acc))

        name, val_acc = test(ctx, net, val_data)
        print('[Epoch %d] Validation: %s=%f' % (epoch, name, val_acc))
        # checkpoint the model, params and optimizer states in the folder /opt/ml/checkpoints
        if checkpoints_enabled and val_acc > best_val_score:
            best_val_score = val_acc
            logging.info('Saving the model, params and optimizer state.')
            net.export(CHECKPOINTS_DIR + "/%.4f-gluon_mnist"%(best_val_score), epoch)
            trainer.save_states(CHECKPOINTS_DIR + '/%.4f-gluon_mnist-%d.states'%(best_val_score, epoch))

    if current_host == hosts[0]:
        save(net, model_dir)


def save(net, model_dir):
    # save the model
    net.export('%s/model'% model_dir)


def define_network():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))
    return net



def test(ctx, net, val_data):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])
    return metric.get()



# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--log-interval', type=float, default=100)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'])

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
