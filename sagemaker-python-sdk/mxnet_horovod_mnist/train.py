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
import gzip
import json
import os
import time
import numpy as np

import mxnet as mx
from mxnet.gluon.data import Dataset, DataLoader
import horovod.mxnet as hvd
from mxnet import autograd, gluon, nd

logging.basicConfig(level=logging.DEBUG)
logging.info('So should it be')

def normalize(x, axis):
    eps = np.finfo(float).eps
    mean = np.mean(x, axis=axis, keepdims=True)
    # avoid division by zero
    std = np.std(x, axis=axis, keepdims=True) + eps
    return (x - mean) / std

def input_transformer(data, label):
    print('Data shape', data.shape)
    # normalize the pixels
    data = normalize(data.astype(np.float32), axis=(1, 2))
    
    # add channel dim
    data = np.expand_dims(data, axis=1)
    return data, label

# MNIST data set
class MNIST(Dataset):
    def __init__(self, data_dir, train=True, transform=None):

        if train:
            images_file="train-images-idx3-ubyte.gz"
            labels_file="train-labels-idx1-ubyte.gz"
        else:
            images_file="t10k-images-idx3-ubyte.gz"
            labels_file="t10k-labels-idx1-ubyte.gz"
        
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

            # normalize 
            images = normalize(images.astype(np.float32), axis=(1, 2))
            
            # add channel dim
            images = np.expand_dims(images, axis=1)

        with gzip.open(os.path.join(data_dir, labels_file), 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return (images, labels)

    
# model
    # Function to define neural network
def conv_nets():
    kernel_size = 5
    strides = 2
    pool_size = 2
    hidden_dim = 512
    output_dim = 10
    activation = 'relu'
    
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

  
def train(args):
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize Horovod
    hvd.init()

    # Horovod: pin context to local rank
    context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())
    num_workers = hvd.size()
    
    train_set = MNIST(args.data_dir, train=True)
    test_set = MNIST(args.data_dir, train=False)

    train_iter = DataLoader(train_set, batch_size=args.batch_size, 
        shuffle=True, last_batch='rollover')
    
    test_iter = DataLoader(test_set, batch_size=args.batch_size,
        shuffle=False, last_batch='rollover')


    # Build model
    model = conv_nets()
    model.cast(args.dtype)
    model.hybridize()

    # Create optimizer
    optimizer_params = {'momentum': args.momentum,
                        'learning_rate': args.lr * hvd.size()}

    opt = mx.optimizer.create('sgd', **optimizer_params)

    # Initialize parameters
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                                 magnitude=2)
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
    best_val_acc = 0.0
    print("Start training ...")
    for epoch in range(args.epochs):
        tic = time.time()
        metric.reset()
        for nbatch, (data, label) in enumerate(train_iter, start=1):
            data = data.as_in_context(context)
            label = label.as_in_context(context)
            with autograd.record():
                output = model(data.astype(args.dtype, copy=False))
                loss = loss_fn(output, label)
                loss.backward()

            trainer.step(args.batch_size)
            metric.update([label], [output])

            if nbatch % 100 == 0:
                name, acc = metric.get()
                logging.info('[Epoch %d Batch %d] Training: %s=%f' %
                             (epoch, nbatch, name, acc))

        if hvd.rank() == 0:
            elapsed = time.time() - tic
            speed = nbatch * args.batch_size * hvd.size() / elapsed
            logging.info('Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f',
                         epoch, speed, elapsed)

        # Evaluate model accuracy
        _, train_acc = metric.get()
        name, val_acc = evaluate(model, test_iter, context)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if hvd.rank() == 0:
                model.export(model_dir + '/model')

        if hvd.rank() == 0:
            logging.info('Epoch[%d]\tTrain: %s=%f\tValidation: %s=%f', 
            epoch, name,train_acc, name, val_acc)



    if hvd.rank()==0: 
        global_training_time =time.time() - global_tic 
        print("Global elpased time on training:{}".format(global_training_time))
        device = context.device_type + str(num_workers)
        logging.info('Device info: %s', device)
    return


# Function to evaluate accuracy for a model
def evaluate(model, data_iter, context):
    metric = mx.metric.Accuracy()
    for _, (data, label) in enumerate(data_iter):
        data = data.as_in_context(context)
        label = label.as_in_context(context)
        output = model(data)
        metric.update([label], [output])

    return metric.get()

def parse_args():
    # Handling script arguments
    parser = argparse.ArgumentParser(description='MXNet MNIST Distributed Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size (default: 64)')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='training data type (default: float32)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', help='disable training on GPU (default: False)')
    
    # Container Environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    args = parser.parse_args()

    if not args.no_cuda:
        # Disable CUDA if there are no GPUs.
        if mx.context.num_gpus() == 0:
            args.no_cuda = True
    return args 

if __name__ == "__main__":
    
    args = parse_args()
    # logging.info(args)

    train(args)
