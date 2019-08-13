from __future__ import print_function

import json
import logging
import os
import time

import mxnet as mx
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models


# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def train(current_host, hosts, num_cpus, num_gpus, channel_input_dirs, model_dir, hyperparameters, **kwargs):
    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 128)
    epochs = hyperparameters.get('epochs', 100)
    learning_rate = hyperparameters.get('learning_rate', 0.1)
    momentum = hyperparameters.get('momentum', 0.9)
    log_interval = hyperparameters.get('log_interval', 1)
    wd = hyperparameters.get('wd', 0.0001)

    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync'

    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    net = models.get_model('resnet34_v2', ctx=ctx, pretrained=False, classes=10)
    batch_size *= max(1, len(ctx))

    # load training and validation data
    # we use the gluon.data.vision.CIFAR10 class because of its built in pre-processing logic,
    # but point it at the location where SageMaker placed the data files, so it doesn't download them again.

    part_index = 0
    for i, host in enumerate(hosts):
        if host == current_host:
            part_index = i
            break


    data_dir = channel_input_dirs['training']
    CHECKPOINTS_DIR = '/opt/ml/checkpoints'
    checkpoints_enabled = os.path.exists(CHECKPOINTS_DIR)

    train_data = get_train_data(num_cpus, data_dir, batch_size, (3, 32, 32),
                                num_parts=len(hosts), part_index=part_index)
    test_data = get_test_data(num_cpus, data_dir, batch_size, (3, 32, 32))

    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    # Trainer is for updating parameters with gradient.
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            optimizer_params={'learning_rate': learning_rate, 'momentum': momentum, 'wd': wd},
                            kvstore=kvstore)
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    net.hybridize()

    best_accuracy = 0.0
    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        train_data.reset()
        tic = time.time()
        metric.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if i % log_interval == 0 and i > 0:
                name, acc = metric.get()
                logging.info('Epoch [%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f' %
                             (epoch, i, batch_size / (time.time() - btic), name, acc))
            btic = time.time()

        name, acc = metric.get()
        logging.info('[Epoch %d] training: %s=%f' % (epoch, name, acc))
        logging.info('[Epoch %d] time cost: %f' % (epoch, time.time() - tic))

        name, val_acc = test(ctx, net, test_data)
        logging.info('[Epoch %d] validation: %s=%f' % (epoch, name, val_acc))

        # only save params on primary host
        if checkpoints_enabled and current_host == hosts[0]:
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                logging.info('Saving the model, params and optimizer state')
                net.export(CHECKPOINTS_DIR + "/%.4f-cifar10"%(best_accuracy), epoch)
                trainer.save_states(CHECKPOINTS_DIR + '/%.4f-cifar10-%d.states'%(best_accuracy, epoch))

    return net


def save(net, model_dir):
    # model_dir will be empty except on primary container
    files = os.listdir(model_dir)
    if files:
        best = sorted(os.listdir(model_dir))[-1]
        os.rename(os.path.join(model_dir, best), os.path.join(model_dir, 'model.params'))


def get_data(path, augment, num_cpus, batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    return mx.io.ImageRecordIter(
        path_imgrec=path,
        resize=resize,
        data_shape=data_shape,
        batch_size=batch_size,
        rand_crop=augment,
        rand_mirror=augment,
        preprocess_threads=num_cpus,
        num_parts=num_parts,
        part_index=part_index)


def get_test_data(num_cpus, data_dir, batch_size, data_shape, resize=-1):
    return get_data(os.path.join(data_dir, "test.rec"), False, num_cpus, batch_size, data_shape, resize, 1, 0)


def get_train_data(num_cpus, data_dir, batch_size, data_shape, resize=-1, num_parts=1, part_index=0):
    return get_data(os.path.join(data_dir, "train.rec"), True, num_cpus, batch_size, data_shape, resize, num_parts,
                    part_index)


def test(ctx, net, test_data):
    test_data.reset()
    metric = mx.metric.Accuracy()

    for i, batch in enumerate(test_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """

    net = models.get_model('resnet34_v2', ctx=mx.cpu(), pretrained=False, classes=10)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    parsed = json.loads(data)
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type
