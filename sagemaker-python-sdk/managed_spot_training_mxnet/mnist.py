import argparse
import gzip
import json
import logging
import os
import struct

import mxnet as mx
import numpy as np


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


def get_training_context(num_gpus):
    if num_gpus:
        return [mx.gpu(i) for i in range(num_gpus)]
    else:
        return mx.cpu()


def train(batch_size, epochs, learning_rate, num_gpus, training_channel, testing_channel,
          hosts, current_host, model_dir):
    (train_labels, train_images) = load_data(training_channel)
    (test_labels, test_images) = load_data(testing_channel)
    CHECKPOINTS_DIR = '/opt/ml/checkpoints'
    checkpoints_enabled = os.path.exists(CHECKPOINTS_DIR)

    # Data parallel training - shard the data so each host
    # only trains on a subset of the total data.
    shard_size = len(train_images) // len(hosts)
    for i, host in enumerate(hosts):
        if host == current_host:
            start = shard_size * i
            end = start + shard_size
            break

    train_iter = mx.io.NDArrayIter(train_images[start:end], train_labels[start:end], batch_size,
                                   shuffle=True)
    val_iter = mx.io.NDArrayIter(test_images, test_labels, batch_size)

    logging.getLogger().setLevel(logging.DEBUG)

    kvstore = 'local' if len(hosts) == 1 else 'dist_sync'

    mlp_model = mx.mod.Module(symbol=build_graph(),
                              context=get_training_context(num_gpus))

    checkpoint_callback = None
    if checkpoints_enabled:
        # Create a checkpoint callback that checkpoints the model params and the optimizer state after every epoch at the given path.
        checkpoint_callback = mx.callback.module_checkpoint(mlp_model,
                                                            CHECKPOINTS_DIR + "/mnist",
                                                            period=1,
                                                            save_optimizer_states=True)
    mlp_model.fit(train_iter,
                  eval_data=val_iter,
                  kvstore=kvstore,
                  optimizer='sgd',
                  optimizer_params={'learning_rate': learning_rate},
                  eval_metric='acc',
                  epoch_end_callback = checkpoint_callback,
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  num_epoch=epochs)

    if current_host == hosts[0]:
        save(model_dir, mlp_model)


def save(model_dir, model):
    model.symbol.save(os.path.join(model_dir, 'model-symbol.json'))
    model.save_params(os.path.join(model_dir, 'model-0000.params'))

    signature = [{'name': data_desc.name, 'shape': [dim for dim in data_desc.shape]}
                 for data_desc in model.data_shapes]
    with open(os.path.join(model_dir, 'model-shapes.json'), 'w') as f:
        json.dump(signature, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.1)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()

### NOTE: this function cannot use MXNet
def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io

    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'application/vnd+python.numpy+binary':
        raise RuntimeError('Content type must be application/vnd+python.numpy+binary')

    f = io.BytesIO(payload)
    return np.load(f)

### NOTE: this function cannot use MXNet
def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    logging.info('Invoking user-defined post-processing function')

    # Softmax (assumes batch size 1)
    result = np.squeeze(result)
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())
    content_type = 'application/json'

    return response_body, content_type


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ['SM_NUM_GPUS'])

    train(args.batch_size, args.epochs, args.learning_rate, num_gpus, args.train, args.test,
          args.hosts, args.current_host, args.model_dir)
