# Standard Library
import argparse
import random

# Third Party
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.gluon import nn
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a mxnet gluon model for FashonMNIST dataset"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of Epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument(
        "--context", type=str, default="cpu", help="Context can be either cpu or gpu"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/opt/ml/checkpoints",
        help="Path where checkpoints will be saved.",
    )

    opt = parser.parse_args()
    return opt


def test(ctx, net, val_data):
    metric = mx.metric.Accuracy()
    for i, (data, label) in enumerate(val_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])

    return metric.get()


def train_model(net, epochs, ctx, learning_rate, momentum, train_data, val_data, checkpoint_path):
    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # Trainer is for updating parameters with gradient.
    trainer = gluon.Trainer(
        net.collect_params(), "sgd", {"learning_rate": learning_rate, "momentum": momentum}
    )
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        metric.reset()
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

            if i % 100 == 0 and i > 0:
                name, acc = metric.get()
                print("[Epoch %d Batch %d] Training: %s=%f" % (epoch, i, name, acc))

        name, acc = metric.get()
        print("[Epoch %d] Training: %s=%f" % (epoch, name, acc))
        name, val_acc = test(ctx, net, val_data)
        print("[Epoch %d] Validation: %s=%f" % (epoch, name, val_acc))
        param_file = "{0}/params_{1}.params".format(checkpoint_path, epoch)
        print ("Saving params to:  " + param_file) 
        net.save_parameters(param_file)


def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label


def prepare_data(batch_size):
    train_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST("./data", train=True, transform=transformer),
        batch_size=batch_size,
        shuffle=True,
        last_batch="discard",
    )

    val_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST("./data", train=False, transform=transformer),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_data, val_data


# Create a model using gluon API. The hook is currently
# supports MXNet gluon models only.
def create_gluon_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(64, activation="relu"))
        net.add(nn.Dense(10))
    return net


def validate():
    import os, json
    with open('/opt/ml/input/config/debughookconfig.json') as jsondata:
        configs = json.load(jsondata)
        print("DEBUG HOOK CONFIGURATION: ")
        print(json.dumps(configs, indent=4))
    print("Validation Complete")


def main():
    opt = parse_args()
    mx.random.seed(128)
    random.seed(12)
    np.random.seed(2)

    context = mx.cpu() if opt.context.lower() == "cpu" else mx.gpu()
    # Create a Gluon Model.
    net = create_gluon_model()

    # Start the training.
    train_data, val_data = prepare_data(opt.batch_size)

    train_model(
        net=net,
        epochs=opt.epochs,
        ctx=context,
        learning_rate=opt.learning_rate,
        momentum=0.9,
        train_data=train_data,
        val_data=val_data,
        checkpoint_path = opt.checkpoint_path
    )
    validate()


if __name__ == "__main__":
    main()
