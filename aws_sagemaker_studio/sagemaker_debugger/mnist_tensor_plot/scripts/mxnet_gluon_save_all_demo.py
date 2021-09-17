# Standard Library
import argparse
import time
import uuid

# Third Party
import mxnet as mx
from mxnet import autograd, gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train a mxnet gluon model for MNIST dataset")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    opt = parser.parse_args()
    return opt


def acc(output, label):
    return (output.argmax(axis=1) == label.astype("float32")).mean().asscalar()


def train_model(batch_size, net, train_data, valid_data):
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})
    # Start the training.
    for epoch in range(1):
        train_loss, train_acc, valid_acc = 0.0, 0.0, 0.0
        tic = time.time()
        for data, label in train_data:
            data = data.as_in_context(mx.cpu(0))
            # forward + backward
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            # update parameters
            trainer.step(batch_size)
            # calculate training metrics
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)
        # calculate validation accuracy
        for data, label in valid_data:
            data = data.as_in_context(mx.cpu(0))
            valid_acc += acc(net(data), label)
        print(
            "Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec"
            % (
                epoch,
                train_loss / len(train_data),
                train_acc / len(train_data),
                valid_acc / len(valid_data),
                time.time() - tic,
            )
        )


def prepare_data(batch_size):
    mnist_train = datasets.MNIST(train=True)
    X, y = mnist_train[0]
    ("X shape: ", X.shape, "X dtype", X.dtype, "y:", y)
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    X, y = mnist_train[0:10]
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])
    mnist_train = mnist_train.transform_first(transformer)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    mnist_valid = gluon.data.vision.FashionMNIST(train=False)
    valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer), batch_size=batch_size, num_workers=4
    )
    return train_data, valid_data


# Create a model using gluon API. Note: debugger hook currently
# supports MXNet gluon models only.
def create_gluon_model():
    # Create Model in Gluon
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
    net.initialize(init=init.Xavier(), ctx=mx.cpu())
    return net


def main():
    opt = parse_args()
    # Create a Gluon Model.
    net = create_gluon_model()

    # Start the training.
    batch_size = opt.batch_size
    train_data, valid_data = prepare_data(batch_size)

    train_model(batch_size, net, train_data, valid_data)


if __name__ == "__main__":
    main()
