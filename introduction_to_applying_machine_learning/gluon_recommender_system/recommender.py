import json
import logging
import os
import time

import mxnet as mx
import numpy as np
from mxnet import gluon, nd, ndarray
from mxnet.metric import MSE

os.system("pip install pandas")
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

#########
# Globals
#########

batch_size = 1024


##########
# Training
##########


def train(channel_input_dirs, hyperparameters, hosts, num_gpus, **kwargs):

    # get data
    training_dir = channel_input_dirs["train"]
    train_iter, test_iter, customer_index, product_index = prepare_train_data(training_dir)

    # get hyperparameters
    num_embeddings = hyperparameters.get("num_embeddings", 64)
    opt = hyperparameters.get("opt", "sgd")
    lr = hyperparameters.get("lr", 0.02)
    momentum = hyperparameters.get("momentum", 0.9)
    wd = hyperparameters.get("wd", 0.0)
    epochs = hyperparameters.get("epochs", 5)

    # define net
    ctx = mx.gpu()

    net = MFBlock(
        max_users=customer_index.shape[0],
        max_items=product_index.shape[0],
        num_emb=num_embeddings,
        dropout_p=0.5,
    )

    net.collect_params().initialize(mx.init.Xavier(magnitude=60), ctx=ctx, force_reinit=True)
    net.hybridize()

    trainer = gluon.Trainer(
        net.collect_params(), opt, {"learning_rate": lr, "wd": wd, "momentum": momentum}
    )

    # execute
    trained_net = execute(train_iter, test_iter, net, trainer, epochs, ctx)

    return trained_net, customer_index, product_index


class MFBlock(gluon.HybridBlock):
    def __init__(self, max_users, max_items, num_emb, dropout_p=0.5):
        super(MFBlock, self).__init__()

        self.max_users = max_users
        self.max_items = max_items
        self.dropout_p = dropout_p
        self.num_emb = num_emb

        with self.name_scope():
            self.user_embeddings = gluon.nn.Embedding(max_users, num_emb)
            self.item_embeddings = gluon.nn.Embedding(max_items, num_emb)

            self.dropout_user = gluon.nn.Dropout(dropout_p)
            self.dropout_item = gluon.nn.Dropout(dropout_p)

            self.dense_user = gluon.nn.Dense(num_emb, activation="relu")
            self.dense_item = gluon.nn.Dense(num_emb, activation="relu")

    def hybrid_forward(self, F, users, items):
        a = self.user_embeddings(users)
        a = self.dense_user(a)

        b = self.item_embeddings(items)
        b = self.dense_item(b)

        predictions = self.dropout_user(a) * self.dropout_item(b)
        predictions = F.sum(predictions, axis=1)

        return predictions


def execute(train_iter, test_iter, net, trainer, epochs, ctx):
    loss_function = gluon.loss.L2Loss()
    for e in range(epochs):
        print("epoch: {}".format(e))
        for i, (user, item, label) in enumerate(train_iter):

            user = user.as_in_context(ctx)
            item = item.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with mx.autograd.record():
                output = net(user, item)
                loss = loss_function(output, label)
            loss.backward()
            trainer.step(batch_size)

        print(
            "EPOCH {}: MSE ON TRAINING and TEST: {}. {}".format(
                e,
                eval_net(train_iter, net, ctx, loss_function),
                eval_net(test_iter, net, ctx, loss_function),
            )
        )
    print("end of training")
    return net


def eval_net(data, net, ctx, loss_function):
    acc = MSE()
    for i, (user, item, label) in enumerate(data):

        user = user.as_in_context(ctx)
        item = item.as_in_context(ctx)
        label = label.as_in_context(ctx)

        predictions = net(user, item).reshape((batch_size, 1))
        acc.update(preds=[predictions], labels=[label])

    return acc.get()[1]


def save(model, model_dir):
    net, customer_index, product_index = model
    net.save_params("{}/model.params".format(model_dir))
    f = open("{}/MFBlock.params".format(model_dir), "w")
    json.dump(
        {
            "max_users": net.max_users,
            "max_items": net.max_items,
            "num_emb": net.num_emb,
            "dropout_p": net.dropout_p,
        },
        f,
    )
    f.close()
    customer_index.to_csv("{}/customer_index.csv".format(model_dir), index=False)
    product_index.to_csv("{}/product_index.csv".format(model_dir), index=False)


######
# Data
######


def prepare_train_data(training_dir):
    f = os.listdir(training_dir)
    df = pd.read_csv(os.path.join(training_dir, f[0]), delimiter="\t", error_bad_lines=False)
    df = df[["customer_id", "product_id", "star_rating"]]
    customers = df["customer_id"].value_counts()
    products = df["product_id"].value_counts()

    # Filter long-tail
    customers = customers[customers >= 5]
    products = products[products >= 10]

    reduced_df = df.merge(pd.DataFrame({"customer_id": customers.index})).merge(
        pd.DataFrame({"product_id": products.index})
    )
    customers = reduced_df["customer_id"].value_counts()
    products = reduced_df["product_id"].value_counts()

    # Number users and items
    customer_index = pd.DataFrame(
        {"customer_id": customers.index, "user": np.arange(customers.shape[0])}
    )
    product_index = pd.DataFrame(
        {"product_id": products.index, "item": np.arange(products.shape[0])}
    )

    reduced_df = reduced_df.merge(customer_index).merge(product_index)

    # Split train and test
    test_df = reduced_df.groupby("customer_id").last().reset_index()

    train_df = reduced_df.merge(
        test_df[["customer_id", "product_id"]],
        on=["customer_id", "product_id"],
        how="outer",
        indicator=True,
    )
    train_df = train_df[(train_df["_merge"] == "left_only")]

    # MXNet data iterators
    train = gluon.data.ArrayDataset(
        nd.array(train_df["user"].values, dtype=np.float32),
        nd.array(train_df["item"].values, dtype=np.float32),
        nd.array(train_df["star_rating"].values, dtype=np.float32),
    )
    test = gluon.data.ArrayDataset(
        nd.array(test_df["user"].values, dtype=np.float32),
        nd.array(test_df["item"].values, dtype=np.float32),
        nd.array(test_df["star_rating"].values, dtype=np.float32),
    )

    train_iter = gluon.data.DataLoader(
        train, shuffle=True, num_workers=4, batch_size=batch_size, last_batch="rollover"
    )
    test_iter = gluon.data.DataLoader(
        train, shuffle=True, num_workers=4, batch_size=batch_size, last_batch="rollover"
    )

    return train_iter, test_iter, customer_index, product_index


#########
# Hosting
#########


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    ctx = mx.cpu()
    f = open("{}/MFBlock.params".format(model_dir), "r")
    block_params = json.load(f)
    f.close()
    net = MFBlock(
        max_users=block_params["max_users"],
        max_items=block_params["max_items"],
        num_emb=block_params["num_emb"],
        dropout_p=block_params["dropout_p"],
    )
    net.load_params("{}/model.params".format(model_dir), ctx)
    customer_index = pd.read_csv("{}/customer_index.csv".format(model_dir))
    product_index = pd.read_csv("{}/product_index.csv".format(model_dir))
    return net, customer_index, product_index


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    ctx = mx.cpu()
    parsed = json.loads(data)

    trained_net, customer_index, product_index = net
    users = (
        pd.DataFrame({"customer_id": parsed["customer_id"]})
        .merge(customer_index, how="left")["user"]
        .values
    )
    items = (
        pd.DataFrame({"product_id": parsed["product_id"]})
        .merge(product_index, how="left")["item"]
        .values
    )

    predictions = trained_net(
        nd.array(users).as_in_context(ctx), nd.array(items).as_in_context(ctx)
    )
    response_body = json.dumps(predictions.asnumpy().tolist())

    return response_body, output_content_type
