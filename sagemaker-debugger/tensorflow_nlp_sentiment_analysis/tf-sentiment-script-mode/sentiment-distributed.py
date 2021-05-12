import argparse
import os

import horovod.tensorflow.keras as hvd
import numpy as np
import tensorflow.compat.v2 as tf

max_features = 20000
maxlen = 400
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)

    # data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


def get_train_data(train_dir):

    x_train = np.load(os.path.join(train_dir, "x_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    print("x train", x_train.shape, "y train", y_train.shape)

    return x_train, y_train


def get_test_data(test_dir):

    x_test = np.load(os.path.join(test_dir, "x_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))
    print("x test", x_test.shape, "y test", y_test.shape)

    return x_test, y_test


def get_model():

    embedding_layer = tf.keras.layers.Embedding(max_features, embedding_dims, input_length=maxlen)

    sequence_input = tf.keras.Input(shape=(maxlen,), dtype="int32")
    embedded_sequences = embedding_layer(sequence_input)
    x = tf.keras.layers.Dropout(0.2)(embedded_sequences)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="valid", activation="relu", strides=1)(
        x
    )
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(hidden_dims, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    preds = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(sequence_input, preds)


if __name__ == "__main__":

    args, _ = parse_args()

    hvd.init()
    lr = 0.001
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(lr * hvd.size())

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt)

    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)

    model = get_model()

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
    ]

    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint("checkpoint-{epoch}.h5"))

    verbose = 1 if hvd.rank() == 0 else 0

    # hook = KerasHook(out_dir='/tmp/test')
    model.fit(
        x_train,
        y_train,
        steps_per_epoch=500 // hvd.size(),
        callbacks=callbacks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
    )

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.saved_model.save(model, args.model_dir)
