# Third Party
import horovod.tensorflow.keras as hvd
import tensorflow.compat.v2 as tf


def get_data(batch_size):
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(
        path="mnist-%d.npz" % hvd.rank()
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
            tf.cast(mnist_labels, tf.int64),
        )
    )
    dataset = dataset.repeat().shuffle(10000).batch(batch_size)
    return dataset


def get_model():
    mnist_model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, [3, 3], activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return mnist_model


def train(model, dataset, epoch, initial_lr):
    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(initial_lr, warmup_epochs=3, verbose=1),
    ]

    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint("checkpoint-{epoch}.h5"))

    verbose = 1 if hvd.rank() == 0 else 0

    model.fit(
        dataset,
        steps_per_epoch=500 // hvd.size(),
        callbacks=callbacks,
        epochs=epoch,
        verbose=verbose,
    )


if __name__ == "__main__":
    lr = 0.001
    batch_size = 64
    num_epochs = 200

    hvd.init()

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

    dataset = get_data(batch_size)
    mnist_model = get_model()

    train(model=mnist_model, dataset=dataset, epoch=num_epochs, initial_lr=lr)
