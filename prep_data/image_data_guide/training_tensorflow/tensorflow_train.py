import argparse
import os
import pathlib

import tensorflow as tf
import tensorflow_datasets as tfds


def tfrecord_parser(record):
    features = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(record, features)
    return tf.io.decode_jpeg(parsed_features["image_raw"]), parsed_features["label"]


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_hue(image, 0.1)
    return (image, label)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=50)
    arg_parser.add_argument("--batch-size", type=int, default=4)
    arg_parser.add_argument("--learning-rate", type=float, default=0.001)

    arg_parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    arg_parser.add_argument(
        "--validation-dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    args, _ = arg_parser.parse_known_args()

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_data = pathlib.Path(args.train_dir) / "train.tfrecord"
    val_data = pathlib.Path(args.validation_dir) / "val.tfrecord"

    train_ds = tf.data.TFRecordDataset(
        filenames=[train_data.as_posix()], num_parallel_reads=AUTOTUNE
    )

    val_ds = tf.data.TFRecordDataset(filenames=[val_data.as_posix()], num_parallel_reads=AUTOTUNE)

    train_ds = (
        train_ds.map(tfrecord_parser, num_parallel_calls=AUTOTUNE)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds.map(tfrecord_parser, num_parallel_calls=AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    if any(gpu_devices):
        device = gpu_devices[0].device_type
    else:
        device = "/cpu:0"
    print(f"Training with: {device}")

    with tf.device(device):

        base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")

        global_avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(11, activation="softmax")(global_avg)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

        optimizer = tf.keras.optimizers.SGD(lr=args.learning_rate, momentum=0.9, decay=0.01)

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        print("Beginning Training...")
        model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, verbose=2)

        model.save("/opt/ml/model/model")
