# Standard Library
import argparse
import os

# Third Party
import tensorflow as tf


def data_augmentation(image, label):
    import tensorflow_addons as tfa

    for i in range(20):
        image = tfa.image.gaussian_filter2d(image=image, filter_shape=(11, 11), sigma=0.8)
    return image, label


def parse_image_function(example_proto):

    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_raw(features["image"], tf.uint8)
    image.set_shape([3 * 32 * 32])
    image = tf.reshape(image, [32, 32, 3])
    label = tf.cast(features["label"], tf.int32)

    return image, label


def get_dataset(batch_size, channel_name, dataset_bottleneck=False):
    from sagemaker_tensorflow import PipeModeDataset

    dataset = PipeModeDataset(channel_name, record_format="TFRecord").repeat()

    dataset = dataset.map(parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if dataset_bottleneck:
        dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


def get_model(lr):
    from tensorflow.keras.applications.resnet50 import ResNet50

    model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)

    opt = tf.optimizers.Adam(lr)

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=["accuracy"]
    )

    return model


def train(model, dataset, epoch):
    model.fit(
        dataset,
        steps_per_epoch=500,
        epochs=epoch,
        verbose=2,
    )


def install_dependencies():
    from subprocess import call

    call(["pip", "install", "--upgrade", "pip"])
    call(["pip", "install", "tensorflow_addons"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--dataset_bottleneck", type=lambda s: s.lower() in ["true", "t", "yes", "1"], default=False
    )
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")

    args = parser.parse_args()
    install_dependencies()

    dataset = get_dataset(
        batch_size=args.batch_size, channel_name="train", dataset_bottleneck=args.dataset_bottleneck
    )

    num_gpus = os.environ.get("SM_NUM_GPUS")
    if num_gpus is not None and int(num_gpus) > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = get_model(lr=args.lr)
    else:
        model = get_model(lr=args.lr)

    train(model=model, dataset=dataset, epoch=args.epoch)
