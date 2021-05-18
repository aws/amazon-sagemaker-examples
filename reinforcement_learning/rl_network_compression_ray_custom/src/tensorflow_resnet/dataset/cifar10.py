import os

import tensorflow as tf

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3

NUM_CLASSES = 10
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label

_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
if os.getenv("SM_HOSTS", None) is None:
    _DATA_DIR = "cifar10_data"
else:
    _DATA_DIR = "/opt/ml/input/data/training/"
NUM_IMAGES = {
    "train": 50000,
    "validation": 10000,
}
DATASET_NAME = "CIFAR-10"
_NUM_DATA_FILES = 5


def get_filenames(is_training, data_dir):
    """Returns a list of filenames."""
    data_dir = os.path.join(data_dir, "cifar-10-batches-bin")

    assert os.path.exists(data_dir), (
        "Run cifar10_download_and_extract.py first to download and extract the " "CIFAR-10 data."
    )

    if is_training:
        return [
            os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, _NUM_DATA_FILES + 1)
        ]
    else:
        return [os.path.join(data_dir, "test_batch.bin")]


def parse_record(raw_record, is_training, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
    # Convert bytes to a vector of uint8 that is record_bytes long.
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(record_vector[0], tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES], [_NUM_CHANNELS, _HEIGHT, _WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    image = preprocess_image(image, is_training)
    image = tf.cast(image, dtype)

    return image, label


def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def input_fn(
    is_training,
    data_dir,
    batch_size,
    num_epochs=1,
    dtype=tf.float32,
    datasets_num_private_threads=20,
    num_parallel_batches=20,
):
    """Input function which provides batches for train or eval.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for images/features
      datasets_num_private_threads: Number of private threads for tf.data.
      num_parallel_batches: Number of parallel batches for tf.data.

    Returns:
      A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=NUM_IMAGES["train"],
        parse_record_fn=parse_record,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        num_parallel_batches=num_parallel_batches,
    )


def process_record_dataset(
    dataset,
    is_training,
    batch_size,
    shuffle_buffer,
    parse_record_fn,
    num_epochs=1,
    dtype=tf.float32,
    datasets_num_private_threads=None,
    num_parallel_batches=1,
):
    """Given a Dataset with raw records, return an iterator over the records.

    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.
      dtype: Data type to use for images/features.
      datasets_num_private_threads: Number of threads for a private
        threadpool created for all datasets computation.
      num_parallel_batches: Number of parallel batches for tf.data.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False,
        )
    )
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset
