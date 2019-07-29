import logging
import os
import tensorflow as tf

from model_def import HEIGHT, WIDTH, DEPTH, NUM_CLASSES


NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES


def _get_filenames(channel_name, channel):
    if channel_name in ['train', 'validation', 'eval']:
        return [os.path.join(channel, channel_name + '.tfrecords')]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)
        

def _train_preprocess_fn(image):

    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image


def _dataset_parser(value):

    featdef = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    example = tf.parse_single_example(value, featdef)
    image = tf.decode_raw(example['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(example['label'], tf.int32)
    image = _train_preprocess_fn(image)
    return image, tf.one_hot(label, NUM_CLASSES)


def process_input(epochs, batch_size, channel, channel_name, data_config):
    
    mode = data_config[channel_name]['TrainingInputMode']
    filenames = _get_filenames(channel_name, channel)
    # Repeat infinitely.
    logging.info("Running {} in {} mode".format(channel_name, mode))
    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')
    else:
        dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(10)

    # Parse records.
    dataset = dataset.map(
        _dataset_parser, num_parallel_calls=10)

    # Potentially shuffle records.
    if channel_name == 'train':
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch


