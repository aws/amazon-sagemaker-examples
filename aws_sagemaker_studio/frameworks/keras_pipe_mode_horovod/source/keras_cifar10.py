# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     https://aws.amazon.com/apache-2-0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import re
import os

import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop

logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.INFO)

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
INPUT_TENSOR_NAME = 'inputs_input'  # needs to match the name of the first layer + "_input"


def keras_model_fn(learning_rate, weight_decay, optimizer, momentum, mpi=False, hvd=False):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model is transformed into a TensorFlow Estimator before training and saved in a
    TensorFlow Serving SavedModel at the end of training.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', name='inputs', input_shape=(HEIGHT, WIDTH, DEPTH)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    size = 1
    if mpi:
        size = hvd.size()

    if optimizer.lower() == 'sgd':
        opt = SGD(lr=learning_rate * size, decay=weight_decay, momentum=momentum)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(lr=learning_rate * size, decay=weight_decay)
    else:
        opt = Adam(lr=learning_rate * size, decay=weight_decay)

    if mpi:
        opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def train_input_fn():
    return _input(args.epochs, args.batch_size, args.train, 'train')


def eval_input_fn():
    return _input(args.epochs, args.batch_size, args.eval, 'eval')


def validation_input_fn():
    return _input(args.epochs, args.batch_size, args.validation, 'validation')


def _get_filenames(channel_name, channel):
    if channel_name in ['train', 'validation', 'eval']:
        return [os.path.join(channel, channel_name + '.tfrecords')]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)


def _input(epochs, batch_size, channel, channel_name):
    """Uses the tf.data input pipeline for CIFAR-10 dataset."""
    mode = args.data_config[channel_name]['TrainingInputMode']
    logging.info("Running {} in {} mode".format(channel_name, mode))

    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')
    else:
        filenames = _get_filenames(channel_name, channel)
        dataset = tf.data.TFRecordDataset(filenames)

    # Repeat infinitely.
    dataset = dataset.repeat()
    dataset = dataset.prefetch(10)

    # Parse records.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)

    # Potentially shuffle records.
    if channel_name == 'train':
        # Ensure that the capacity is sufficiently large to provide good random shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    image_batch, label_batch = iterator.get_next()

    return {INPUT_TENSOR_NAME: image_batch}, label_batch


def _train_preprocess_fn(image):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
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
        tf.float32,
    )
    label = tf.cast(example['label'], tf.int32)
    image = _train_preprocess_fn(image)
    return image, tf.one_hot(label, NUM_CLASSES)


def save_model(model, output):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'image': model.input}, outputs={'scores': model.output}
    )

    builder = tf.saved_model.builder.SavedModelBuilder(output+'/1/')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature
        },
    )

    builder.save()
    logging.info("Model successfully saved at: {}".format(output))


def main(args):
    if 'sourcedir.tar.gz' in args.tensorboard_dir:
        tensorboard_dir = re.sub('source/sourcedir.tar.gz', 'model', args.tensorboard_dir)
    else:
        tensorboard_dir = args.tensorboard_dir
    logging.info("Writing TensorBoard logs to {}".format(tensorboard_dir))

    mpi = False
    if 'sagemaker_mpi_enabled' in args.fw_params:
        if args.fw_params['sagemaker_mpi_enabled']:
            import horovod.keras as hvd
            mpi = True
            # Horovod: initialize Horovod.
            hvd.init()

            # Horovod: pin GPU to be used to process local rank (one GPU per process)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = str(hvd.local_rank())
            K.set_session(tf.Session(config=config))
    else:
        hvd = None
    logging.info("Running with MPI={}".format(mpi))

    logging.info("getting data")
    train_dataset = train_input_fn()
    eval_dataset = eval_input_fn()
    validation_dataset = validation_input_fn()

    logging.info("configuring model")
    model = keras_model_fn(args.learning_rate, args.weight_decay, args.optimizer, args.momentum, mpi, hvd)

    callbacks = []
    if mpi:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.h5'))
            callbacks.append(TensorBoard(log_dir=tensorboard_dir, update_freq='epoch'))
    else:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
        callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.h5'))
        callbacks.append(TensorBoard(log_dir=tensorboard_dir, update_freq='epoch'))

    logging.info("Starting training")
    size = 1
    if mpi:
        size = hvd.size()

    model.fit(x=train_dataset[0],
              y=train_dataset[1],
              steps_per_epoch=(num_examples_per_epoch('train') // args.batch_size) // size,
              epochs=args.epochs,
              validation_data=validation_dataset,
              validation_steps=(num_examples_per_epoch('validation') // args.batch_size) // size,
              callbacks=callbacks)

    score = model.evaluate(eval_dataset[0],
                           eval_dataset[1],
                           steps=num_examples_per_epoch('eval') // args.batch_size,
                           verbose=0)

    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))

    # Horovod: Save model only on worker 0 (i.e. master)
    if mpi:
        if hvd.rank() == 0:
            save_model(model, args.model_output_dir)
    else:
        save_model(model, args.model_output_dir)


def num_examples_per_epoch(subset='train'):
    if subset == 'train':
        return 40000
    elif subset == 'validation':
        return 10000
    elif subset == 'eval':
        return 10000
    else:
        raise ValueError('Invalid data subset "%s"' % subset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_TRAIN'),
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_VALIDATION'),
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--eval',
        type=str,
        required=False,
        default=os.environ.get('SM_CHANNEL_EVAL'),
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--model_output_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """)
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training.')
    parser.add_argument(
        '--data-config',
        type=json.loads,
        default=os.environ.get('SM_INPUT_DATA_CONFIG')
    )
    parser.add_argument(
        '--fw-params',
        type=json.loads,
        default=os.environ.get('SM_FRAMEWORK_PARAMS')
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default='0.9'
    )
    args = parser.parse_args()
    main(args)
