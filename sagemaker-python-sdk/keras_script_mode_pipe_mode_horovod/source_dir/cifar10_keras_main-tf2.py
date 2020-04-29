#     Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License").
#     You may not use this file except in compliance with the License.
#     A copy of the License is located at
#    
#         https://aws.amazon.com/apache-2-0/
#    
#     or in the "license" file accompanying this file. This file is distributed
#     on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#     express or implied. See the License for the specific language governing
#     permissions and limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import re
import glob

# import keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

logging.getLogger().setLevel(logging.INFO)

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
INPUT_TENSOR_NAME = 'inputs_input'  # needs to match the name of the first layer + "_input"

def keras_model_fn(learning_rate, weight_decay, optimizer, momentum, mpi=False, hvd=False):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model will be transformed into a TensorFlow Estimator before training and it will be saved in a 
    TensorFlow Serving SavedModel at the end of training.

    Args:
        hyperparameters: The hyperparameters passed to the SageMaker TrainingJob that runs your TensorFlow 
                         training script.
    Returns: A compiled Keras model
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

    size = hvd.size() if mpi else 1

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
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    
    return model

def get_filenames(input_data_dir):
    file_names = glob.glob('{}/*.tfrecords'.format(input_data_dir))
    
    return file_names

def train_input_fn(hvd, mpi=False):
    if mpi:
        return _input(args.epochs, args.batch_size, None, 'train', hvd)
    else:
        return _input(args.epochs, args.batch_size, args.train, 'train')

def eval_input_fn():
    return _input(args.epochs, args.batch_size, args.eval, 'eval')

def validation_input_fn(hvd, mpi=False):
    if mpi:
        return _input(args.epochs, args.batch_size, None, 'validation', hvd)
    else:
        return _input(args.epochs, args.batch_size, args.validation, 'validation')


def _input(epochs, batch_size, channel, channel_name, hvd=None):
    
    # If Horovod, assign channel name using the horovod rank
    if hvd != None:
        channel_name = '{}_{}'.format(channel_name, hvd.rank())
    
    channel_input_dir = args.training_env['channel_input_dirs'][channel_name]
    
    mode = args.data_config[channel_name]['TrainingInputMode']
    
    """Uses the tf.data input pipeline for CIFAR-10 dataset.
    Args:
        mode: Standard names for model modes (tf.estimators.ModeKeys).
        batch_size: The number of samples per batch of input requested.
    """
    
    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')#, benchmark=True)
    else:
        filenames = get_filenames(channel_input_dir)
        dataset = tf.data.TFRecordDataset(filenames)
    
    if 'train' in channel_name:
        dataset = dataset.repeat(epochs)
    else:
        dataset = dataset.repeat(20)

    # Parse records.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)

    # Potentially shuffle records.
#     if hvd == None and 'train' in channel_name:
    if 'train' in channel_name:
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(10)
    
    return dataset

def _train_preprocess_fn(image):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    featdef = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(value, featdef)
    image = tf.io.decode_raw(example['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(example['label'], tf.int32)
    image = _train_preprocess_fn(image)
    
    return image, tf.one_hot(label, NUM_CLASSES)

def main(args):
    mpi = False
    if 'sourcedir.tar.gz' in args.tensorboard_dir:
        tensorboard_dir = re.sub('source/sourcedir.tar.gz', 'output', args.tensorboard_dir)
    else:
        tensorboard_dir = args.tensorboard_dir
    logging.info("Writing TensorBoard logs to {}".format(tensorboard_dir))
    if 'sagemaker_mpi_enabled' in args.fw_params:
        if args.fw_params['sagemaker_mpi_enabled']:
            import horovod.keras as hvd
            mpi = True
            # Horovod: initialize Horovod.
            hvd.init()

            # Horovod: pin GPU to be used to process local rank (one GPU per process)
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    else:
        hvd = None

    train_dataset = train_input_fn(hvd, mpi)
    
    eval_dataset = eval_input_fn()
    validation_dataset = validation_input_fn(hvd, mpi)

    model = keras_model_fn(args.learning_rate, args.weight_decay, args.optimizer, args.momentum, mpi, hvd)

    callbacks = []
    if mpi:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.ckpt',
                                            save_weights_only=True,
                                            verbose=2))
            callbacks.append(TensorBoard(log_dir=tensorboard_dir, update_freq='epoch'))
    else:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
        callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.ckpt',
                                        save_weights_only=True,
                                        verbose=2))
        callbacks.append(TensorBoard(log_dir=tensorboard_dir, update_freq='epoch'))
    logging.info("Starting training")
    size = 1
    if mpi:
        size = hvd.size()
        
    if mpi and hvd.rank() > 0:
        # for horovod training, no validation for non-master nodes (rank > 0)
        model.fit(train_dataset,
                  steps_per_epoch=((num_examples_per_epoch('train') // args.batch_size) // size),
                  epochs=args.epochs, 
                  validation_data=validation_dataset,
                  validation_steps=((num_examples_per_epoch('validation') // args.batch_size) // size), 
                  callbacks=callbacks,
                  verbose=2
                 )
    else:
        model.fit(train_dataset,
                  steps_per_epoch=((num_examples_per_epoch('train') // args.batch_size) // size),
                  epochs=args.epochs, 
                  validation_data=validation_dataset,
                  validation_steps=((num_examples_per_epoch('validation') // args.batch_size) // size), 
                  callbacks=callbacks,
                  verbose=2
                 )
        
    if not mpi or (mpi and hvd.rank() == 0):
        score = model.evaluate(eval_dataset, 
                  steps=num_examples_per_epoch('eval') // args.batch_size,
                  verbose=2)

        logging.info('Test loss:{}'.format(score[0]))
        logging.info('Test accuracy:{}'.format(score[1]))

    # Horovod: Save model only on worker 0 (i.e. master)
    if mpi:
        if hvd.rank() == 0:
            model.save(args.model_output_dir)
    else:
        model.save(args.model_output_dir)

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
        '--training-env',
        type=json.loads,
        default=os.environ.get('SM_TRAINING_ENV')
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
