#     Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import os
from os import listdir
import argparse
from os.path import isfile, join
import numpy as np
import logging
import json
import glob
import datetime


logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

NUM_FEATURES = 50 #50 #50 #50 #50 #50 #50 #7
INPUT_TENSOR_NAME = 'inputs_input'  # needs to match the name of the first layer + "_input"

def list_files_in_dir(which_dir):
    logging.info('\nContents of {}:'.format(which_dir))
    files = glob.glob(which_dir + '/**', recursive=True)
    for f in files:
        logging.info(f)

def _time():
    return datetime.datetime.now().time()

class PipeDebugCallback(tf.keras.callbacks.Callback): # is not called for some reason
  def on_train_batch_begin(self, batch, logs=None):
    logging.info('Training: batch {} BEGINS at {}'.format(batch, _time()))
    list_files_in_dir('/opt/ml/input/data')

  def on_train_batch_end(self, batch, logs=None):
    logging.info('Training: batch {} ENDS at {}'.format(batch, _time()))
    list_files_in_dir('/opt/ml/input/data')

  def on_test_batch_begin(self, batch, logs=None):
    logging.info('Evaluating: batch {} BEGINS at {}'.format(batch, _time()))
    list_files_in_dir('/opt/ml/input/data')

  def on_test_batch_end(self, batch, logs=None):
    logging.info('Evaluating: batch {} ENDS at {}'.format(batch, _time()))
    list_files_in_dir('/opt/ml/input/data')
    
def get_filenames(channel_name, channel):
    list_files_in_dir('/opt/ml/input/data')
    mode = args.data_config[channel_name]['TrainingInputMode']
    fnames = []
    if channel_name in ['train', 'val', 'test']:
        if mode == 'File':
            for f in listdir(channel):
                fnames.append(os.path.join(channel, f))
        logging.info('returning filenames: {}'.format(fnames))
        return [fnames]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)

def train_input_fn():
    return _input(args.epochs, args.batch_size, args.train, 'train')

def test_input_fn():
    return _input(args.epochs, args.batch_size, args.test, 'test')

def val_input_fn():
    return _input(args.epochs, args.batch_size, args.val, 'val')


def _dataset_parser(value):
    """Parse a record from 'value'."""
    feature_description = {
        'features': tf.VarLenFeature(tf.float32),
        'label'   : tf.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.parse_single_example(value, feature_description)

    label = tf.cast(example['label'], tf.int32)
    logging.info('parsed label: {}'.format(label))
    data  = example['features'].values 
    logging.info('parsed features: {}'.format(data))
    
    return data, label


def _input(epochs, batch_size, channel, channel_name):
    mode = args.data_config[channel_name]['TrainingInputMode']
    """Uses the tf.data input pipeline for our dataset.
    Args:
        mode: Standard names for model modes (tf.estimators.ModeKeys).
        batch_size: The number of samples per batch of input requested.
    """
    logging.info("Running {} in {} mode for {} epochs".format(channel_name, mode, epochs))

    filenames = get_filenames(channel_name, channel)

    # Repeat infinitely.
    if mode == 'Pipe':
        from sagemaker_tensorflow import PipeModeDataset
        dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')
    else:
        dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(batch_size)

    # Parse records.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)
    ## TF Dataset question: why does _dataset_parser only get called once per channel??

    # Shuffle training records.
    if channel_name == 'train':
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        buffer_size = args.num_train_samples // args.batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    iterator = dataset.make_one_shot_iterator()
    features_batch, label_batch = iterator.get_next()
    
    with tf.Session() as sess:
        logging.info('type of features_batch: {}, type of values: {}'.format(type(features_batch), 
                                                         type(features_batch)))
        logging.info('label_batch: {}'.format(label_batch))
        logging.info('type of label_batch: {}'.format(type(label_batch)))

    return {INPUT_TENSOR_NAME: features_batch}, label_batch

def save_model(model, output):
    logging.info('Saving model...')
    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, output)
    logging.info('Model successfully saved at: {}, with the following content:'.format(output))
    list_files_in_dir(output)
    return


if __name__=='__main__':
    logging.info('Entering training script at {}'.format(_time()))
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_train_samples', type=int)
    parser.add_argument('--num_val_samples', type=int)
    parser.add_argument('--num_test_samples', type=int)
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--data-config', type=json.loads, 
                        default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    args, _ = parser.parse_known_args()
    logging.info('args: {}'.format(args))
    epochs = args.epochs
    
    logging.info('Getting data')
    train_dataset = train_input_fn()
    test_dataset  = test_input_fn()
    val_dataset   = val_input_fn()

    logging.info('Configuring model')
    
    network = models.Sequential()
    network.add(layers.Dense(32, activation='relu', input_shape=(NUM_FEATURES,), name='inputs'))
    network.add(layers.Dense(32, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))
    network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # NOTE: Ordinarily, you do not need to specify 'steps_per_epoch' and 'validation_steps'.
    # However, when passing in a tf Dataset (like PipeModeDataset) to the fit method, these
    # parameters are required. Else you will get the following error:
    #
    #    ValueError: When using data tensors as input to a model, you should specify
    #    the `steps_per_epoch` argument.
    #
    
    # NOTE: We leverage the number of hosts (driven by instance_count used when launching
    # the training job) when determining the right number of steps for the training. On 
    # single node training, it is sufficient to use the number of steps equivalent to 
    # samples divided by batch size. This ensures the full dataset is used in each epoch.
    # When performing distributed training, the steps can be reduced by a factor of the number
    # of training instances. So for a 2-node job, 1000 steps can be reduced to 500 steps on each host.
    #

    logging.info('\nStarting training at {}'.format(_time()))

    num_hosts = len(os.environ.get('SM_HOSTS'))
    train_steps = args.num_train_samples // args.batch_size // num_hosts
    val_steps   = args.num_val_samples   // args.batch_size // num_hosts
    test_steps  = args.num_test_samples  // args.batch_size // num_hosts
    print('Train Steps: {}, Val Steps: {}, Test Steps: {}'.format(train_steps, val_steps, test_steps))
    
    fitCallbacks = [ModelCheckpoint(os.environ.get('SM_OUTPUT_DATA_DIR') + '/checkpoint-{epoch}.h5'),
                    PipeDebugCallback()]
    network.fit(x=train_dataset[0], y=train_dataset[1],
                steps_per_epoch=train_steps, epochs=args.epochs, 
                validation_data=val_dataset, validation_steps=val_steps,
                callbacks=fitCallbacks)

    logging.info('\nTraining completed at {}'.format(_time()))

    logging.info('\nEvaluating against test set...')
    score = network.evaluate(test_dataset[0], test_dataset[1], 
                             steps=test_steps,
                             verbose=1)

    logging.info('    Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))

    # Save the model if we are executing on the master host
    if args.current_host == args.hosts[0]:
        logging.info('Saving model, since we are master host')
        save_model(network, os.environ.get('SM_MODEL_DIR'))
    else:
        logging.info('NOT saving model, will leave that up to master host')

    logging.info('\nExiting training script at {}'.format(_time()))