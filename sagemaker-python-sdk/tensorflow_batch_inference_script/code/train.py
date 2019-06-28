import argparse
import json
import logging
import os
import re

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from model_def import get_model, HEIGHT, WIDTH, DEPTH, NUM_CLASSES

logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES

#  Copy inference pre/post-processing script so it will be included in the model package
os.system('mkdir /opt/ml/model/code')
os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')


class CustomTensorBoardCallback(TensorBoard):

    def on_batch_end(self, batch, logs=None):
        pass


def get_filenames(channel_name, channel):

    if channel_name in ['train', 'validation', 'eval']:
        return [os.path.join(channel, channel_name + '.tfrecords')]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)


def _input(epochs, batch_size, channel, channel_name):

    mode = args.data_config[channel_name]['TrainingInputMode']
    filenames = get_filenames(channel_name, channel)
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


def save_model(model, output):

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    logging.info("Model successfully saved at: {}".format(output))
    return


def main(args):

    mpi = False
    if 'sourcedir.tar.gz' in args.tensorboard_dir:
        tensorboard_dir = re.sub('source/sourcedir.tar.gz', 'model', args.tensorboard_dir)
    else:
        tensorboard_dir = args.tensorboard_dir
    logging.info("Writing TensorBoard logs to {}".format(tensorboard_dir))

    if 'sagemaker_mpi_enabled' in args.fw_params:
        if args.fw_params['sagemaker_mpi_enabled']:
            import horovod.tensorflow.keras as hvd
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
    train_dataset = _input(args.epochs, args.batch_size, args.train, 'train')
    eval_dataset = _input(args.epochs, args.batch_size, args.eval, 'eval')
    validation_dataset = _input(args.epochs, args.batch_size, args.validation, 'validation')

    logging.info("configuring model")
    model = get_model(args.learning_rate, args.weight_decay, args.optimizer, args.momentum, mpi, hvd)
    callbacks = []
    if mpi:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.h5'))
            callbacks.append(CustomTensorBoardCallback(log_dir=tensorboard_dir))
    else:
        callbacks.append(ModelCheckpoint(args.output_dir + '/checkpoint-{epoch}.h5'))
        callbacks.append(CustomTensorBoardCallback(log_dir=tensorboard_dir))

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
            return save_model(model, args.model_output_dir)
    else:
        return save_model(model, args.model_output_dir)


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

    parser.add_argument('--train',type=str,required=False,default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation',type=str,required=False,default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--eval',type=str,required=False,default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir',type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir',type=str,default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--tensorboard-dir',type=str,default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument('--weight-decay',type=float,default=2e-4,help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate',type=float,default=0.001,help='Initial learning rate.')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--data-config',type=json.loads,default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params',type=json.loads,default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--momentum',type=float,default='0.9')

    args = parser.parse_args()

    main(args)
