import argparse
import codecs
import json
import logging
import numpy as np
import os
import re

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from model_def import get_model, HEIGHT, WIDTH, DEPTH, NUM_CLASSES
from utilities import process_input


logging.getLogger().setLevel(logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)


#  Copy inference pre/post-processing script so it will be included in the model package
os.system('mkdir /opt/ml/model/code')
os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')


class CustomTensorBoardCallback(TensorBoard):
    
    def on_batch_end(self, batch, logs=None):
        pass

    
def save_history(path, history):

    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            history_for_json[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
           if  type(history.history[key][0]) == np.float32 or type(history.history[key][0]) == np.float64:
               history_for_json[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json, f, separators=(',', ':'), sort_keys=True, indent=4) 


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
            hvd.init()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = str(hvd.local_rank())
            K.set_session(tf.Session(config=config))
    else:
        hvd = None

    logging.info("Running with MPI={}".format(mpi))
    logging.info("getting data")
    train_dataset = process_input(args.epochs, args.batch_size, args.train, 'train', args.data_config)
    eval_dataset = process_input(args.epochs, args.batch_size, args.eval, 'eval', args.data_config)
    validation_dataset = process_input(args.epochs, args.batch_size, args.validation, 'validation', args.data_config)

    logging.info("configuring model")
    model = get_model(args.learning_rate, args.weight_decay, args.optimizer, args.momentum, 1, mpi, hvd)
    callbacks = []
    if mpi:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
        if hvd.rank() == 0:
            callbacks.append(ModelCheckpoint(args.output_data_dir + '/checkpoint-{epoch}.h5'))
            callbacks.append(CustomTensorBoardCallback(log_dir=tensorboard_dir))
    else:
        callbacks.append(ModelCheckpoint(args.output_data_dir + '/checkpoint-{epoch}.h5'))
        callbacks.append(CustomTensorBoardCallback(log_dir=tensorboard_dir))
        
    logging.info("Starting training")
    size = 1
    if mpi:
        size = hvd.size()
        
    history = model.fit(x=train_dataset[0], 
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

    if mpi:
        if hvd.rank() == 0:
            save_history(args.model_dir + "/hvd_history.p", history)
            return save_model(model, args.model_output_dir)
    else:
        save_history(args.model_dir + "/hvd_history.p", history)
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
    parser.add_argument('--output_data_dir',type=str,default=os.environ.get('SM_OUTPUT_DATA_DIR'))
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
