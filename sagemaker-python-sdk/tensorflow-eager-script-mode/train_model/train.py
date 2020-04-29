import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

from model_def import get_model


tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


def get_train_data(train_dir):
    
    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape,'y train', y_train.shape)

    return x_train, y_train


def get_test_data(test_dir):
    
    x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    print('x test', x_test.shape,'y test', y_test.shape)

    return x_test, y_test
   

if __name__ == "__main__":
        
    args, _ = parse_args()
    
    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)
    
    device = '/cpu:0' 
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    with tf.device(device):
        
        model = get_model()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        model.compile(optimizer=optimizer, loss='mse')    
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(x_test, y_test))

        # evaluate on test set
        scores = model.evaluate(x_test, y_test, batch_size, verbose=2)
        print("Test MSE :", scores)

        # save checkpoint for locally loading in notebook
        saver = tfe.Saver(model.variables)
        saver.save(args.model_dir + '/weights.ckpt')
        # create a separate SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
        tf.contrib.saved_model.save_keras_model(model, args.model_dir)

            
        