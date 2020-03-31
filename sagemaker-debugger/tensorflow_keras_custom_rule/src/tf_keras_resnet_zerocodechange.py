"""
This script is a ResNet training script which uses Tensorflow's Keras interface.
It is designed to be used with SageMaker Debugger in an official SageMaker Framework container (i.e. AWS Deep Learning Container). You will notice that this script looks exactly like a normal TensorFlow training script.
The hook needed by SageMaker Debugger to save tensors during training will be automatically added in those environments. 
The hook will load configuration from json configuration that SageMaker will put in the training container from the configuration provided using the SageMaker python SDK when creating a job.
For more information, please refer to https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md 
"""

# Standard Library
import argparse
import random
import time

# Third Party
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def train(batch_size, epoch, model):
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()

    Y_train = to_categorical(y_train, 10)
    Y_valid = to_categorical(y_valid, 10)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_valid -= mean_image
    X_train /= 128.
    X_valid /= 128.

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epoch,
              validation_data=(X_valid, Y_valid),
              shuffle=True)


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")
    opt = parser.parse_args()

    model = ResNet50(weights=None, input_shape=(32,32,3), classes=10)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # start the training.
    train(opt.batch_size, opt.epoch, model)

if __name__ == "__main__":
    main()
