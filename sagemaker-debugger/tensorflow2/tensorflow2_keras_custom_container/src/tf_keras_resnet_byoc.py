"""
This script is a ResNet training script which uses Tensorflow's Keras interface.
It has been orchestrated with SageMaker Debugger hooks to allow saving tensors during training.
These hooks have been instrumented to read from json configuration that SageMaker will put in the training container.
Configuration provided to the SageMaker python SDK when creating a job will be passed on to the hook.
This allows you to use the same script with differing configurations across different runs.
If you use an official SageMaker Framework container (i.e. AWS Deep Learning Container), then
you do not have to orchestrate your script as below. Hooks will automatically be added in those environments.
For more information, please refer to https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md
"""

# Standard Library
import argparse
import random

# Third Party
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import smdebug.tensorflow as smd


def train(batch_size, epoch, model, hook):
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

    hook.set_mode(smd.modes.TRAIN)
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epoch,
              validation_data=(X_valid, Y_valid),
              shuffle=True,
              callbacks=[hook])


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Number of steps to train for. If this" "is passed, it overrides num_epochs",
    )
    parser.add_argument(
        "--num_eval_steps",
        type=int,
        help="Number of steps to evaluate for. If this"
             "is passed, it doesnt evaluate over the full eval set",
    )
    args = parser.parse_args()

    if args.random_seed:
        tf.random.set_seed(2)
        np.random.seed(2)
        random.seed(12)

    model = ResNet50(weights=None, input_shape=(32,32,3), classes=10)

    # Create hook from the configuration provided through sagemaker python sdk
    hook = smd.KerasHook.create_from_json_file()
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # start the training.
    train(args.batch_size, args.epoch, model, hook)

if __name__ == "__main__":
    main()
