"""
This script is a ResNet training script which uses Tensorflow's Keras interface, and provides an example of how to use SageMaker Debugger when you use your own custom container in SageMaker or your own script outside SageMaker.
It has been orchestrated with SageMaker Debugger hooks to allow saving tensors during training.
These hooks have been instrumented to read from a JSON configuration that SageMaker puts in the training container.
Configuration provided to the SageMaker python SDK when creating a job will be passed on to the hook.
This allows you to use the same script with different configurations across different runs.

If you use an official SageMaker Framework container (i.e. AWS Deep Learning Container), you do not have to orchestrate your script as below. Hooks are automatically added in those environments. This experience is called a "zero script change". For more information, see https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md#zero-script-change. An example of the same is provided at https://github.com/awslabs/amazon-sagemaker-examples/sagemaker-debugger/tensorflow2/tensorflow2_zero_code_change.
"""

# Standard Library
import argparse
import random

# Third Party
import numpy as np

# smdebug modification: Import smdebug support for Tensorflow
import smdebug.tensorflow as smd
import tensorflow.compat.v2 as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


def train(batch_size, epoch, model, hook):
    (X_train, y_train), (X_valid, y_valid) = cifar10.load_data()

    Y_train = to_categorical(y_train, 10)
    Y_valid = to_categorical(y_valid, 10)

    X_train = X_train.astype("float32")
    X_valid = X_valid.astype("float32")

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_valid -= mean_image
    X_train /= 128.0
    X_valid /= 128.0

    # register hook to save the following scalar values
    hook.save_scalar("epoch", epoch)
    hook.save_scalar("batch_size", batch_size)
    hook.save_scalar("train_steps_per_epoch", len(X_train) / batch_size)
    hook.save_scalar("valid_steps_per_epoch", len(X_valid) / batch_size)

    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(X_valid, Y_valid),
        shuffle=False,
        # smdebug modification: Pass the hook as a Keras callback
        callbacks=[hook],
    )


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--random_seed", type=bool, default=False)

    args = parser.parse_args()

    if args.random_seed:
        tf.random.set_seed(2)
        np.random.seed(2)
        random.seed(12)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)

        # smdebug modification:
        # Create hook from the configuration provided through sagemaker python sdk.
        # This configuration is provided in the form of a JSON file.
        # Default JSON configuration file:
        # {
        #     "LocalPath": <path on device where tensors will be saved>
        # }"
        # Alternatively, you could pass custom debugger configuration (using DebuggerHookConfig)
        # through SageMaker Estimator. For more information, https://github.com/aws/sagemaker-python-sdk/blob/master/doc/amazon_sagemaker_debugger.rst
        hook = smd.KerasHook.create_from_json_file()

        opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # start the training.
    train(args.batch_size, args.epoch, model, hook)


if __name__ == "__main__":
    main()
