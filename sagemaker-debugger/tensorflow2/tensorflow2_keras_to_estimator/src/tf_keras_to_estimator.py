"""
This script is a ResNet training script which uses Tensorflow's Keras model to Estimator interface.
It has been orchestrated with SageMaker Debugger hook to allow saving tensors during training.
Here, the hook has been created using its constructor to allow running this locally for your experimentation.
When you want to run this script in SageMaker, it is recommended to create the hook from json file.
"""

# Standard Library
import argparse

# Third Party
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.resnet50 import ResNet50

# First Party
import smdebug.tensorflow as smd


def train(classifier, batch_size, epoch, model, hook):
    def input_fn(training=True):
        datasets, info = tfds.load(name="cifar10", with_info=True, as_supervised=True)
        cifar10_train, cifar10_test = datasets["train"], datasets["test"]

        def scale(image, label):
            image = tf.cast(image, tf.float32)
            image /= 255.0

            return image, label

        data = cifar10_test.map(scale)
        if training:
            data = cifar10_train.map(scale).shuffle(10000).repeat()
        return data.batch(batch_size)

    # save_scalar() API can be used to save arbitrary scalar values that may
    # or may not be related to training.
    # Ref: https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#common-hook-api
    hook.save_scalar("epoch", epoch, sm_metric=True)

    hook.set_mode(mode=smd.modes.TRAIN)
    classifier.train(input_fn=lambda: input_fn(training=True), steps=25, hooks=[hook])
    hook.set_mode(mode=smd.modes.EVAL)
    classifier.evaluate(input_fn=lambda: input_fn(training=False), hooks=[hook])

    hook.save_scalar("batch_size", batch_size, sm_metric=True)


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--save_interval", type=int, default=500)
    opt = parser.parse_args()

    model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
    optimizer = tf.keras.optimizers.Adam()

    ##### Enabling SageMaker Debugger ###########
    # wrap the optimizer so the hook can identify the gradients
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    ##### Enabling SageMaker Debugger ###########
    # creating hook
    hook = smd.EstimatorHook(
        out_dir=opt.out_dir,
        # Information on default collections https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#default-collections-saved
        include_collections=["weights", "biases", "gradients", "default"],
        save_config=smd.SaveConfig(save_interval=opt.save_interval),
    )

    # Create the Estimator
    classifier = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=opt.model_dir)

    # start the training.
    train(classifier, opt.batch_size, opt.epoch, model, hook)


if __name__ == "__main__":
    main()
