"""
This script is a ResNet training script which uses Tensorflow's Keras model to Estimator interface.
Using this script with the official SageMaker Framework container (i.e. AWS Deep Learning Container), enables you to make use of the SageMaker Debugger without any Debugger related
additions to the training script. Hooks will automatically be added. This experience is termed as "Zero Script Change". For more information, please refer to https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md#zero-script-change.
In Zero Script Change experience, a deafult set of collections are saved: https://github.com/awslabs/sagemaker-debugger/blob/master/docs/api.md#default-collections-saved. For information on how to customize the list of collections that are saved, please refer to https://github.com/awslabs/sagemaker-debugger/blob/master/docs/sagemaker.md#saving-reductions-for-a-custom-collection.
"""

# Standard Library
import argparse

# Third Party
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.resnet50 import ResNet50


def train(classifier, batch_size, epoch, model):
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

    classifier.train(input_fn=lambda: input_fn(training=True), steps=25)
    classifier.evaluate(input_fn=lambda: input_fn(training=False))


def main():
    parser = argparse.ArgumentParser(description="Train resnet50 cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--model_dir", type=str, default="./model_keras_resnet")
    opt = parser.parse_args()

    model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
    optimizer = tf.keras.optimizers.Adam()

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # Create the Estimator
    classifier = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                       model_dir=opt.model_dir)

    # start the training.
    train(classifier, opt.batch_size, opt.epoch, model)


if __name__ == "__main__":
    main()
