
import argparse
import os

# Third Party
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import to_categorical


def model(x_train, y_train, x_valid, y_valid, batch_size, epoch, optimizer):
    print('batch_size: {}, epoch: {}, optimizer: {}'.format(batch_size, epoch, optimizer))

    Y_train = to_categorical(y_train, 10)
    Y_valid = to_categorical(y_valid, 10)

    X_train = x_train.astype("float32")
    X_valid = x_valid.astype("float32")

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_valid -= mean_image
    X_train /= 128.0
    X_valid /= 128.0

    model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print('Start the training')

    # start the training.
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        verbose=1,
    )

    return model


def _load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_labels.npy'))
    return x_train, y_train


def _load_validation_data(base_dir):
    x_validation = np.load(os.path.join(base_dir, 'validation_data.npy'))
    y_validation = np.load(os.path.join(base_dir, 'validation_labels.npy'))
    return x_validation, y_validation


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


def main():
    args, unknown = _parse_args()

    train_data, train_labels = _load_training_data(args.train)
    validation_data, validation_labels = _load_validation_data(args.validation)

    cifar10_classifier = model(train_data,
                               train_labels,
                               validation_data,
                               validation_labels,
                               args.batch_size,
                               args.epoch,
                               args.optimizer)

    cifar10_classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')

if __name__ == "__main__":
    main()