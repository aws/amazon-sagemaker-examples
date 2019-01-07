"""
This sample shows how to use python 3 with TensorFlow and SageMaker
"""
import argparse
from keras.models import Sequential
from keras.layers import Dense
import numpy
import os

from model_exporter_keras_to_pb import ModelExporterKerasToProtobuf


def input_transformer_load(filename):
    data = numpy.loadtxt(filename, delimiter=",")
    x = data[:, 1:8]
    y = data[:, 0]
    return x, y


def train(training_dir, training_filename, val_dir, val_filename, model_snapshotdir, epochs=10, batch_size=32):
    """
    This is fully customisable code to train your model.
    :param training_dir:
    :param training_filename:
    :param val_dir:
    :param val_filename:
    :param model_snapshotdir:
    :param epochs:
    :param batch_size:
    :return: Returns the trained model
    """
    # Step 1: Do your training here
    # Define deep learning architecture
    # In this this is a fully connected multi-layer perceptron
    model = Sequential([Dense(10, activation='relu', name='input-layer'),
                        Dense(10, activation='relu', name='hidden-layer'),
                        Dense(1, activation='linear', name='output-layer')])

    # For a mean squared error regression problem
    # Using RMSProp optimiser with mean squared error 
    model.compile(optimizer='rmsprop',
                  loss='mse', metrics=['mse'])

    # load train & test data
    train_x, train_y = input_transformer_load(os.path.join(training_dir, training_filename))
    val_x, val_y = input_transformer_load(os.path.join(val_dir, val_filename))
    # Start training
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_x, val_y))
    # model evaluate
    scores = model.evaluate(val_x, val_y)
    # model save in keras default format
    model_keras_path = os.path.join(model_snapshotdir, "abalone_age_predictor.h5")
    model.save(model_keras_path)

    # Step 2: Log your metrics in a special format so that it can be extracted using a regular expression.
    # This allows SageMaker to report this metrics and allows hyper parameter tuning
    # Note: Use a special marker for SageMaker to extract the metrics, say ## Metric ##
    print("## metric_mean_squared_error ##: {}".format(scores[1]))

    # Step 3: Save your model in pb format so that SageMaker TensorFlow container can serve this
    model_pb_exporter = ModelExporterKerasToProtobuf()
    # SageMaker Tensorflow serving container is capable of serving more than one model.
    # The container expects the model.pb to be within a directory structure <model_name>/<model_version>
    model_name = "abalone_age_predictor"
    # Note: the name of the version directory should match the regex re.match('^\d+$', dir)
    model_version = "1"
    # Follow this exact naming convention
    model_pb_path = os.path.join(model_snapshotdir, model_name, model_version)
    model_pb_exporter(model_keras_path, model_pb_path)

    return model


if '__main__' == __name__:
    parser = argparse.ArgumentParser()

    # Train dir files
    parser.add_argument("--traindata", help="The input file wrt to the training directory", required=True)
    # The environment variable SM_CHANNEL_TRAIN is defined
    parser.add_argument('--traindata-dir',
                        help='The directory containing training artifacts such as training data',
                        default=os.environ.get('SM_CHANNEL_TRAIN', "."))
    # val dir files
    parser.add_argument("--validationdata", help="The validation input file wrt to the val directory", required=True)
    parser.add_argument('--validationdata-dir',
                        help='The directory containing validation artifacts such as validation data',
                        default=os.environ.get('SM_CHANNEL_VALIDATION', "."))

    # output dir to save any files such as predictions, logs, etc
    parser.add_argument("--outputdir", help="The output dir to save results",
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', "result_data")
                        )

    parser.add_argument("--model_dir", help="Do not use this.. required by SageMaker", default=None)

    # This is where the model needs to be saved to
    parser.add_argument("--snapshot_dir", help="The directory to save the snapshot to..",
                        default=os.environ.get('SM_MODEL_DIR', "."))

    # Additional parameters for your code
    parser.add_argument("--epochs", help="The number of epochs", default=10, type=int)
    parser.add_argument("--batch-size", help="The mini batch size", default=30, type=int)

    args = parser.parse_args()

    # set up dir
    if not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir)

    train(args.traindata_dir, args.traindata, args.validationdata_dir, args.validationdata, args.snapshot_dir,
          args.epochs, args.batch_size)
