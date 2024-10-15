from __future__ import absolute_import

import os
import sys
import time

from utils import (
    ExitSignalHandler,
    load_json_object,
    print_files_in_path,
    print_json_object,
    save_model_artifacts,
    write_failure_file,
)

hyperparameters_file_path = "/opt/ml/input/config/hyperparameters.json"
inputdataconfig_file_path = "/opt/ml/input/config/inputdataconfig.json"
resource_file_path = "/opt/ml/input/config/resourceconfig.json"
data_files_path = "/opt/ml/input/data/"
failure_file_path = "/opt/ml/output/failure"
model_artifacts_path = "/opt/ml/model/"

training_job_name_env = "TRAINING_JOB_NAME"
training_job_arn_env = "TRAINING_JOB_ARN"


def train():
    try:
        print("\nRunning training...")

        if os.path.exists(hyperparameters_file_path):
            hyperparameters = load_json_object(hyperparameters_file_path)
            print("\nHyperparameters configuration:")
            print_json_object(hyperparameters)

        if os.path.exists(inputdataconfig_file_path):
            input_data_config = load_json_object(inputdataconfig_file_path)
            print("\nInput data configuration:")
            print_json_object(input_data_config)

            for key in input_data_config:
                print("\nList of files in {0} channel: ".format(key))
                channel_path = data_files_path + key + "/"
                print_files_in_path(channel_path)

        if os.path.exists(resource_file_path):
            resource_config = load_json_object(resource_file_path)
            print("\nResource configuration:")
            print_json_object(resource_config)

        if training_job_name_env in os.environ:
            print("\nTraining job name: ")
            print(os.environ[training_job_name_env])

        if training_job_arn_env in os.environ:
            print("\nTraining job ARN: ")
            print(os.environ[training_job_arn_env])

        # This object is used to handle SIGTERM and SIGKILL signals.
        signal_handler = ExitSignalHandler()

        # Dummy net.
        net = None

        # Run training loop.
        epochs = 5
        for x in range(epochs):
            print("\nRunning epoch {0}...".format(x))

            time.sleep(30)

            if signal_handler.exit_now:
                print("Received SIGTERM/SIGINT. Saving training state and exiting.")
                # Save state here.
                save_model_artifacts(model_artifacts_path, net)
                sys.exit(0)

            print("Completed epoch {0}.".format(x))

        # At the end of the training loop, we have to save model artifacts.
        save_model_artifacts(model_artifacts_path, net)

        print("\nTraining completed!")
    except Exception as e:
        write_failure_file(failure_file_path, str(e))
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        print("Missing required argument 'train'.", file=sys.stderr)
        sys.exit(1)
