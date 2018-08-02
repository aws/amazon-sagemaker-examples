#!/usr/bin/env python

# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function

import os
import json
import sys
import subprocess
import traceback

# These are the paths to where SageMaker mounts interesting things in your container.
prefix = '/opt/ml/'
input_path = os.path.join(prefix,'input/data')
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name = 'training'
training_path = os.path.join(input_path, channel_name)

# default params
training_script = 'cifar10.py'
default_params = ['--model-dir', str(model_path)]


# Execute your training algorithm.
def _run(cmd):
    """Invokes your training algorithm."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
    stdout, stderr = process.communicate()

    return_code = process.poll()
    if return_code:
        error_msg = 'Return Code: {}, CMD: {}, Err: {}'.format(return_code, cmd, stderr)
        raise Exception(error_msg)


def _hyperparameters_to_cmd_args(hyperparameters):
    """
    Converts our hyperparameters, in json format, into key-value pair suitable for passing to our training
    algorithm.
    """
    cmd_args_list = []

    for key, value in hyperparameters.items():
        cmd_args_list.append('--{}'.format(key))
        cmd_args_list.append(value)

    return cmd_args_list


if __name__ == '__main__':
    try:
        # Amazon SageMaker makes our specified hyperparameters available within the
        # /opt/ml/input/config/hyperparameters.json.
        # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container
        with open(param_path, 'r') as tc:
            training_params = json.load(tc)

        python_executable = sys.executable
        cmd_args = _hyperparameters_to_cmd_args(training_params)

        train_cmd = [python_executable, training_script] + default_params + cmd_args

        _run(train_cmd)
        print('Training complete.')

        # A zero exit code causes the job to be marked a Succeeded.
        sys.exit(0)
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


