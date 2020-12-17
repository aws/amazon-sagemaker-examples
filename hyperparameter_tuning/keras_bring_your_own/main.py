# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import argparse
import os
import shlex
import subprocess
import tempfile

import numpy as np
import tensorflow as tf

import sagemaker
from sagemaker.estimator import Estimator

NUM_CLASSES = 10

sagemaker_session = sagemaker.Session()


def build_image(name, version):
    cmd = 'docker build -t %s --build-arg VERSION=%s -f Dockerfile .' % (name, version)
    subprocess.check_call(shlex.split(cmd))


def push_image(name):
    cmd = 'aws ecr get-login --no-include-email --region us-west-2'
    login = subprocess.check_output(shlex.split(cmd)).strip()

    subprocess.check_call(shlex.split(login))

    cmd = 'docker push %s' % name
    subprocess.check_call(shlex.split(cmd))


def get_tensorflow_version_tag(framework_version, instance_type):
    is_gpu = instance_type[3] == 'p'
    return '%s-gpu' % framework_version if is_gpu else framework_version


def get_image_name(ecr_repository, tensorflow_version_tag):
    return '%s:tensorflow-%s' % (ecr_repository, tensorflow_version_tag)


def upload_channel(channel_name, x, y):
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    file_path = tempfile.mkdtemp()
    np.savez_compressed(os.path.join(file_path, 'cifar-10-npz-compressed.npz'), x=x, y=y)

    return sagemaker_session.upload_data(path=file_path, key_prefix='data/DEMO-keras-cifar10/%s' % channel_name)


def upload_training_data():
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    train_data_location = upload_channel('train', x_train, y_train)
    test_data_location = upload_channel('test', x_test, y_test)

    return {'train': train_data_location, 'test': test_data_location}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ecr-repository', help='ECR repo where images will be pushed',
                        default='add-ecr-repo-here', required=True)
    parser.add_argument('--tf-version', default='latest')
    parser.add_argument('--instance-type', default='local', choices=['local', 'ml.c5.xlarge', 'ml.p2.xlarge'])
    args = parser.parse_args()

    tensorflow_version_tag = get_tensorflow_version_tag(args.tf_version, args.instance_type)

    image_name = get_image_name(args.ecr_repository, tensorflow_version_tag)

    build_image(image_name, tensorflow_version_tag)

    if not args.instance_type.startswith('local'):
        push_image(image_name)

    hyperparameters = dict(batch_size=32, data_augmentation=True, learning_rate=.0001,
                           width_shift_range=.1, height_shift_range=.1)

    estimator = Estimator(image_name, role='SageMakerRole', train_instance_count=1,
                          train_instance_type=args.instance_type, hyperparameters=hyperparameters)

    channels = upload_training_data()

    estimator.fit(channels)
