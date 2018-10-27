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

from __future__ import absolute_import

import base64
import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

import boto3

IMAGE_TEMPLATE = "{account}.dkr.ecr.{region}.amazonaws.com/{image_name}:{version}"

DOCKERFILE_TEMPLATE = """
FROM {0}

RUN apt-get update && apt-get install -y --no-install-recommends git

RUN git clone https://github.com/mvsusp/sagemaker-containers.git \
-b mvs-sagemaker-containers-train-improvements && cd sagemaker-containers \
&& pip install . --quiet --disable-pip-version-check

COPY {1} /opt/ml/code

ENV PYTHONPATH /opt/ml/code:$PYTHONPATH
ENV SAGEMAKER_TRAINING_MODULE {2}"""


def build(base_image, entrypoint, source_dir, tag):

    _ecr_login_if_needed(base_image)

    with _tmpdir() as tmp:
        shutil.copytree(source_dir, os.path.join(tmp, 'src'))

        dockerfile = DOCKERFILE_TEMPLATE.format(base_image, 'src', entrypoint[:-3])

        with open(os.path.join(tmp, 'Dockerfile'), mode='w') as f:
            print(dockerfile)
            f.write(dockerfile)

        _execute(['docker', 'build', '-t', tag, tmp])


def push(tag, aws_account=None, aws_region=None):
    session = boto3.Session()

    aws_account = aws_account or session.client("sts").get_caller_identity()['Account']
    aws_region = aws_region or session.region_name

    ecr_repo = '%s.dkr.ecr.%s.amazonaws.com' % (aws_account, aws_region)

    print("Pushing docker image to ECR repository %s/%s\n" % (ecr_repo, tag))
    repository_name, version = tag.split(':')

    ecr_client = boto3.client('ecr', region_name=aws_region)
    try:

        ecr_client.describe_repositories(repositoryNames=[repository_name])['repositories']

    except ecr_client.exceptions.RepositoryNotFoundException:

        ecr_client.create_repository(repositoryName=repository_name)

        print("Created new ECR repository: %s" % repository_name)

    _ecr_login(ecr_client, aws_account)

    ecr_tag = '%s/%s' % (ecr_repo, tag)

    _execute(['docker', 'tag', tag, ecr_tag])

    _execute(['docker', 'push', ecr_tag])

    return ecr_tag


def _ecr_login(ecr_client, aws_account):
    auth = ecr_client.get_authorization_token(registryIds=[aws_account])
    authorization_data = auth['authorizationData'][0]

    raw_token = base64.b64decode(authorization_data['authorizationToken'])
    token = raw_token.decode('utf-8').strip('AWS:')
    ecr_url = auth['authorizationData'][0]['proxyEndpoint']

    cmd = ['docker', 'login', '-u', 'AWS', '-p', token, ecr_url]
    _execute(cmd)


def _ecr_login_if_needed(image):
    ecr_client = boto3.client('ecr')

    # Only ECR images need login
    if not ('dkr.ecr' in image and 'amazonaws.com' in image):
        return

    # do we have the image?
    if _check_output('docker images -q %s' % image).strip():
        return

    aws_account = image.split('.')[0]
    _ecr_login(ecr_client, aws_account)


@contextlib.contextmanager
def _tmpdir(suffix='', prefix='tmp', dir=None):  # type: (str, str, str) -> None
    """Create a temporary directory with a context manager. The file is deleted when the context exits.

    The prefix, suffix, and dir arguments are the same as for mkstemp().

    Args:
        suffix (str):  If suffix is specified, the file name will end with that suffix, otherwise there will be no
                        suffix.
        prefix (str):  If prefix is specified, the file name will begin with that prefix; otherwise,
                        a default prefix is used.
        dir (str):  If dir is specified, the file will be created in that directory; otherwise, a default directory is
                        used.
    Returns:
        str: path to the directory
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    yield tmp
    shutil.rmtree(tmp)


def _execute(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        _stream_output(process)
    except RuntimeError as e:
        # _stream_output() doesn't have the command line. We will handle the exception
        # which contains the exit code and append the command line to it.
        msg = "Failed to run: %s, %s" % (command, str(e))
        raise RuntimeError(msg)


def _stream_output(process):
    """Stream the output of a process to stdout

    This function takes an existing process that will be polled for output. Only stdout
    will be polled and sent to sys.stdout.

    Args:
        process(subprocess.Popen): a process that has been started with
            stdout=PIPE and stderr=STDOUT

    Returns (int): process exit code
    """
    exit_code = None

    while exit_code is None:
        stdout = process.stdout.readline().decode("utf-8")
        sys.stdout.write(stdout)
        exit_code = process.poll()

    if exit_code != 0:
        raise RuntimeError("Process exited with code: %s" % exit_code)


def _check_output(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    success = True
    try:
        output = subprocess.check_output(cmd, *popenargs, **kwargs)
    except subprocess.CalledProcessError as e:
        output = e.output
        success = False

    output = output.decode("utf-8")
    if not success:
        print("Command output: %s" % output)
        raise Exception("Failed to run %s" % ",".join(cmd))

    return output
