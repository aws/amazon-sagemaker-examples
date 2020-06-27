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
import time
import shlex
import shutil
import subprocess
import sys
import tempfile

import boto3
import json

IMAGE_TEMPLATE = "{account}.dkr.ecr.{region}.amazonaws.com/{image_name}:{version}"


def build_and_push_docker_image(repository_name, dockerfile='Dockerfile', build_args={}):
    """Builds a docker image from the specified dockerfile, and pushes it to
    ECR.  Handles things like ECR login, creating the repository.

    Returns the name of the created docker image in ECR
    """
    base_image = _find_base_image_in_dockerfile(dockerfile)
    _ecr_login_if_needed(base_image)
    _build_from_dockerfile(repository_name, dockerfile, build_args)
    ecr_tag = push(repository_name)
    return ecr_tag


def _build_from_dockerfile(repository_name, dockerfile='Dockerfile', build_args={}):
    build_cmd = ['docker', 'build', '-t', repository_name, '-f', dockerfile, '.']
    for k,v in build_args.items():
        build_cmd += ['--build-arg', '%s=%s' % (k,v)]

    print("Building docker image %s from %s" % (repository_name, dockerfile))
    _execute(build_cmd)
    print("Done building docker image %s" % repository_name)
    

def _find_base_image_in_dockerfile(dockerfile):
    dockerfile_lines = open(dockerfile).readlines()
    from_line = list(filter(lambda line: line.startswith("FROM "), dockerfile_lines))[0].rstrip()
    base_image = from_line[5:]
    return base_image


def push(tag, aws_account=None, aws_region=None):
    """
    Push the builded tag to ECR.

    Args:
        tag (string): tag which you named your algo
        aws_account (string): aws account of the ECR repo
        aws_region (string): aws region where the repo is located

    Returns:
        (string): ECR repo image that was pushed
    """
    session = boto3.Session()
    aws_account = aws_account or session.client("sts").get_caller_identity()['Account']
    aws_region = aws_region or session.region_name
    try:
        repository_name, version = tag.split(':')
    except ValueError:  # split failed because no :
        repository_name = tag
        version = "latest"
    ecr_client = session.client('ecr', region_name=aws_region)

    _create_ecr_repo(ecr_client, repository_name)
    _ecr_login(ecr_client, aws_account)
    ecr_tag = _push(aws_account, aws_region, tag)

    return ecr_tag


def _push(aws_account, aws_region, tag):
    ecr_repo = '%s.dkr.ecr.%s.amazonaws.com' % (aws_account, aws_region)
    ecr_tag = '%s/%s' % (ecr_repo, tag)
    _execute(['docker', 'tag', tag, ecr_tag])
    print("Pushing docker image to ECR repository %s/%s\n" % (ecr_repo, tag))
    _execute(['docker', 'push', ecr_tag])
    print("Done pushing %s" % ecr_tag)
    return ecr_tag


def _create_ecr_repo(ecr_client, repository_name):
    """
    Create the repo if it doesn't already exist.
    """
    try:
        ecr_client.create_repository(repositoryName=repository_name)
        print("Created new ECR repository: %s" % repository_name)
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print("ECR repository already exists: %s" % repository_name)


def _ecr_login(ecr_client, aws_account):
    auth = ecr_client.get_authorization_token(registryIds=[aws_account])
    authorization_data = auth['authorizationData'][0]

    raw_token = base64.b64decode(authorization_data['authorizationToken'])
    token = raw_token.decode('utf-8').strip('AWS:')
    ecr_url = auth['authorizationData'][0]['proxyEndpoint']

    cmd = ['docker', 'login', '-u', 'AWS', '-p', token, ecr_url]
    _execute(cmd, quiet=True)
    print("Logged into ECR")


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


def _execute(command, quiet=False):
    if not quiet:
        print("$ %s" % ' '.join(command))
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
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
