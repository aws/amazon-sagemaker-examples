# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

import os
import logging
import shutil
import subprocess
import json
import re
import errno

from distutils.dir_util import copy_tree
from six.moves.urllib.parse import urlparse

from sagemaker import s3

logger = logging.getLogger(__name__)


def copy_directory_structure(destination_directory, relative_path):
    """Creates intermediate directory structure for relative_path.

    Create all the intermediate directories required for relative_path to
    exist within destination_directory. This assumes that relative_path is a
    directory located within root_dir.

    Examples:
        destination_directory: /tmp/destination relative_path: test/unit/
        will create: /tmp/destination/test/unit

    Args:
        destination_directory (str): root of the destination directory where the
            directory structure will be created.
        relative_path (str): relative path that will be created within
            destination_directory
    """
    full_path = os.path.join(destination_directory, relative_path)
    if os.path.exists(full_path):
        return

    os.makedirs(destination_directory, relative_path)


def move_to_destination(source, destination, job_name, sagemaker_session):
    """Move source to destination.

    Can handle uploading to S3.

    Args:
        source (str): root directory to move
        destination (str): file:// or s3:// URI that source will be moved to.
        job_name (str): SageMaker job name.
        sagemaker_session (sagemaker.Session): a sagemaker_session to interact
            with S3 if needed

    Returns:
        (str): destination URI
    """
    parsed_uri = urlparse(destination)
    if parsed_uri.scheme == "file":
        dir_path = os.path.abspath(parsed_uri.netloc + parsed_uri.path)
        recursive_copy(source, dir_path)
        final_uri = destination
    elif parsed_uri.scheme == "s3":
        bucket = parsed_uri.netloc
        path = s3.s3_path_join(parsed_uri.path, job_name)
        final_uri = s3.s3_path_join("s3://", bucket, path)
        sagemaker_session.upload_data(source, bucket, path)
    else:
        raise ValueError("Invalid destination URI, must be s3:// or file://, got: %s" % destination)

    try:
        shutil.rmtree(source)
    except OSError as exc:
        # on Linux, when docker writes to any mounted volume, it uses the container's user. In most
        # cases this is root. When the container exits and we try to delete them we can't because
        # root owns those files. We expect this to happen, so we handle EACCESS. Any other error
        # we will raise the exception up.
        if exc.errno == errno.EACCES:
            logger.warning("Failed to delete: %s Please remove it manually.", source)
        else:
            logger.error("Failed to delete: %s", source)
            raise

    return final_uri


def recursive_copy(source, destination):
    """A wrapper around distutils.dir_util.copy_tree.

    This won't throw any exception when the source directory does not exist.

    Args:
        source (str): source path
        destination (str): destination path
    """
    if os.path.isdir(source):
        copy_tree(source, destination)


def kill_child_processes(pid):
    """Kill child processes

    Kills all nested child process ids for a specific pid

    Args:
        pid (int): process id
    """
    child_pids = get_child_process_ids(pid)
    for child_pid in child_pids:
        os.kill(child_pid, 15)


def get_child_process_ids(pid):
    """Retrieve all child pids for a certain pid

    Recursively scan each childs process tree and add it to the output

    Args:
        pid (int): process id

    Returns:
        (List[int]): Child process ids
    """
    cmd = f"pgrep -P {pid}".split()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if err:
        return []
    pids = [int(pid) for pid in output.decode("utf-8").split()]
    if pids:
        for child_pid in pids:
            return pids + get_child_process_ids(child_pid)
    else:
        return []


def get_docker_host():
    """Discover remote docker host address (if applicable) or use "localhost"

    Use "docker context inspect" to read current docker host endpoint url,
    url must start with "tcp://"

    Args:

    Returns:
        docker_host (str): Docker host DNS or IP address
    """
    cmd = "docker context inspect".split()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if err:
        return "localhost"
    docker_context_string = output.decode("utf-8")
    docker_context_host_url = json.loads(docker_context_string)[0]["Endpoints"]["docker"]["Host"]
    parsed_url = urlparse(docker_context_host_url)
    if parsed_url.hostname and parsed_url.scheme == "tcp":
        return parsed_url.hostname
    return "localhost"


def get_using_dot_notation(dictionary, keys):
    """Extract `keys` from dictionary where keys is a string in dot notation.

    Args:
        dictionary (Dict)
        keys (str)

    Returns:
        Nested object within dictionary as defined by "keys"

    Raises:
     ValueError if the provided key does not exist in input dictionary
    """
    try:
        if keys is None:
            return dictionary
        split_keys = keys.split(".", 1)
        key = split_keys[0]
        rest = None
        if len(split_keys) > 1:
            rest = split_keys[1]
        bracket_accessors = re.findall(r"\[(.+?)]", key)
        if bracket_accessors:
            pre_bracket_key = key.split("[", 1)[0]
            inner_dict = dictionary[pre_bracket_key]
        else:
            inner_dict = dictionary[key]
        for bracket_accessor in bracket_accessors:
            if (
                bracket_accessor.startswith("'")
                and bracket_accessor.endswith("'")
                or bracket_accessor.startswith('"')
                and bracket_accessor.endswith('"')
            ):
                # key accessor
                inner_key = bracket_accessor[1:-1]
            else:
                # list accessor
                inner_key = int(bracket_accessor)
            inner_dict = inner_dict[inner_key]
        return get_using_dot_notation(inner_dict, rest)
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"{keys} does not exist in input dictionary.")
