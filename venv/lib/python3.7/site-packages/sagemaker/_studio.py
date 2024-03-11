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
"""Provides internal tooling for studio environments."""
from __future__ import absolute_import

import json
import logging

from pathlib import Path

STUDIO_PROJECT_CONFIG = ".sagemaker-code-config"

logger = logging.getLogger(__name__)


def _append_project_tags(tags=None, working_dir=None):
    """Appends the project tag to the list of tags, if it exists.

    Args:
        working_dir: the working directory to start looking.
        tags: the list of tags to append to.

    Returns:
        A possibly extended list of tags that includes the project id.
    """
    path = _find_config(working_dir)
    if path is None:
        return tags

    config = _load_config(path)
    if config is None:
        return tags

    additional_tags = _parse_tags(config)
    if additional_tags is None:
        return tags

    all_tags = tags or []
    additional_tags = [tag for tag in additional_tags if tag not in all_tags]
    all_tags.extend(additional_tags)

    return all_tags


def _find_config(working_dir=None):
    """Gets project config on SageMaker Studio platforms, if it exists.

    Args:
        working_dir: the working directory to start looking.

    Returns:
        The project config path, if it exists. Otherwise None.
    """
    try:
        wd = Path(working_dir) if working_dir else Path.cwd()

        path = None
        while path is None and not wd.match("/"):
            candidate = wd / STUDIO_PROJECT_CONFIG
            if Path.exists(candidate):
                path = candidate
            wd = wd.parent

        return path
    except Exception as e:  # pylint: disable=W0703
        logger.debug("Could not find the studio project config. %s", e)


def _load_config(path):
    """Parse out the projectId attribute if it exists at path.

    Args:
        path: path to project config

    Returns:
        Project config Json, or None if it does not exist.
    """
    try:
        with open(path, "r") as f:
            content = f.read().strip()
        config = json.loads(content)

        return config
    except Exception as e:  # pylint: disable=W0703
        logger.debug("Could not load project config. %s", e)


def _parse_tags(config):
    """Parse out appropriate attributes and formats as tags.

    Args:
        config: project config dict

    Returns:
        List of tags
    """
    try:
        return [
            {"Key": "sagemaker:project-id", "Value": config["sagemakerProjectId"]},
            {"Key": "sagemaker:project-name", "Value": config["sagemakerProjectName"]},
        ]
    except Exception as e:  # pylint: disable=W0703
        logger.debug("Could not parse project config. %s", e)
