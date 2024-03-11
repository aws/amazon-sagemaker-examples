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
"""This file contains util functions for the sagemaker Defaults Config.

These utils may be used inside or outside the config module.
"""
from __future__ import absolute_import

import logging
import sys


def get_sagemaker_config_logger():
    """Return a logger with the name 'sagemaker.config'

    If the logger to be returned has no level or handlers set, this will get level and handler
    attributes. (So if the SDK user has setup loggers in a certain way, that setup will not be
    changed by this function.) It is safe to make repeat calls to this function.
    """
    sagemaker_config_logger = logging.getLogger("sagemaker.config")
    sagemaker_logger = logging.getLogger("sagemaker")

    if sagemaker_config_logger.level == logging.NOTSET:
        sagemaker_config_logger.setLevel(logging.INFO)

    # check sagemaker_logger here as well, so that if handlers were set for the parent logger
    # already, we dont change behavior for the child logger
    if len(sagemaker_config_logger.handlers) == 0 and len(sagemaker_logger.handlers) == 0:
        # use sys.stdout so logs dont show up with a red background in a notebook
        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter("%(name)s %(levelname)-4s - %(message)s")
        handler.setFormatter(formatter)
        sagemaker_config_logger.addHandler(handler)

        # if a handler is being added, we dont want the root handler to also process the same events
        sagemaker_config_logger.propagate = False

    return sagemaker_config_logger


def _log_sagemaker_config_single_substitution(source_value, config_value, config_key_path: str):
    """Informs the SDK user whether a config value was present and automatically substituted

    Args:
        source_value: The value that will be used if it exists. Usually, this is user-provided
            input to a Class or to a session.py method, or None if no input was provided.
        config_value: The value fetched from sagemaker_config. If it exists, this is the value that
            will be used if direct_input is None.
        config_key_path: A string denoting the path of keys that point to the config value in the
            sagemaker_config.

    Returns:
        None. Logs information to the "sagemaker.config" logger.
    """
    logger = get_sagemaker_config_logger()

    if config_value is not None:

        if source_value is None:
            # Sagemaker Config value is going to be used. By default the user should know about
            # this scenario because the behavior they expect could change because of a new config
            # value being injected in.
            # However, it may not be safe to log ARNs to stdout by default. We can include more
            # diagnostic info if the user enabled DEBUG logs though.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Applied value\n  config key = %s\n  config value that will be used = %s",
                    config_key_path,
                    config_value,
                )
            else:
                logger.info(
                    "Applied value from config key = %s",
                    config_key_path,
                )

        # The cases below here are logged as just debug statements because this info can be useful
        # when debugging the config, but should not affect current behavior with/without the config.

        elif source_value is not None and config_value == source_value:
            # Sagemaker Config had a value defined that is NOT going to be used here.
            # Either (1) the config value was already fetched and applied earlier, or
            # (2) the user happened to pass in the same value.
            logger.debug(
                (
                    "Skipped value\n"
                    "  config key = %s\n"
                    "  config value = %s\n"
                    "  source value that will be used = %s"
                ),
                config_key_path,
                config_value,
                source_value,
            )
        elif source_value is not None and config_value != source_value:
            # Sagemaker Config had a value defined that is NOT going to be used
            # and the config value has not already been applied earlier (we know because the values
            # are different).
            logger.debug(
                (
                    "Skipped value\n"
                    "  config key = %s\n"
                    "  config value = %s\n"
                    "  source value that will be used = %s",
                ),
                config_key_path,
                config_value,
                source_value,
            )
    else:
        # nothing was specified in the config and nothing is being automatically applied
        logger.debug("Skipped value because no value defined\n  config key = %s", config_key_path)


def _log_sagemaker_config_merge(
    source_value=None,
    config_value=None,
    merged_source_and_config_value=None,
    config_key_path: str = None,
):
    """Informs the SDK user whether a config value was present and automatically substituted

    Args:
        source_value: The dict or object that would be used if no default values existed. Usually,
            this is user-provided input to a Class or to a session.py method, or None if no input
            was provided.
        config_value: The dict or object fetched from sagemaker_config. If it exists, this is the
            value that will be used if source_value is None.
        merged_source_and_config_value: The value that results from the merging of source_value and
            original_config_value. This will be the value used.
        config_key_path: A string denoting the path of keys that point to the config value in the
            sagemaker_config.

    Returns:
        None. Logs information to the "sagemaker.config" logger.
    """
    logger = get_sagemaker_config_logger()

    if config_value:

        if source_value != merged_source_and_config_value:
            # Sagemaker Config value(s) were used and affected the final object/dictionary. By
            # default the user should know about this scenario because the behavior they expect
            # could change because of new config values being injected in.
            # However, it may not be safe to log ARNs to stdout by default. We can include more
            # diagnostic info if the user enabled DEBUG logs though.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    (
                        "Applied value(s)\n"
                        "  config key = %s\n"
                        "  config value = %s\n"
                        "  source value = %s\n"
                        "  combined value that will be used = %s"
                    ),
                    config_key_path,
                    config_value,
                    source_value,
                    merged_source_and_config_value,
                )
            else:
                logger.info(
                    "Applied value(s) from config key = %s",
                    config_key_path,
                )

        # The cases below here are logged as just debug statements because this info can be useful
        # when debugging the config, but should not affect current behavior with/without the config.

        else:
            # Sagemaker Config had a value defined that is NOT going to be used here.
            # Either (1) the config value was already fetched and applied earlier, or
            # (2) the user happened to pass in the same values.
            logger.debug(
                (
                    "Skipped value(s)\n"
                    "  config key = %s\n"
                    "  config value = %s\n"
                    "  source value that will be used = %s"
                ),
                config_key_path,
                config_value,
                merged_source_and_config_value,
            )

    else:
        # nothing was specified in the config and nothing is being automatically applied
        logger.debug("Skipped value because no value defined\n  config key = %s", config_key_path)
