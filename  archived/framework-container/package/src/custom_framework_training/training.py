from __future__ import absolute_import

import logging

from sagemaker_training import entry_point, environment

logger = logging.getLogger(__name__)


def train(training_env):
    logger.info("Invoking user training script.")

    entry_point.run(
        training_env.module_dir,
        training_env.user_entry_point,
        training_env.to_cmd_args(),
        training_env.to_env_vars(),
    )


def main():
    training_env = environment.Environment()
    train(training_env)
