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

import json
import logging
import os

logger = logging.getLogger(__name__)


def train():
    """Runs the configured SAGEMAKER_TRAINING_COMMAND with all
    the hyperparameters.
    """
    os.chdir("/opt/ml/code")

    user_args = os.environ.get("SM_USER_ARGS", "")
    logger.info("SM_USER_ARGS=%s" % user_args)
    logger.info("All eniron vars=%s" % os.environ)
    hyperparams = " ".join(json.loads(user_args))

    params_blob = os.environ.get("SM_TRAINING_ENV", "")
    params = json.loads(params_blob)
    hyperparams_dict = params["hyperparameters"]

    s3_bucket = hyperparams_dict.get("s3_bucket", "gsaur-test")
    s3_prefix = hyperparams_dict.get("s3_prefix", "sagemaker")

    base_cmd = os.environ.get("SAGEMAKER_TRAINING_COMMAND", "python train.py")
    cmd = "%s %s" % (base_cmd, hyperparams)
    logger.info("Launching training command: %s" % cmd)
    retval = os.system(cmd)
    if retval != 0:
        msg = "Train command returned exit code %s" % retval
        logger.error(msg)
        raise RuntimeError(msg)
