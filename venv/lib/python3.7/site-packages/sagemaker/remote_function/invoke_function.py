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
"""An entry point for invoking remote function inside a job."""

from __future__ import absolute_import

import argparse
import sys
import json
import os

import boto3
from sagemaker.experiments.run import Run
from sagemaker.remote_function.job import (
    KEY_EXPERIMENT_NAME,
    KEY_RUN_NAME,
)

from sagemaker.session import Session
from sagemaker.remote_function.errors import handle_error
from sagemaker.remote_function import logging_config


SUCCESS_EXIT_CODE = 0


def _parse_agrs():
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--s3_base_uri", type=str, required=True)
    parser.add_argument("--s3_kms_key", type=str)
    parser.add_argument("--run_in_context", type=str)

    args, _ = parser.parse_known_args()
    return args


def _get_sagemaker_session(region):
    """Get sagemaker session for interacting with AWS or Sagemaker services"""
    boto_session = boto3.session.Session(region_name=region)
    return Session(boto_session=boto_session)


def _load_run_object(run_in_context: str, sagemaker_session: Session) -> Run:
    """Load current run in json string into run object"""
    run_dict = json.loads(run_in_context)
    return Run(
        experiment_name=run_dict.get(KEY_EXPERIMENT_NAME),
        run_name=run_dict.get(KEY_RUN_NAME),
        sagemaker_session=sagemaker_session,
    )


def _execute_remote_function(sagemaker_session, s3_base_uri, s3_kms_key, run_in_context, hmac_key):
    """Execute stored remote function"""
    from sagemaker.remote_function.core.stored_function import StoredFunction

    stored_function = StoredFunction(
        sagemaker_session=sagemaker_session,
        s3_base_uri=s3_base_uri,
        s3_kms_key=s3_kms_key,
        hmac_key=hmac_key,
    )

    if run_in_context:
        run_obj = _load_run_object(run_in_context, sagemaker_session)
        with run_obj:
            stored_function.load_and_invoke()
    else:
        stored_function.load_and_invoke()


def main():
    """Entry point for invoke function script"""

    logger = logging_config.get_logger()

    exit_code = SUCCESS_EXIT_CODE

    try:
        args = _parse_agrs()
        region = args.region
        s3_base_uri = args.s3_base_uri
        s3_kms_key = args.s3_kms_key
        run_in_context = args.run_in_context

        hmac_key = os.getenv("REMOTE_FUNCTION_SECRET_KEY")

        sagemaker_session = _get_sagemaker_session(region)
        _execute_remote_function(
            sagemaker_session=sagemaker_session,
            s3_base_uri=s3_base_uri,
            s3_kms_key=s3_kms_key,
            run_in_context=run_in_context,
            hmac_key=hmac_key,
        )

    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Error encountered while invoking the remote function.")
        exit_code = handle_error(
            error=e,
            sagemaker_session=sagemaker_session,
            s3_base_uri=s3_base_uri,
            s3_kms_key=s3_kms_key,
            hmac_key=hmac_key,
        )
    finally:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
