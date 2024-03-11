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
"""An entry point for runtime environment. This must be kept independent of SageMaker PySDK"""
from __future__ import absolute_import

import argparse
import sys
import os
import shutil
import pathlib

if __package__ is None or __package__ == "":
    from runtime_environment_manager import RuntimeEnvironmentManager, get_logger
else:
    from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
        RuntimeEnvironmentManager,
        get_logger,
    )

SUCCESS_EXIT_CODE = 0
DEFAULT_FAILURE_CODE = 1

REMOTE_FUNCTION_WORKSPACE = "sm_rf_user_ws"
BASE_CHANNEL_PATH = "/opt/ml/input/data"
FAILURE_REASON_PATH = "/opt/ml/output/failure"
PRE_EXECUTION_SCRIPT_NAME = "pre_exec.sh"
JOB_REMOTE_FUNCTION_WORKSPACE = "sagemaker_remote_function_workspace"


logger = get_logger()


def main():
    """Entry point for bootstrap script"""

    exit_code = DEFAULT_FAILURE_CODE

    try:
        args = _parse_agrs()
        client_python_version = args.client_python_version
        job_conda_env = args.job_conda_env

        conda_env = job_conda_env or os.getenv("SAGEMAKER_JOB_CONDA_ENV")

        RuntimeEnvironmentManager()._validate_python_version(client_python_version, conda_env)

        _bootstrap_runtime_environment(client_python_version, conda_env)

        exit_code = SUCCESS_EXIT_CODE
    except Exception as e:  # pylint: disable=broad-except
        logger.exception("Error encountered while bootstrapping runtime environment: %s", e)

        _write_failure_reason_file(str(e))
    finally:
        sys.exit(exit_code)


def _bootstrap_runtime_environment(
    client_python_version: str,
    conda_env: str = None,
):
    """Bootstrap runtime environment for remote function invocation

    Args:
        conda_env (str): conda environment to be activated. Default is None.
    """
    workspace_archive_dir_path = os.path.join(BASE_CHANNEL_PATH, REMOTE_FUNCTION_WORKSPACE)

    if not os.path.exists(workspace_archive_dir_path):
        logger.info(
            "Directory '%s' does not exist. Assuming no dependencies to bootstrap.",
            workspace_archive_dir_path,
        )
        return

    # Unpack user workspace archive first.
    workspace_archive_path = os.path.join(workspace_archive_dir_path, "workspace.zip")
    if not os.path.isfile(workspace_archive_path):
        logger.info(
            "Workspace archive '%s' does not exist. Assuming no dependencies to bootstrap.",
            workspace_archive_dir_path,
        )
        return

    workspace_unpack_dir = pathlib.Path(os.getcwd()).absolute()
    shutil.unpack_archive(filename=workspace_archive_path, extract_dir=workspace_unpack_dir)
    logger.info("Successfully unpacked workspace archive at '%s'.", workspace_unpack_dir)
    workspace_unpack_dir = pathlib.Path(workspace_unpack_dir, JOB_REMOTE_FUNCTION_WORKSPACE)

    # Handle pre-execution commands
    path_to_pre_exec_script = os.path.join(workspace_unpack_dir, PRE_EXECUTION_SCRIPT_NAME)
    RuntimeEnvironmentManager().run_pre_exec_script(pre_exec_script_path=path_to_pre_exec_script)

    # Handle dependencies file.
    dependencies_file = None
    for file in os.listdir(workspace_unpack_dir):
        if file.endswith(".txt") or file.endswith(".yml") or file.endswith(".yaml"):
            dependencies_file = os.path.join(workspace_unpack_dir, file)
            break

    if dependencies_file:
        RuntimeEnvironmentManager().bootstrap(
            local_dependencies_file=dependencies_file,
            conda_env=conda_env,
            client_python_version=client_python_version,
        )
    else:
        logger.info(
            "Did not find any dependency file in workspace directory at '%s'."
            " Assuming no additional dependencies to install.",
            workspace_archive_dir_path,
        )


def _write_failure_reason_file(failure_msg):
    """Create a file 'failure' with failure reason written if bootstrap runtime env failed.

    See: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    Args:
        failure_msg: The content of file to be written.
    """
    if not os.path.exists(FAILURE_REASON_PATH):
        with open(FAILURE_REASON_PATH, "w") as f:
            f.write("RuntimeEnvironmentError: " + failure_msg)


def _parse_agrs():
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_conda_env", type=str)
    parser.add_argument("--client_python_version")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main()
