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
"""SageMaker runtime environment module. This must be kept independent of SageMaker PySDK"""

from __future__ import absolute_import


import logging
import sys
import shlex
import os
import subprocess
import time


class _UTCFormatter(logging.Formatter):
    """Class that overrides the default local time provider in log formatter."""

    converter = time.gmtime


def get_logger():
    """Return a logger with the name 'sagemaker'"""
    sagemaker_logger = logging.getLogger("sagemaker.remote_function")
    if len(sagemaker_logger.handlers) == 0:
        sagemaker_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = _UTCFormatter("%(asctime)s %(name)s %(levelname)-8s %(message)s")
        handler.setFormatter(formatter)
        sagemaker_logger.addHandler(handler)
        # don't stream logs with the root logger handler
        sagemaker_logger.propagate = 0

    return sagemaker_logger


logger = get_logger()


class RuntimeEnvironmentManager:
    """Runtime Environment Manager class to manage runtime environment."""

    def snapshot(self, dependencies: str = None) -> str:
        """Creates snapshot of the user's environment

        If a req.txt or conda.yml file is provided, it verifies their existence and
        returns the local file path
        If ``auto_capture`` is set, this method will take the snapshot of
        user's dependencies installed in the local runtime.
        Current support for ``auto_capture``:
        * conda env, generate a yml file and return it's local path

        Args:
            dependencies (str): Local path where dependencies file exists.

        Returns:
            file path of the existing or generated dependencies file
        """

        # No additional dependencies specified
        if dependencies is None:
            return None

        if dependencies == "auto_capture":
            return self._capture_from_local_runtime()

        # Dependencies specified as either req.txt or conda_env.yml
        if (
            dependencies.endswith(".txt")
            or dependencies.endswith(".yml")
            or dependencies.endswith(".yaml")
        ):
            self._is_file_exists(dependencies)
            return dependencies

        raise ValueError(f'Invalid dependencies provided: "{dependencies}"')

    def _capture_from_local_runtime(self) -> str:
        """Generates dependencies list from the user's local runtime.

        Raises RuntimeEnvironmentError if not able to.

        Currently supports: conda environments
        """

        # Try to capture dependencies from the conda environment, if any.
        conda_env_name = self._get_active_conda_env_name()
        conda_env_prefix = self._get_active_conda_env_prefix()
        if conda_env_name:
            logger.info("Found conda_env_name: '%s'", conda_env_name)
        elif conda_env_prefix:
            logger.info("Found conda_env_prefix: '%s'", conda_env_prefix)
        else:
            raise ValueError("No conda environment seems to be active.")

        if conda_env_name == "base":
            logger.warning(
                "We recommend using an environment other than base to "
                "isolate your project dependencies from conda dependencies"
            )

        local_dependencies_path = os.path.join(os.getcwd(), "env_snapshot.yml")
        self._export_conda_env_from_prefix(conda_env_prefix, local_dependencies_path)

        return local_dependencies_path

    def _get_active_conda_env_prefix(self) -> str:
        """Returns the conda prefix from the set environment variable. None otherwise."""
        return os.getenv("CONDA_PREFIX")

    def _get_active_conda_env_name(self) -> str:
        """Returns the conda environment name from the set environment variable. None otherwise."""
        return os.getenv("CONDA_DEFAULT_ENV")

    def bootstrap(
        self, local_dependencies_file: str, client_python_version: str, conda_env: str = None
    ):
        """Bootstraps the runtime environment by installing the additional dependencies if any.

        Args:
            local_dependencies_file (str): path where dependencies file exists.
            conda_env (str): conda environment to be activated. Default is None.

        Returns: None
        """

        if local_dependencies_file.endswith(".txt"):
            if conda_env:
                self._install_req_txt_in_conda_env(conda_env, local_dependencies_file)
                self._write_conda_env_to_file(conda_env)

            else:
                self._install_requirements_txt(local_dependencies_file, _python_executable())

        elif local_dependencies_file.endswith(".yml") or local_dependencies_file.endswith(".yaml"):
            if conda_env:
                self._update_conda_env(conda_env, local_dependencies_file)
            else:
                conda_env = "sagemaker-runtime-env"
                self._create_conda_env(conda_env, local_dependencies_file)
                self._validate_python_version(client_python_version, conda_env)
            self._write_conda_env_to_file(conda_env)

    def run_pre_exec_script(self, pre_exec_script_path: str):
        """Runs script of pre-execution commands if existing.

        Args:
            pre_exec_script_path (str): Path to pre-execution command script file.
        """
        if os.path.isfile(pre_exec_script_path):
            logger.info("Running pre-execution commands in '%s'", pre_exec_script_path)
            return_code, error_logs = _run_pre_execution_command_script(pre_exec_script_path)

            if return_code:
                error_message = (
                    f"Encountered error while running pre-execution commands. Reason: {error_logs}"
                )
                raise RuntimeEnvironmentError(error_message)
        else:
            logger.info(
                "'%s' does not exist. Assuming no pre-execution commands to run",
                pre_exec_script_path,
            )

    def _is_file_exists(self, dependencies):
        """Check whether the dependencies file exists at the given location.

        Raises error if not
        """
        if not os.path.isfile(dependencies):
            raise ValueError(f'No dependencies file named "{dependencies}" was found.')

    def _install_requirements_txt(self, local_path, python_executable):
        """Install requirements.txt file"""
        cmd = f"{python_executable} -m pip install -r {local_path}"
        logger.info("Running command: '%s' in the dir: '%s' ", cmd, os.getcwd())
        _run_shell_cmd(cmd)
        logger.info("Command %s ran successfully", cmd)

    def _create_conda_env(self, env_name, local_path):
        """Create conda env using conda yml file"""

        cmd = f"{self._get_conda_exe()} env create -n {env_name} --file {local_path}"
        logger.info("Creating conda environment %s using: %s.", env_name, cmd)
        _run_shell_cmd(cmd)
        logger.info("Conda environment %s created successfully.", env_name)

    def _install_req_txt_in_conda_env(self, env_name, local_path):
        """Install requirements.txt in the given conda environment"""

        cmd = f"{self._get_conda_exe()} run -n {env_name} pip install -r {local_path}"
        logger.info("Activating conda env and installing requirements: %s", cmd)
        _run_shell_cmd(cmd)
        logger.info("Requirements installed successfully in conda env %s", env_name)

    def _update_conda_env(self, env_name, local_path):
        """Update conda env using conda yml file"""

        cmd = f"{self._get_conda_exe()} env update -n {env_name} --file {local_path}"
        logger.info("Updating conda env: %s", cmd)
        _run_shell_cmd(cmd)
        logger.info("Conda env %s updated succesfully", env_name)

    def _export_conda_env_from_prefix(self, prefix, local_path):
        """Export the conda env to a conda yml file"""

        cmd = f"{self._get_conda_exe()} env export -p {prefix} --no-builds > {local_path}"
        logger.info("Exporting conda environment: %s", cmd)
        _run_shell_cmd(cmd)
        logger.info("Conda environment %s exported successfully", prefix)

    def _write_conda_env_to_file(self, env_name):
        """Writes conda env to the text file"""

        file_name = "remote_function_conda_env.txt"
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "w") as output_file:
            output_file.write(env_name)

    def _get_conda_exe(self):
        """Checks whether conda or mamba is available to use"""

        if not subprocess.Popen(["which", "mamba"]).wait():
            return "mamba"
        if not subprocess.Popen(["which", "conda"]).wait():
            return "conda"
        raise ValueError("Neither conda nor mamba is installed on the image")

    def _python_version_in_conda_env(self, env_name):
        """Returns python version inside a conda environment"""
        cmd = f"{self._get_conda_exe()} run -n {env_name} python --version"
        try:
            output = (
                subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT)
                .decode("utf-8")
                .strip()
            )
            # convert 'Python 3.7.16' to [3, 7, 16]
            version = output.split("Python ")[1].split(".")
            return version[0] + "." + version[1]
        except subprocess.CalledProcessError as e:
            raise RuntimeEnvironmentError(e.output)

    def _current_python_version(self):
        """Returns the current python version where program is running"""

        return f"{sys.version_info.major}.{sys.version_info.minor}".strip()

    def _validate_python_version(self, client_python_version: str, conda_env: str = None):
        """Validate the python version

        Validates if the python version where remote function runs
        matches the one used on client side.
        """
        if conda_env:
            job_python_version = self._python_version_in_conda_env(conda_env)
        else:
            job_python_version = self._current_python_version()
        if client_python_version.strip() != job_python_version.strip():
            raise RuntimeEnvironmentError(
                f"Python version found in the container is '{job_python_version}' which "
                f"does not match python version '{client_python_version}' on the local client. "
                f"Please make sure that the python version used in the training container "
                f"is same as the local python version."
            )


def _run_and_get_output_shell_cmd(cmd: str) -> str:
    """Run and return the output of the given shell command"""
    return subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT).decode("utf-8")


def _run_pre_execution_command_script(script_path: str):
    """This method runs a given shell script using subprocess

    Raises RuntimeEnvironmentError if the shell script fails
    """
    current_dir = os.path.dirname(script_path)

    process = subprocess.Popen(
        ["/bin/bash", "-eu", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=current_dir,
    )

    _log_output(process)
    error_logs = _log_error(process)
    return_code = process.wait()

    return return_code, error_logs


def _run_shell_cmd(cmd: str):
    """This method runs a given shell command using subprocess

    Raises RuntimeEnvironmentError if the command fails
    """

    process = subprocess.Popen((cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    _log_output(process)
    error_logs = _log_error(process)
    return_code = process.wait()
    if return_code:
        error_message = f"Encountered error while running command '{cmd}'. Reason: {error_logs}"
        raise RuntimeEnvironmentError(error_message)


def _log_output(process):
    """This method takes in Popen process and logs the output of that process"""
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            logger.info(str(line, "UTF-8"))


def _log_error(process):
    """This method takes in Popen process and logs the error of that process.

    Returns those logs as a string
    """

    error_logs = ""
    with process.stderr as pipe:
        for line in iter(pipe.readline, b""):
            error_str = str(line, "UTF-8")
            if "ERROR:" in error_str:
                logger.error(error_str)
            else:
                logger.warning(error_str)
            error_logs = error_logs + error_str

    return error_logs


def _python_executable():
    """Return the real path for the Python executable, if it exists.

    Return RuntimeEnvironmentError otherwise.

    Returns:
        (str): The real path of the current Python executable.
    """
    if not sys.executable:
        raise RuntimeEnvironmentError(
            "Failed to retrieve the path for the Python executable binary"
        )
    return sys.executable


class RuntimeEnvironmentError(Exception):
    """The base exception class for bootstrap env excepitons"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
