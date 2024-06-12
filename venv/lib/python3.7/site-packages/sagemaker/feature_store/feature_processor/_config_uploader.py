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
"""Contains classes for preparing and uploading configs for a scheduled feature processor."""
from __future__ import absolute_import
from typing import Callable, Dict, Tuple, List

import attr

from sagemaker import Session
from sagemaker.feature_store.feature_processor._constants import (
    SPARK_JAR_FILES_PATH,
    SPARK_PY_FILES_PATH,
    SPARK_FILES_PATH,
    S3_DATA_DISTRIBUTION_TYPE,
)
from sagemaker.inputs import TrainingInput
from sagemaker.remote_function.core.stored_function import StoredFunction
from sagemaker.remote_function.job import (
    _prepare_and_upload_dependencies,
    _prepare_and_upload_runtime_scripts,
    _JobSettings,
    RUNTIME_SCRIPTS_CHANNEL_NAME,
    REMOTE_FUNCTION_WORKSPACE,
    SPARK_CONF_WORKSPACE,
    _prepare_and_upload_spark_dependent_files,
)
from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentManager,
)
from sagemaker.remote_function.spark_config import SparkConfig
from sagemaker.s3 import s3_path_join


@attr.s
class ConfigUploader:
    """Prepares and uploads customer provided configs to S3"""

    remote_decorator_config: _JobSettings = attr.ib()
    runtime_env_manager: RuntimeEnvironmentManager = attr.ib()

    def prepare_step_input_channel_for_spark_mode(
        self, func: Callable, s3_base_uri: str, sagemaker_session: Session
    ) -> Tuple[Dict, Dict]:
        """Prepares input channels for SageMaker Pipeline Step."""
        self._prepare_and_upload_callable(func, s3_base_uri, sagemaker_session)
        bootstrap_scripts_s3uri = self._prepare_and_upload_runtime_scripts(
            self.remote_decorator_config.spark_config,
            s3_base_uri,
            self.remote_decorator_config.s3_kms_key,
            sagemaker_session,
        )
        dependencies_list_path = self.runtime_env_manager.snapshot(
            self.remote_decorator_config.dependencies
        )
        user_dependencies_s3uri = self._prepare_and_upload_dependencies(
            dependencies_list_path,
            self.remote_decorator_config.include_local_workdir,
            self.remote_decorator_config.pre_execution_commands,
            self.remote_decorator_config.pre_execution_script,
            s3_base_uri,
            self.remote_decorator_config.s3_kms_key,
            sagemaker_session,
        )

        (
            submit_jars_s3_paths,
            submit_py_files_s3_paths,
            submit_files_s3_path,
            config_file_s3_uri,
        ) = self._prepare_and_upload_spark_dependent_files(
            self.remote_decorator_config.spark_config,
            s3_base_uri,
            self.remote_decorator_config.s3_kms_key,
            sagemaker_session,
        )

        input_data_config = {
            RUNTIME_SCRIPTS_CHANNEL_NAME: TrainingInput(
                s3_data=bootstrap_scripts_s3uri,
                s3_data_type="S3Prefix",
                distribution=S3_DATA_DISTRIBUTION_TYPE,
            )
        }
        if user_dependencies_s3uri:
            input_data_config[REMOTE_FUNCTION_WORKSPACE] = TrainingInput(
                s3_data=s3_path_join(s3_base_uri, REMOTE_FUNCTION_WORKSPACE),
                s3_data_type="S3Prefix",
                distribution=S3_DATA_DISTRIBUTION_TYPE,
            )

        if config_file_s3_uri:
            input_data_config[SPARK_CONF_WORKSPACE] = TrainingInput(
                s3_data=config_file_s3_uri,
                s3_data_type="S3Prefix",
                distribution=S3_DATA_DISTRIBUTION_TYPE,
            )

        return input_data_config, {
            SPARK_JAR_FILES_PATH: submit_jars_s3_paths,
            SPARK_PY_FILES_PATH: submit_py_files_s3_paths,
            SPARK_FILES_PATH: submit_files_s3_path,
        }

    def _prepare_and_upload_callable(
        self, func: Callable, s3_base_uri: str, sagemaker_session: Session
    ) -> None:
        """Prepares and uploads callable to S3"""
        stored_function = StoredFunction(
            sagemaker_session=sagemaker_session,
            s3_base_uri=s3_base_uri,
            hmac_key=self.remote_decorator_config.environment_variables[
                "REMOTE_FUNCTION_SECRET_KEY"
            ],
            s3_kms_key=self.remote_decorator_config.s3_kms_key,
        )
        stored_function.save(func)

    def _prepare_and_upload_dependencies(
        self,
        local_dependencies_path: str,
        include_local_workdir: bool,
        pre_execution_commands: List[str],
        pre_execution_script_local_path: str,
        s3_base_uri: str,
        s3_kms_key: str,
        sagemaker_session: Session,
    ) -> str:
        """Upload the training step dependencies to S3 if present"""
        return _prepare_and_upload_dependencies(
            local_dependencies_path=local_dependencies_path,
            include_local_workdir=include_local_workdir,
            pre_execution_commands=pre_execution_commands,
            pre_execution_script_local_path=pre_execution_script_local_path,
            s3_base_uri=s3_base_uri,
            s3_kms_key=s3_kms_key,
            sagemaker_session=sagemaker_session,
        )

    def _prepare_and_upload_runtime_scripts(
        self,
        spark_config: SparkConfig,
        s3_base_uri: str,
        s3_kms_key: str,
        sagemaker_session: Session,
    ) -> str:
        """Copy runtime scripts to a folder and upload to S3"""
        return _prepare_and_upload_runtime_scripts(
            spark_config=spark_config,
            s3_base_uri=s3_base_uri,
            s3_kms_key=s3_kms_key,
            sagemaker_session=sagemaker_session,
        )

    def _prepare_and_upload_spark_dependent_files(
        self,
        spark_config: SparkConfig,
        s3_base_uri: str,
        s3_kms_key: str,
        sagemaker_session: Session,
    ) -> Tuple:
        """Upload the spark dependencies to S3 if present"""
        if not spark_config:
            return None, None, None, None

        return _prepare_and_upload_spark_dependent_files(
            spark_config=spark_config,
            s3_base_uri=s3_base_uri,
            s3_kms_key=s3_kms_key,
            sagemaker_session=sagemaker_session,
        )
