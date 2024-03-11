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
"""SageMaker remote function client."""
from __future__ import absolute_import

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time
import threading
from typing import Dict, List, Tuple, Any
import functools
import itertools
import inspect

from botocore.exceptions import ClientError
from sagemaker.exceptions import UnexpectedStatusException
from sagemaker.experiments._run_context import _RunContext

import sagemaker.remote_function.core.serialization as serialization
from sagemaker.remote_function.errors import RemoteFunctionError, ServiceError, DeserializationError
from sagemaker.remote_function.core.stored_function import RESULTS_FOLDER, EXCEPTION_FOLDER
from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentError,
)

from sagemaker.session import Session
from sagemaker.s3 import s3_path_join
from sagemaker.remote_function.job import _JobSettings, _Job, _RunInfo
from sagemaker.remote_function import logging_config
from sagemaker.utils import name_from_base, base_from_name
from sagemaker.remote_function.spark_config import SparkConfig

_API_CALL_LIMIT = {
    "SubmittingIntervalInSecs": 1,
    "MinBatchPollingIntervalInSecs": 10,
    "PollingIntervalInSecs": 0.5,
}

# Possible future states.
_PENDING = "PENDING"
_RUNNING = "RUNNING"
# The future was cancelled by the user...
_CANCELLED = "CANCELLED"
_FINISHED = "FINISHED"

logger = logging_config.get_logger()


def remote(
    _func=None,
    *,
    dependencies: str = None,
    pre_execution_commands: List[str] = None,
    pre_execution_script: str = None,
    environment_variables: Dict[str, str] = None,
    image_uri: str = None,
    include_local_workdir: bool = False,
    instance_count: int = 1,
    instance_type: str = None,
    job_conda_env: str = None,
    job_name_prefix: str = None,
    keep_alive_period_in_seconds: int = 0,
    max_retry_attempts: int = 1,
    max_runtime_in_seconds: int = 24 * 60 * 60,
    role: str = None,
    s3_kms_key: str = None,
    s3_root_uri: str = None,
    sagemaker_session: Session = None,
    security_group_ids: List[str] = None,
    subnets: List[str] = None,
    tags: List[Tuple[str, str]] = None,
    volume_kms_key: str = None,
    volume_size: int = 30,
    encrypt_inter_container_traffic: bool = None,
    spark_config: SparkConfig = None,
    use_spot_instances=False,
    max_wait_time_in_seconds=None,
):
    """Decorator for running the annotated function as a SageMaker training job.

    This decorator wraps the annotated code and runs it as a new SageMaker job synchronously
    with the provided runtime settings.

    If a parameter value is not set, the decorator first looks up the value from the SageMaker
    configuration file. If no value is specified in the configuration file or no configuration file
    is found, the decorator selects the default as specified below. For more information, see
    `Configuring and using defaults with the SageMaker Python SDK <https://sagemaker.readthedocs.io/
    en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk>`_.

    Args:
        _func (Optional): A Python function to run as a SageMaker training job.

        dependencies (str): Either the path to a dependencies file or the reserved keyword
          ``auto_capture``. Defaults to ``None``.
          If ``dependencies`` is provided, the value must be one of the following:

          * A path to a conda environment.yml file. The following conditions apply.

            * If job_conda_env is set, then the conda environment is updated by installing
              dependencies from the yaml file and the function is invoked within that
              conda environment. For this to succeed, the specified conda environment must
              already exist in the image.
            * If the environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image, then the
              conda environment is updated by installing dependencies from the yaml file and the
              function is invoked within that conda environment. For this to succeed, the
              conda environment name must already be set in ``SAGEMAKER_JOB_CONDA_ENV``, and
              ``SAGEMAKER_JOB_CONDA_ENV`` must already exist in the image.
            * If none of the previous conditions are met, a new conda environment named
              ``sagemaker-runtime-env`` is created and the function annotated with the remote
              decorator is invoked in that conda environment.

          * A path to a requirements.txt file. The following conditions apply.

            * If ``job_conda_env`` is set in the remote decorator, dependencies are installed
              within that conda environment and the function annotated with the remote decorator
              is invoked in the same conda environment. For this to succeed, the specified
              conda environment must already exist in the image.
            * If an environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image,
              dependencies are installed within that conda environment and the function annotated
              with the remote decorator is invoked in the same. For this to succeed, the conda
              environment name must already be set in ``SAGEMAKER_JOB_CONDA_ENV``, and
              ``SAGEMAKER_JOB_CONDA_ENV`` must already exist in the image.
            * If none of the above conditions are met, conda is not used. Dependencies are
              installed at the system level, without any virtual environment, and the function
              annotated with the remote decorator is invoked using the Python runtime available
              in the system path.

          * The parameter dependencies is set to ``auto_capture``. SageMaker will automatically
            generate an env_snapshot.yml corresponding to the current active conda environment’s
            snapshot. You do not need to provide a dependencies file. The following conditions
            apply:

            * You must run the remote function within an active conda environment.
            * When installing the dependencies on the training job, the same conditions as when
              dependencies is set to a path to a conda environment file apply. These conditions are
              as follows:

              * If job_conda_env is set, then the conda environment is updated by installing
                dependencies from the yaml file and the function is invoked within that
                conda environment. For this to succeed, the specified conda environment must
                already exist in the image.
              * If the environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image, then
                the conda environment is updated by installing dependencies from the yaml file
                and the function is invoked within that conda environment. For this to
                succeed, the conda environment name must already be set in
                ``SAGEMAKER_JOB_CONDA_ENV``, and ``SAGEMAKER_JOB_CONDA_ENV`` must already exist
                in the image.
              * If none of the previous conditions are met, a new conda environment with name
                ``sagemaker-runtime-env`` is created and the function annotated with the
                remote decorator is invoked in that conda environment.

          * ``None``. SageMaker will assume that there are no dependencies to install while
            executing the remote annotated function in the training job.

        pre_execution_commands (List[str]): List of commands to be executed prior to executing
          remote function. Only one of ``pre_execution_commands`` or ``pre_execution_script``
          can be specified at the same time. Defaults to None.

        pre_execution_script (str): Path to script file to be executed prior to executing
          remote function. Only one of ``pre_execution_commands`` or ``pre_execution_script``
          can be specified at the same time. Defaults to None.

        environment_variables (Dict): The environment variables used inside the decorator function.
          Defaults to ``None``.

        image_uri (str): The universal resource identifier (URI) location of a Docker image on
          Amazon Elastic Container Registry (ECR). Defaults to the following based on where the SDK
          is running:

            * For users who specify ``spark_config`` and want to run the function in a Spark
              application, the ``image_uri`` should be ``None``. A SageMaker Spark image will
              be used for training, otherwise a ``ValueError`` is thrown.
            * For users on SageMaker Studio notebooks, the image used as the kernel image for the
              notebook is used.
            * For other users, it is resolved to base python image with the same python version
              as the environment running the local code.

          If no compatible image is found, a ValueError is thrown.

        include_local_workdir (bool): A flag to indicate that the remote function should include
          local directories. Set to ``True`` if the remote function code imports local modules and
          methods that are not available via PyPI or conda. Default value is ``False``.

        instance_count (int): The number of instances to use. Defaults to 1.
          NOTE: Remote function does not support instance_count > 1 for non Spark jobs.

        instance_type (str): The Amazon Elastic Compute Cloud (EC2) instance type to use to run
          the SageMaker job. e.g. ml.c4.xlarge. If not provided, a ValueError is thrown.

        job_conda_env (str): The name of the conda environment to activate during job's runtime.
          Defaults to ``None``.

        job_name_prefix (str): The prefix used used to create the underlying SageMaker job.

        keep_alive_period_in_seconds (int): The duration in seconds to retain and reuse provisioned
          infrastructure after the completion of a training job, also known as SageMaker managed
          warm pools. The use of warmpools reduces the latency time spent to provision new
          resources. The default value for ``keep_alive_period_in_seconds`` is 0.
          NOTE: Additional charges associated with warm pools may apply. Using this parameter also
          activates a new persistent cache feature, which will further reduce job start up
          latency than over using SageMaker managed warm pools alone by caching the package source
          downloaded in the previous runs.

        max_retry_attempts (int): The max number of times the job is retried on
          ``InternalServerFailure`` Error from SageMaker service. Defaults to 1.

        max_runtime_in_seconds (int): The upper limit in seconds to be used for training. After
          this specified amount of time, SageMaker terminates the job regardless of its current
          status. Defaults to 1 day or (86400 seconds).

        role (str): The IAM role (either name or full ARN) used to run your SageMaker training
          job. Defaults to:

          * the SageMaker default IAM role if the SDK is running in SageMaker Notebooks or
            SageMaker Studio Notebooks.
          * if not above, a ValueError is be thrown.

        s3_kms_key (str): The key used to encrypt the input and output data. Default to ``None``.

        s3_root_uri (str): The root S3 folder to which the code archives and data are
          uploaded to. Defaults to ``s3://<sagemaker-default-bucket>``.

        sagemaker_session (sagemaker.session.Session): The underlying SageMaker session to which
          SageMaker service calls are delegated to (default: None). If not provided, one is created
          using a default configuration chain.

        security_group_ids (List[str): A list of security group IDs. Defaults to ``None`` and the
          training job is created without VPC config.

        subnets (List[str): A list of subnet IDs. Defaults to ``None`` and the job is created
          without VPC config.

        tags (List[Tuple[str, str]): A list of tags attached to the job. Defaults to ``None`` and
          the training job is created without tags.

        volume_kms_key (str): An Amazon Key Management Service (KMS) key used to encrypt an
          Amazon Elastic Block Storage (EBS) volume attached to the training instance. Defaults to
          ``None``.

        volume_size (int): The size in GB of the storage volume for storing input and output data
          during training. Defaults to ``30``.

        encrypt_inter_container_traffic (bool): A flag that specifies whether traffic between
          training containers is encrypted for the training job. Defaults to ``False``.

        spark_config (SparkConfig): Configurations to the Spark application that runs on
          Spark image. If ``spark_config`` is specified, a SageMaker Spark image uri
          will be used for training. Note that ``image_uri`` can not be specified at the
          same time otherwise a ``ValueError`` is thrown. Defaults to ``None``.

        use_spot_instances (bool): Specifies whether to use SageMaker Managed Spot instances for
          training. If enabled then the ``max_wait_time_in_seconds`` arg should also be set.
          Defaults to ``False``.

        max_wait_time_in_seconds (int): Timeout in seconds waiting for spot training job.
          After this amount of time Amazon SageMaker will stop waiting for managed spot training
          job to complete. Defaults to ``None``.
    """

    def _remote(func):

        job_settings = _JobSettings(
            dependencies=dependencies,
            pre_execution_commands=pre_execution_commands,
            pre_execution_script=pre_execution_script,
            environment_variables=environment_variables,
            image_uri=image_uri,
            include_local_workdir=include_local_workdir,
            instance_count=instance_count,
            instance_type=instance_type,
            job_conda_env=job_conda_env,
            job_name_prefix=job_name_prefix,
            keep_alive_period_in_seconds=keep_alive_period_in_seconds,
            max_retry_attempts=max_retry_attempts,
            max_runtime_in_seconds=max_runtime_in_seconds,
            role=role,
            s3_kms_key=s3_kms_key,
            s3_root_uri=s3_root_uri,
            sagemaker_session=sagemaker_session,
            security_group_ids=security_group_ids,
            subnets=subnets,
            tags=tags,
            volume_kms_key=volume_kms_key,
            volume_size=volume_size,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            spark_config=spark_config,
            use_spot_instances=use_spot_instances,
            max_wait_time_in_seconds=max_wait_time_in_seconds,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            if instance_count > 1 and not spark_config:
                raise ValueError(
                    "Remote function do not support training on multi instances. "
                    + "Please provide instance_count = 1"
                )

            RemoteExecutor._validate_submit_args(func, *args, **kwargs)

            job = _Job.start(job_settings, func, args, kwargs)

            try:
                job.wait()
            except UnexpectedStatusException as usex:
                if usex.actual_status == "Failed":
                    try:
                        exception = serialization.deserialize_exception_from_s3(
                            sagemaker_session=job_settings.sagemaker_session,
                            s3_uri=s3_path_join(
                                job_settings.s3_root_uri, job.job_name, EXCEPTION_FOLDER
                            ),
                            hmac_key=job.hmac_key,
                        )
                    except ServiceError as serr:
                        chained_e = serr.__cause__
                        if (
                            isinstance(chained_e, ClientError)
                            and chained_e.response["Error"]["Code"]  # pylint: disable=no-member
                            == "404"
                            and chained_e.response["Error"]["Message"]  # pylint: disable=no-member
                            == "Not Found"
                        ):
                            describe_result = job.describe()
                            if (
                                "FailureReason" in describe_result
                                and describe_result["FailureReason"]
                                and "RuntimeEnvironmentError: " in describe_result["FailureReason"]
                            ):
                                failure_msg = describe_result["FailureReason"].replace(
                                    "RuntimeEnvironmentError: ", ""
                                )
                                raise RuntimeEnvironmentError(failure_msg)
                            raise RemoteFunctionError(
                                "Failed to execute remote function. "
                                + "Check corresponding job for details."
                            )
                        raise serr

                    raise exception

                raise TimeoutError(
                    "Job for remote function timed out before reaching a termination status."
                )

            if job.describe()["TrainingJobStatus"] == "Completed":
                return serialization.deserialize_obj_from_s3(
                    sagemaker_session=job_settings.sagemaker_session,
                    s3_uri=s3_path_join(job_settings.s3_root_uri, job.job_name, RESULTS_FOLDER),
                    hmac_key=job.hmac_key,
                )

            if job.describe()["TrainingJobStatus"] == "Stopped":
                raise RemoteFunctionError("Job for remote function has been aborted.")

            return None

        wrapper.job_settings = job_settings
        wrapper.wrapped_func = func
        return wrapper

    if _func is None:
        return _remote
    return _remote(_func)


class _SubmitRequest:
    """Class that holds parameters and data for creating a new job."""

    def __init__(
        self, future, job_settings: _JobSettings, func, func_args, func_kwargs, run_info=None
    ):
        self.future = future
        self.job_settings = job_settings
        self.func = func
        self.args = func_args
        self.kwargs = func_kwargs
        self.run_info = run_info


def _submit_worker(executor):
    """Background worker that submits job requests."""

    def has_work_to_do():
        return (
            len(executor._pending_request_queue) > 0
            and len(executor._running_jobs) < executor.max_parallel_jobs
        )

    try:
        while True:
            with executor._state_condition:
                executor._state_condition.wait_for(has_work_to_do)
                request = executor._pending_request_queue[0]

            if request is None:
                with executor._state_condition:
                    # remove the anchor from the pending queue
                    executor._pending_request_queue.popleft()
                return

            time.sleep(_API_CALL_LIMIT["SubmittingIntervalInSecs"])
            # submit a new job
            job = request.future._start_and_notify(
                request.job_settings, request.func, request.args, request.kwargs, request.run_info
            )

            with executor._state_condition:
                if job:
                    executor._running_jobs[job.job_name] = job
                # remove the request from the pending queue
                executor._pending_request_queue.popleft()
    except Exception:  # pylint: disable=broad-except
        logger.exception("Error occurred while submitting CreateTrainingJob requests.")


def _polling_worker(executor):
    """Background worker that polls the status of the running jobs."""
    try:
        while True:
            with executor._state_condition:
                if (
                    executor._shutdown
                    and len(executor._running_jobs) + len(executor._pending_request_queue) == 0
                ):
                    return

            time.sleep(
                max(
                    _API_CALL_LIMIT["MinBatchPollingIntervalInSecs"]
                    - len(executor._running_jobs) * _API_CALL_LIMIT["PollingIntervalInSecs"],
                    0,
                )
            )

            # check if running jobs are terminated
            for job_name in list(executor._running_jobs.keys()):
                try:
                    time.sleep(_API_CALL_LIMIT["PollingIntervalInSecs"])
                    if executor._running_jobs[job_name].describe()["TrainingJobStatus"] in [
                        "Completed",
                        "Failed",
                        "Stopped",
                    ]:
                        with executor._state_condition:
                            del executor._running_jobs[job_name]
                            executor._state_condition.notify_all()
                except Exception as e:  # pylint: disable=broad-except
                    if (
                        not isinstance(e, ClientError)
                        or e.response["Error"]["Code"]  # pylint: disable=no-member
                        != "LimitExceededException"
                    ):
                        # Couldn't check the job status, move on
                        logger.exception(
                            "Error occurred while checking the status of job %s", job_name
                        )
                        with executor._state_condition:
                            del executor._running_jobs[job_name]
                            executor._state_condition.notify_all()
    except Exception:  # pylint: disable=broad-except
        logger.exception("Error occurred while monitoring the job statuses.")


class RemoteExecutor(object):
    """Run Python functions asynchronously as SageMaker jobs"""

    def __init__(
        self,
        *,
        dependencies: str = None,
        pre_execution_commands: List[str] = None,
        pre_execution_script: str = None,
        environment_variables: Dict[str, str] = None,
        image_uri: str = None,
        include_local_workdir: bool = False,
        instance_count: int = 1,
        instance_type: str = None,
        job_conda_env: str = None,
        job_name_prefix: str = None,
        keep_alive_period_in_seconds: int = 0,
        max_parallel_jobs: int = 1,
        max_retry_attempts: int = 1,
        max_runtime_in_seconds: int = 24 * 60 * 60,
        role: str = None,
        s3_kms_key: str = None,
        s3_root_uri: str = None,
        sagemaker_session: Session = None,
        security_group_ids: List[str] = None,
        subnets: List[str] = None,
        tags: List[Tuple[str, str]] = None,
        volume_kms_key: str = None,
        volume_size: int = 30,
        encrypt_inter_container_traffic: bool = None,
        spark_config: SparkConfig = None,
        use_spot_instances=False,
        max_wait_time_in_seconds=None,
    ):
        """Constructor for RemoteExecutor

        If a parameter value is not set, the constructor first looks up the value from the
        SageMaker configuration file. If no value is specified in the configuration file or
        no configuration file is found, the constructor selects the default as specified below.
        For more information, see `Configuring and using defaults with the SageMaker Python SDK
        <https://sagemaker.readthedocs.io/en/stable/overview.html
        #configuring-and-using-defaults-with-the-sagemaker-python-sdk>`_.

        Args:
            _func (Optional): A Python function to run as a SageMaker training job.

            dependencies (str): Either the path to a dependencies file or the reserved keyword
              ``auto_capture``. Defaults to ``None``.
              If ``dependencies`` is provided, the value must be one of the following:

            * A path to a conda environment.yml file. The following conditions apply.

              * If job_conda_env is set, then the conda environment is updated by installing
                dependencies from the yaml file and the function is invoked within that
                conda environment. For this to succeed, the specified conda environment must
                already exist in the image.
              * If the environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image, then
                the conda environment is updated by installing dependencies from the yaml file and
                the function is invoked within that conda environment. For this to succeed, the
                conda environment name must already be set in ``SAGEMAKER_JOB_CONDA_ENV``, and
                ``SAGEMAKER_JOB_CONDA_ENV`` must already exist in the image.
              * If none of the previous conditions are met, a new conda environment named
                ``sagemaker-runtime-env`` is created and the function annotated with the remote
                decorator is invoked in that conda environment.

            * A path to a requirements.txt file. The following conditions apply.

              * If ``job_conda_env`` is set in the remote decorator, dependencies are installed
                within that conda environment and the function annotated with the remote decorator
                is invoked in the same conda environment. For this to succeed, the specified
                conda environment must already exist in the image.
              * If an environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image,
                dependencies are installed within that conda environment and the function annotated
                with the remote decorator is invoked in the same. For this to succeed, the
                conda environment name must already be set in ``SAGEMAKER_JOB_CONDA_ENV``, and
                ``SAGEMAKER_JOB_CONDA_ENV`` must already exist in the image.
              * If none of the above conditions are met, conda is not used. Dependencies are
                installed at the system level, without any virtual environment, and the function
                annotated with the remote decorator is invoked using the Python runtime available
                in the system path.

            * The parameter dependencies is set to ``auto_capture``. SageMaker will automatically
                generate an env_snapshot.yml corresponding to the current active conda environment’s
                snapshot. You do not need to provide a dependencies file. The following conditions
                apply:

              * You must run the remote function within an active conda environment.
              * When installing the dependencies on the training job, the same conditions as when
                dependencies is set to a path to a conda environment file apply. These conditions
                are as follows:

                  * If job_conda_env is set, then the conda environment is updated by installing
                    dependencies from the yaml file and the function is invoked within that
                    conda environment. For this to succeed, the specified conda environment must
                    already exist in the image.
                  * If the environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image,
                    then the conda environment is updated by installing dependencies from the yaml
                    file and the function is invoked within that conda environment. For this to
                    succeed, the conda environment name must already be set in
                    ``SAGEMAKER_JOB_CONDA_ENV``, and ``SAGEMAKER_JOB_CONDA_ENV`` must already exist
                    in the image.
                  * If none of the previous conditions are met, a new conda environment with name
                    ``sagemaker-runtime-env`` is created and the function annotated with the
                    remote decorator is invoked in that conda environment.

              * ``None``. SageMaker will assume that there are no dependencies to install while
                executing the remote annotated function in the training job.

            pre_execution_commands (List[str]): List of commands to be executed prior to executing
              remote function. Only one of ``pre_execution_commands`` or ``pre_execution_script``
              can be specified at the same time. Defaults to None.

            pre_execution_script (str): Path to script file to be executed prior to executing
              remote function. Only one of ``pre_execution_commands`` or ``pre_execution_script``
              can be specified at the same time. Defaults to None.

            environment_variables (Dict): The environment variables used inside the decorator
              function. Defaults to ``None``.

            image_uri (str): The universal resource identifier (URI) location of a Docker image on
              Amazon Elastic Container Registry (ECR). Defaults to the following based on where the
              SDK is running:

              * For users who specify ``spark_config`` and want to run the function in a Spark
                application, the ``image_uri`` should be ``None``. A SageMaker Spark image will
                be used for training, otherwise a ``ValueError`` is thrown.
              * For users on SageMaker Studio notebooks, the image used as the kernel image for
                the notebook is used.
              * For other users, it is resolved to base python image with the same python
                version as the environment running the local code.

              If no compatible image is found, a ValueError is thrown.

            include_local_workdir (bool): A flag to indicate that the remote function should include
              local directories. Set to ``True`` if the remote function code imports local modules
              and methods that are not available via PyPI or conda. Default value is ``False``.

            instance_count (int): The number of instances to use. Defaults to 1.
              NOTE: Remote function does not support instance_count > 1 for non Spark jobs.

            instance_type (str): The Amazon Elastic Compute Cloud (EC2) instance type to use to run
              the SageMaker job. e.g. ml.c4.xlarge. If not provided, a ValueError is thrown.

            job_conda_env (str): The name of the conda environment to activate during job's runtime.
              Defaults to ``None``.

            job_name_prefix (str): The prefix used used to create the underlying SageMaker job.

            keep_alive_period_in_seconds (int): The duration in seconds to retain and reuse
              provisioned infrastructure after the completion of a training job, also known as
              SageMaker managed warm pools. The use of warmpools reduces the latency time spent to
              provision new resources. The default value for ``keep_alive_period_in_seconds`` is 0.
              NOTE: Additional charges associated with warm pools may apply. Using this parameter
              also activates a new pesistent cache feature, which will further reduce job start
              up latency than over using SageMaker managed warm pools alone by caching the package
              source downloaded in the previous runs.

            max_parallel_jobs (int): Maximum number of jobs that run in parallel. Defaults to 1.

            max_retry_attempts (int): The max number of times the job is retried on
              ``InternalServerFailure`` Error from SageMaker service. Defaults to 1.

            max_runtime_in_seconds (int): The upper limit in seconds to be used for training. After
              this specified amount of time, SageMaker terminates the job regardless of its current
              status. Defaults to 1 day or (86400 seconds).

            role (str): The IAM role (either name or full ARN) used to run your SageMaker training
              job. Defaults to:

              * the SageMaker default IAM role if the SDK is running in SageMaker Notebooks or
                SageMaker Studio Notebooks.
              * if not above, a ValueError is be thrown.

            s3_kms_key (str): The key used to encrypt the input and output data.
              Default to ``None``.

            s3_root_uri (str): The root S3 folder to which the code archives and data are
              uploaded to. Defaults to ``s3://<sagemaker-default-bucket>``.

            sagemaker_session (sagemaker.session.Session): The underlying SageMaker session to which
              SageMaker service calls are delegated to (default: None). If not provided, one is
              created using a default configuration chain.

            security_group_ids (List[str): A list of security group IDs. Defaults to ``None`` and
              the training job is created without VPC config.

            subnets (List[str): A list of subnet IDs. Defaults to ``None`` and the job is
              created without VPC config.

            tags (List[Tuple[str, str]): A list of tags attached to the job. Defaults to ``None``
              and the training job is created without tags.

            volume_kms_key (str): An Amazon Key Management Service (KMS) key used to encrypt an
              Amazon Elastic Block Storage (EBS) volume attached to the training instance.
              Defaults to ``None``.

            volume_size (int): The size in GB of the storage volume for storing input and output
              data during training. Defaults to ``30``.

            encrypt_inter_container_traffic (bool): A flag that specifies whether traffic between
              training containers is encrypted for the training job. Defaults to ``False``.

            enable_network_isolation (bool): A flag that specifies whether container will run in
              network isolation mode. Defaults to ``False``. Network isolation mode restricts the
              container access to outside networks (such as the Internet). The container does not
              make any inbound or outbound network calls. Also known as Internet-free mode.

            spark_config (SparkConfig): Configurations to the Spark application that runs on
              Spark image. If ``spark_config`` is specified, a SageMaker Spark image uri
              will be used for training. Note that ``image_uri`` can not be specified at the
              same time otherwise a ``ValueError`` is thrown. Defaults to ``None``.

            use_spot_instances (bool): Specifies whether to use SageMaker Managed Spot instances for
              training. If enabled then the ``max_wait_time_in_seconds`` arg should also be set.
              Defaults to ``False``.

            max_wait_time_in_seconds (int): Timeout in seconds waiting for spot training job.
              After this amount of time Amazon SageMaker will stop waiting for managed spot training
              job to complete. Defaults to ``None``.
        """
        self.max_parallel_jobs = max_parallel_jobs

        if self.max_parallel_jobs <= 0:
            raise ValueError("max_parallel_jobs must be greater than 0.")

        if instance_count > 1 and not spark_config:
            raise ValueError(
                "Remote function do not support training on multi instances. "
                + "Please provide instance_count = 1"
            )

        self.job_settings = _JobSettings(
            dependencies=dependencies,
            pre_execution_commands=pre_execution_commands,
            pre_execution_script=pre_execution_script,
            environment_variables=environment_variables,
            image_uri=image_uri,
            include_local_workdir=include_local_workdir,
            instance_count=instance_count,
            instance_type=instance_type,
            job_conda_env=job_conda_env,
            job_name_prefix=job_name_prefix,
            keep_alive_period_in_seconds=keep_alive_period_in_seconds,
            max_retry_attempts=max_retry_attempts,
            max_runtime_in_seconds=max_runtime_in_seconds,
            role=role,
            s3_kms_key=s3_kms_key,
            s3_root_uri=s3_root_uri,
            sagemaker_session=sagemaker_session,
            security_group_ids=security_group_ids,
            subnets=subnets,
            tags=tags,
            volume_kms_key=volume_kms_key,
            volume_size=volume_size,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            spark_config=spark_config,
            use_spot_instances=use_spot_instances,
            max_wait_time_in_seconds=max_wait_time_in_seconds,
        )

        self._state_condition = threading.Condition()
        self._pending_request_queue = deque()
        # For thread safety, see
        # https://web.archive.org/web/20201108091210/http://effbot.org/pyfaq/what-kinds-of-global-value-mutation-are-thread-safe.htm
        self._running_jobs = dict()
        self._shutdown = False

        self._workers: ThreadPoolExecutor = None

    def submit(self, func, *args, **kwargs):
        """Execute the input function as a SageMaker job asynchronously.

        Args:
            func: Python function to run as a SageMaker job.
            *args: Positional arguments to the input function.
            **kwargs: keyword arguments to the input function
        """
        if self._shutdown:
            raise RuntimeError("Cannot schedule new remote function executions after shutdown")

        self._validate_submit_args(func, *args, **kwargs)

        with self._state_condition:
            future = Future()

            run_info = None
            if _RunContext.get_current_run() is not None:
                run = _RunContext.get_current_run()
                run_info = _RunInfo(run.experiment_name, run.run_name)

            self._pending_request_queue.append(
                _SubmitRequest(future, self.job_settings, func, args, kwargs, run_info)
            )

            if self._workers is None:
                self._workers = ThreadPoolExecutor(2)
                self._workers.submit(_submit_worker, self)
                self._workers.submit(_polling_worker, self)

            self._state_condition.notify_all()

        return future

    def map(self, func, *iterables):
        """Return an iterator that applies function to every item of iterable, yielding the results.

        If additional iterables arguments are passed, function must take that many arguments and
        is applied to the items from all iterables in parallel. With multiple iterables, the
        iterator stops when the shortest iterable is exhausted.

        Args:
            func: Python function to run as a SageMaker job.
            iterables: Arguments of the input python function.
        """

        futures = map(self.submit, itertools.repeat(func), *iterables)
        return [future.result() for future in futures]

    def shutdown(self):
        """Prevent more function executions to be submitted to this executor."""
        with self._state_condition:
            self._shutdown = True

            # give a signal to the submitting worker so that it doesn't block on empty queue forever
            self._pending_request_queue.append(None)

            self._state_condition.notify_all()

        if self._workers is not None:
            self._workers.shutdown(wait=True)

    def __enter__(self):
        """Create an executor instance and return it"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Make sure the executor instance is shutdown."""
        self.shutdown()
        return False

    @staticmethod
    def _validate_submit_args(func, *args, **kwargs):
        """Validates input args passed to submit method."""

        full_arg_spec = inspect.getfullargspec(func)

        # args related validations

        is_accepting_variable_positional_args = full_arg_spec.varargs is not None
        num_default_positional_args = len(full_arg_spec.defaults) if full_arg_spec.defaults else 0
        minimum_num_expected_positional_args = len(full_arg_spec.args) - num_default_positional_args

        if not is_accepting_variable_positional_args and len(args) > len(full_arg_spec.args):
            raise TypeError(
                f"{func.__name__}() takes {len(full_arg_spec.args)} positional "
                + f"{'arguments' if len(full_arg_spec.args) > 1 else 'argument'} but {len(args)} "
                + f"{'were' if len(args) > 1 else 'was'} given."
            )

        if len(args) < minimum_num_expected_positional_args:
            missing_positional_args = full_arg_spec.args[
                len(args) : minimum_num_expected_positional_args
            ]
            missing_args = list(filter(lambda arg: arg not in kwargs, missing_positional_args))
            if missing_args:
                missing_args_str = (
                    ", ".join(map(lambda x: f"'{x}'", missing_args[:-1]))
                    + f", and '{missing_args[-1]}'"
                    if len(missing_args) > 1
                    else f"'{missing_args[0]}'"
                )
                raise TypeError(
                    f"{func.__name__}() missing {len(missing_args)} required positional "
                    + f"{'arguments' if len(missing_args) > 1 else 'argument'}: {missing_args_str}"
                )

        # kwargs related validations

        for k in kwargs:
            if k in full_arg_spec.args and len(args) > full_arg_spec.args.index(k):
                raise TypeError(f"{func.__name__}() got multiple values for argument '{k}'")
            if k not in full_arg_spec.kwonlyargs and k not in full_arg_spec.args:
                raise TypeError(f"{func.__name__}() got an unexpected keyword argument '{k}'")

        missing_kwargs = [
            k
            for k in full_arg_spec.kwonlyargs
            if k not in full_arg_spec.kwonlydefaults and k not in kwargs
        ]
        if missing_kwargs:
            missing_kwargs_string = (
                ", ".join(map(lambda x: f"'{x}'", missing_kwargs[:-1]))
                + f", and '{missing_kwargs[-1]}'"
                if len(missing_kwargs) > 1
                else f"'{missing_kwargs[0]}'"
            )

            raise TypeError(
                f"{func.__name__}() missing {len(missing_kwargs)} required keyword-only "
                + f"{'arguments' if len(missing_kwargs) > 1 else 'argument'}: "
                + f"{missing_kwargs_string}"
            )


class Future(object):
    """Class representing a reference to a SageMaker job result.

    Reference to the SageMaker job created as a result of the remote function run. The job may
    or may not have finished running.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._state = _PENDING
        self._job = None
        self._exception = None
        self._return = None

    @staticmethod
    def from_describe_response(describe_training_job_response, sagemaker_session):
        """Construct a Future from a describe_training_job_response object."""
        future = Future()
        job_exception = None
        client_exception = None
        job_return = None
        job = _Job.from_describe_response(describe_training_job_response, sagemaker_session)
        if describe_training_job_response["TrainingJobStatus"] in ["Stopping", "Stopped"]:
            state = _CANCELLED
        elif describe_training_job_response["TrainingJobStatus"] == "Completed":
            state = _FINISHED
            try:
                job_return = serialization.deserialize_obj_from_s3(
                    sagemaker_session=sagemaker_session,
                    s3_uri=s3_path_join(job.s3_uri, RESULTS_FOLDER),
                    hmac_key=job.hmac_key,
                )
            except DeserializationError as e:
                client_exception = e
            except ServiceError as e:
                client_exception = e
        elif describe_training_job_response["TrainingJobStatus"] == "Failed":
            state = _FINISHED
            try:
                job_exception = serialization.deserialize_exception_from_s3(
                    sagemaker_session=sagemaker_session,
                    s3_uri=s3_path_join(job.s3_uri, EXCEPTION_FOLDER),
                    hmac_key=job.hmac_key,
                )
            except ServiceError as serr:
                chained_e = serr.__cause__
                if (
                    isinstance(chained_e, ClientError)
                    and chained_e.response["Error"]["Code"] == "404"  # pylint: disable=no-member
                    and chained_e.response["Error"]["Message"]  # pylint: disable=no-member
                    == "Not Found"
                ):
                    if (
                        "FailureReason" in describe_training_job_response
                        and describe_training_job_response["FailureReason"]
                        and "RuntimeEnvironmentError: "
                        in describe_training_job_response["FailureReason"]
                    ):
                        failure_msg = describe_training_job_response["FailureReason"].replace(
                            "RuntimeEnvironmentError: ", ""
                        )
                        job_exception = RuntimeEnvironmentError(failure_msg)
                    else:
                        job_exception = RemoteFunctionError(
                            "Failed to execute remote function. "
                            + "Check corresponding job for details."
                        )
                else:
                    job_exception = serr
            except DeserializationError as e:
                client_exception = e
        else:
            state = _RUNNING

        future._job = job
        future._state = state
        future._exception = job_exception or client_exception
        future._return = job_return
        return future

    def _start_and_notify(
        self, job_settings: _JobSettings, func, func_args, func_kwargs, run_info=None
    ):
        """Start and record the newly created job in the future object.

        The job is recorded if one is successfully started. Otherwise, the exception is
        recorded. The state update is broadcast to other waiting threads.
        """
        with self._condition:
            if self._state in [_PENDING]:

                try:
                    self._job = _Job.start(job_settings, func, func_args, func_kwargs, run_info)
                except (Exception,) as e:  # pylint: disable=broad-except
                    self._exception = e
                    self._state = _FINISHED
                    self._condition.notify_all()
                    return None

                self._state = _RUNNING
                self._condition.notify_all()
                return self._job
            return None

    def result(self, timeout: float = None) -> Any:
        """Returns the SageMaker job result.

        This method waits for the SageMaker job created from the remote function execution to
        complete for up to the timeout value (if specified). If timeout is ``None``,
        this method will wait until the SageMaker job completes.

        Args:
            timeout (float): Timeout in seconds to wait until the job is completed. ``None`` by
              default.

        Returns:
            The Python object returned by the remote function.
        """
        try:
            self.wait(timeout)
        except UnexpectedStatusException:
            pass

        with self._condition:
            if self._state == _PENDING:
                raise RuntimeError()

            if self._state == _RUNNING:
                if self._job.describe()["TrainingJobStatus"] == "Completed":
                    self._return = serialization.deserialize_obj_from_s3(
                        sagemaker_session=self._job.sagemaker_session,
                        s3_uri=s3_path_join(self._job.s3_uri, RESULTS_FOLDER),
                        hmac_key=self._job.hmac_key,
                    )
                    self._state = _FINISHED
                    return self._return
                if self._job.describe()["TrainingJobStatus"] == "Failed":
                    try:
                        self._exception = serialization.deserialize_exception_from_s3(
                            sagemaker_session=self._job.sagemaker_session,
                            s3_uri=s3_path_join(self._job.s3_uri, EXCEPTION_FOLDER),
                            hmac_key=self._job.hmac_key,
                        )
                    except ServiceError as serr:
                        chained_e = serr.__cause__
                        if (
                            isinstance(chained_e, ClientError)
                            and chained_e.response["Error"]["Code"]  # pylint: disable=no-member
                            == "404"
                            and chained_e.response["Error"]["Message"]  # pylint: disable=no-member
                            == "Not Found"
                        ):
                            if (
                                "FailureReason" in self._job.describe()
                                and self._job.describe()["FailureReason"]
                                and "RuntimeEnvironmentError: "
                                in self._job.describe()["FailureReason"]
                            ):
                                failure_msg = self._job.describe()["FailureReason"].replace(
                                    "RuntimeEnvironmentError: ", ""
                                )
                                self._exception = RuntimeEnvironmentError(failure_msg)
                            else:
                                self._exception = RemoteFunctionError(
                                    "Failed to execute remote function. "
                                    + "Check corresponding job for details."
                                )
                        else:
                            self._exception = serr
                    self._state = _FINISHED
                elif self._job.describe()["TrainingJobStatus"] == "Stopped":
                    self._state = _CANCELLED
                    raise RemoteFunctionError("Job for remote function has been aborted.")
                else:
                    raise TimeoutError(
                        "Job for remote function timed out before reaching a termination status."
                    )

            if self._state == _FINISHED:
                if self._exception:
                    raise self._exception
                return self._return

            return None

    def wait(
        self,
        timeout: int = None,
    ) -> None:
        """Wait for the underlying SageMaker job to complete.

        This method waits for the SageMaker job created as a result of the remote function run
        to complete for up to the timeout value (if specified). If timeout is ``None``, this method
        will block until the job is completed.

        Args:
            timeout (int): Timeout in seconds to wait until the job is completed before it is
              stopped. Defaults to ``None``.

        Returns:
            None
        """

        with self._condition:
            if self._state == _PENDING:
                self._condition.wait(timeout=timeout)

            if self._state == _RUNNING:
                self._job.wait(timeout=timeout)

    def cancel(self) -> bool:
        """Cancel the function execution.

        This method prevents the SageMaker job being created or stops the underlying SageMaker job
        early if it is already in progress.

        Returns:
            ``True`` if the underlying SageMaker job created as a result of the remote function
            run is cancelled.
        """
        with self._condition:
            if self._state == _FINISHED:
                return False
            if self._state == _CANCELLED:
                return True

            if self._job:
                self._job.stop()
            self._state = _CANCELLED
            return True

    def running(self) -> bool:
        """Check if the underlying SageMaker job is running.

        Returns:
            ``True`` if the underlying SageMaker job is still running. ``False``, otherwise.
        """
        with self._condition:
            return self._state == _RUNNING

    def cancelled(self) -> bool:
        """Check if the underlying SageMaker job was cancelled.

        Returns:
            ``True`` if the underlying SageMaker job was cancelled. ``False``, otherwise.
        """
        with self._condition:
            return self._state == _CANCELLED

    def done(self) -> bool:
        """Check if the underlying SageMaker job is finished.

        Returns:
            ``True`` if the underlying SageMaker job finished running. ``False``, otherwise.
        """
        with self._condition:
            if self._state == _RUNNING and self._job.describe()["TrainingJobStatus"] in [
                "Completed",
                "Failed",
            ]:
                self._state = _FINISHED
                return True

            if self._state == _FINISHED:
                return True

            return False


def get_future(job_name, sagemaker_session=None) -> Future:
    """Get a future object with information about a job with the given job_name.

    Args:
        job_name (str): name of the underlying SageMaker job created as a result of the remote
          function run.

        sagemaker_session (sagemaker.session.Session): A session object that manages interactions
          with Amazon SageMaker APIs and any other AWS services needed.

    Returns:
        A `sagemaker.remote_function.client.Future` instance.
    """
    if not sagemaker_session:
        sagemaker_session = Session()
    describe_training_job_response = sagemaker_session.sagemaker_client.describe_training_job(
        TrainingJobName=job_name
    )
    return Future.from_describe_response(describe_training_job_response, sagemaker_session)


def list_futures(job_name_prefix, sagemaker_session=None):
    """Generates Future objects with information about jobs with given job_name_prefix.

    Args:
        job_name_prefix (str): A prefix used to identify the SageMaker jobs associated with remote
          function run.
        sagemaker_session (sagemaker.session.Session): A session object that manages interactions
          with Amazon SageMaker APIs and any other AWS services needed.

    Yields:
        A `sagemaker.remote_function.client.Future` instance.
    """
    if not sagemaker_session:
        sagemaker_session = Session()
    job_name = name_from_base(job_name_prefix)
    # perform the following transformation because we might have trimmed the job_name_prefix while
    # creating the job.
    transformed_job_name_prefix = base_from_name(job_name)
    next_token = None
    list_training_job_kwargs = {"NameContains": transformed_job_name_prefix}
    while True:
        if next_token:
            list_training_job_kwargs["NextToken"] = next_token
        list_training_job_response = sagemaker_session.sagemaker_client.list_training_jobs(
            **list_training_job_kwargs
        )
        training_job_names = [
            job["TrainingJobName"] for job in list_training_job_response["TrainingJobSummaries"]
        ]
        for training_job_name in training_job_names:
            describe_training_job_response = (
                sagemaker_session.sagemaker_client.describe_training_job(
                    TrainingJobName=training_job_name
                )
            )
            yield Future.from_describe_response(describe_training_job_response, sagemaker_session)
        if "NextToken" in list_training_job_response:
            next_token = list_training_job_response["NextToken"]
        else:
            break
