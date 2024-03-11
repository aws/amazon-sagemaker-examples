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
"""This module stores JumpStart implementation of Estimator class."""
from __future__ import absolute_import


from typing import Dict, List, Optional, Union
from sagemaker import session
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.debugger.debugger import DebuggerHookConfig, RuleBase, TensorBoardOutputConfig
from sagemaker.debugger.profiler_config import ProfilerConfig

from sagemaker.estimator import Estimator
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.inputs import FileSystemInput, TrainingInput
from sagemaker.instance_group import InstanceGroup
from sagemaker.jumpstart.accessors import JumpStartModelsAccessor
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.exceptions import INVALID_MODEL_ID_ERROR_MSG

from sagemaker.jumpstart.factory.estimator import get_deploy_kwargs, get_fit_kwargs, get_init_kwargs
from sagemaker.jumpstart.factory.model import get_default_predictor
from sagemaker.jumpstart.utils import (
    is_valid_model_id,
    resolve_model_sagemaker_config_field,
)
from sagemaker.utils import stringify_object
from sagemaker.model_monitor.data_capture_config import DataCaptureConfig
from sagemaker.predictor import PredictorBase


from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.workflow.entities import PipelineVariable


class JumpStartEstimator(Estimator):
    """JumpStartEstimator class.

    This class sets defaults based on the model ID and version.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        region: Optional[str] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        role: Optional[str] = None,
        instance_count: Optional[Union[int, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = None,
        keep_alive_period_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        volume_size: Optional[Union[int, PipelineVariable]] = None,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        max_run: Optional[Union[int, PipelineVariable]] = None,
        input_mode: Optional[Union[str, PipelineVariable]] = None,
        output_path: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[session.Session] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        subnets: Optional[List[Union[str, PipelineVariable]]] = None,
        security_group_ids: Optional[List[Union[str, PipelineVariable]]] = None,
        model_uri: Optional[str] = None,
        model_channel_name: Optional[Union[str, PipelineVariable]] = None,
        metric_definitions: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        encrypt_inter_container_traffic: Union[bool, PipelineVariable] = None,
        use_spot_instances: Optional[Union[bool, PipelineVariable]] = None,
        max_wait: Optional[Union[int, PipelineVariable]] = None,
        checkpoint_s3_uri: Optional[Union[str, PipelineVariable]] = None,
        checkpoint_local_path: Optional[Union[str, PipelineVariable]] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
        rules: Optional[List[RuleBase]] = None,
        debugger_hook_config: Optional[Union[DebuggerHookConfig, bool]] = None,
        tensorboard_output_config: Optional[TensorBoardOutputConfig] = None,
        enable_sagemaker_metrics: Optional[Union[bool, PipelineVariable]] = None,
        profiler_config: Optional[ProfilerConfig] = None,
        disable_profiler: Optional[bool] = None,
        environment: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        max_retry_attempts: Optional[Union[int, PipelineVariable]] = None,
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        git_config: Optional[Dict[str, str]] = None,
        container_log_level: Optional[Union[int, PipelineVariable]] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[Union[str, PipelineVariable]] = None,
        dependencies: Optional[List[str]] = None,
        instance_groups: Optional[List[InstanceGroup]] = None,
        training_repository_access_mode: Optional[Union[str, PipelineVariable]] = None,
        training_repository_credentials_provider_arn: Optional[Union[str, PipelineVariable]] = None,
        container_entry_point: Optional[List[str]] = None,
        container_arguments: Optional[List[str]] = None,
        disable_output_compression: Optional[bool] = None,
    ):
        """Initializes a ``JumpStartEstimator``.

        This method sets model-specific defaults for the ``Estimator.__init__`` method.

        Only model ID is required to instantiate this class, however any field can be overriden.
        Any field set to ``None`` does not get passed to the parent class method.


        Args:
            model_id (Optional[str]): JumpStart model ID to use. See
                https://sagemaker.readthedocs.io/en/stable/doc_utils/pretrainedmodels.html
                for list of model IDs.
            model_version (Optional[str]): Version for JumpStart model to use (Default: None).
            tolerate_vulnerable_model (Optional[bool]): True if vulnerable versions of model
                specifications should be tolerated (exception not raised). If False, raises an
                exception if the script used by this version of the model has dependencies
                with known security vulnerabilities. (Default: None).
            tolerate_deprecated_model (Optional[bool]): True if deprecated models should be
                tolerated (exception not raised). False if these models should raise an exception.
                (Default: None).
            region (Optional[str]): The AWS region in which to launch the model. (Default: None).
            image_uri (Optional[Union[str, PipelineVariable]]): The container image to use for
                training. (Default: None).
            role (Optional[str]): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource. (Default: None).
            instance_count (Optional[Union[int, PipelineVariable]]): Number of Amazon EC2
                instances to usefor training. Required if instance_groups is not set.
                (Default: None).
            instance_type (Optional[Union[str, PipelineVariable]]): Type of EC2 instance to use
                for training, for example, ``'ml.c4.xlarge'``. Required if instance_groups is
                not set. (Default: None).
            keep_alive_period_in_seconds (Optional[int]): The duration of time in seconds
                to retain configured resources in a warm pool for subsequent
                training jobs. (Default: None).
            volume_size (Optional[int, PipelineVariable]): Size in GB of the storage volume to
                use for storing input and output data during training.

                Must be large enough to store training data if File mode is
                used, which is the default mode.

                When you use an ML instance with the EBS-only storage option
                such as ``ml.c5`` and ``ml.p2``,
                you must define the size of the EBS
                volume through the ``volume_size`` parameter in the estimator class.

                .. note::

                    When you use an ML instance with `NVMe SSD volumes
                    <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ssd-instance-store.html#nvme-ssd-volumes>`_
                    such as ``ml.p4d``, ``ml.g4dn``, and ``ml.g5``,
                    do not include this parameter in the estimator configuration.
                    If you use one of those ML instance types,
                    SageMaker doesn't provision Amazon EBS General Purpose SSD
                    (gp2) storage nor take this parameter to adjust the NVMe instance storage.
                    Available storage is fixed to the NVMe instance storage
                    capacity. SageMaker configures storage paths for training
                    datasets, checkpoints, model artifacts, and outputs to use the
                    entire capacity of the instance storage.

                    Note that if you include this parameter and specify a number that
                    exceeds the size of the NVMe volume attached to the instance type,
                    SageMaker returns an ``Invalid VolumeSizeInGB`` error.

                To look up instance types and their instance storage types
                and volumes, see `Amazon EC2 Instance Types
                <http://aws.amazon.com/ec2/instance-types/>`_.

                To find the default local paths defined by the SageMaker
                training platform, see `Amazon SageMaker Training Storage
                Folders for Training Datasets, Checkpoints, Model Artifacts,
                and Outputs
                <https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html>`_.
                (Default: None).
            volume_kms_key (Optional[Union[str, PipelineVariable]]): KMS key ID for encrypting EBS
                volume attached to the training instance. (Default: None).
            max_run (Optional[Union[int, PipelineVariable]]): Timeout in seconds for training.
                After this amount of time Amazon SageMaker terminates
                the job regardless of its current status. (Default: None).
            input_mode (Optional[Union[str, PipelineVariable]]): The input mode that the
                algorithm supports. Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the
                  S3 location to a local directory.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the
                  container via a Unix-named pipe.

                This argument can be overriden on a per-channel basis using
                ``sagemaker.inputs.TrainingInput.input_mode``. (Default: None).
            output_path (Optional[Union[str, PipelineVariable]]): S3 location for saving
                the training result (model artifacts and output files). If not specified,
                results are stored to a default bucket. If the bucket with the specific name
                does not exist, the estimator creates the bucket during the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method execution.
                (Default: None).
            output_kms_key (Optional[Union[str, PipelineVariable]]): KMS key ID for encrypting the
                training output. (Default: None).
            base_job_name (Optional[str]): Prefix for training job name when the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method launches.
                If not specified, the estimator generates a default job name,
                based on the training image name and current timestamp. (Default: None).
            sagemaker_session (Optional[sagemaker.session.Session]): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain. (Default: None).
            hyperparameters (Optional[Union[dict[str, str],dict[str, PipelineVariable]]]):
                Dictionary containing the hyperparameters to initialize this estimator with.

                .. caution::
                    You must not include any security-sensitive information, such as
                    account access IDs, secrets, and tokens, in the dictionary for configuring
                    hyperparameters. SageMaker rejects the training job request and returns an
                    validation error for detected credentials, if such user input is found.

                (Default: None).
            tags (Optional[Union[list[dict[str, str], list[dict[str, PipelineVariable]]]]):
                List of tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
                (Default: None).
            subnets (Optional[Union[list[str], list[PipelineVariable]]]): List of subnet ids.
                If not specified training job will be created without VPC config. (Default: None).
            security_group_ids (Optional[Union[list[str], list[PipelineVariable]]]): List of
                security group ids. If not specified training job will be created without
                VPC config. (Default: None).
            model_uri (Optional[str]): URI where a pre-trained model is stored, either
                locally or in S3 (Default: None). If specified, the estimator
                will create a channel pointing to the model so the training job
                can download it. This model can be a 'model.tar.gz' from a
                previous training job, or other artifacts coming from a
                different source.

                In local mode, this should point to the path in which the model
                is located and not the file itself, as local Docker containers
                will try to mount the URI as a volume.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization

                (Default: None).
            model_channel_name (Optional[Union[str, PipelineVariable]]): Name of the channel where
                'model_uri' will be downloaded. (Default: None).
            metric_definitions (Optional[Union[list[dict[str, str], list[dict[str,
                PipelineVariable]]]]): A list of dictionaries that defines the metric(s)
                used to evaluate the training jobs. Each dictionary contains two keys: 'Name'
                for the name of the metric, and 'Regex' for the regular expression used to extract
                the metric from the logs. This should be defined only for jobs that
                don't use an Amazon algorithm. (Default: None).
            encrypt_inter_container_traffic (Optional[Union[bool, PipelineVariable]]]): Specifies
                whether traffic between training containers is encrypted for the training job
                (Default: None).
            use_spot_instances (Optional[Union[bool, PipelineVariable]]): Specifies whether to
                use SageMaker Managed Spot instances for training. If enabled then the
                ``max_wait`` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (Default: None).
            max_wait (Optional[Union[int, PipelineVariable]]): Timeout in seconds waiting
                for spot training job. After this amount of time Amazon
                SageMaker will stop waiting for managed spot training job to
                complete. (Default: None).
            checkpoint_s3_uri (Optional[Union[str, PipelineVariable]]): The S3 URI in which
                to persist checkpoints that the algorithm persists (if any) during training.
                (Default: None).
            checkpoint_local_path (Optional[Union[str, PipelineVariable]]): The local path
                that the algorithm writes its checkpoints to. SageMaker will persist all
                files under this path to `checkpoint_s3_uri` continually during
                training. On job startup the reverse happens - data from the
                s3 location is downloaded to this path before the algorithm is
                started. If the path is unset then SageMaker assumes the
                checkpoints will be provided under `/opt/ml/checkpoints/`.
                (Default: None).
            enable_network_isolation (Optional[Union[bool, PipelineVariable]]): Specifies
                whether container will run in network isolation mode. Network isolation mode
                restricts the container access to outside networks (such as the Internet).
                The container does not make any inbound or outbound network calls.
                Also known as Internet-free mode. (Default: None).
            rules (Optional[list[:class:`~sagemaker.debugger.RuleBase`]]): A list of
                :class:`~sagemaker.debugger.RuleBase` objects used to define
                SageMaker Debugger rules for real-time analysis
                (Default: None). For more information,
                see `Continuous analyses through rules
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html
                #continuous-analyses-through-rules)>`_.
                (Default: None).
            debugger_hook_config (Optional[Union[DebuggerHookConfig, bool]]):
                Configuration for how debugging information is emitted with
                SageMaker Debugger. If not specified, a default one is created using
                the estimator's ``output_path``, unless the region does not
                support SageMaker Debugger. To disable SageMaker Debugger,
                set this parameter to ``False``. For more information, see
                `Capture real-time debugging data during model training in Amazon SageMaker
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#
                capture-real-time-debugging-data-during-model-training-in-amazon-sagemaker>`_.
                (Default: None).
            tensorboard_output_config (:class:`~sagemaker.debugger.TensorBoardOutputConfig`):
                Configuration for customizing debugging visualization using TensorBoard.
                For more information, see `Capture real time tensorboard data
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#
                capture-real-time-tensorboard-data-from-the-debugging-hook>`_.
                (Default: None).
            enable_sagemaker_metrics (Optional[Union[bool, PipelineVariable]]): enable
                SageMaker Metrics Time Series. For more information, see `AlgorithmSpecification
                API <https://docs.aws.amazon.com/sagemaker/latest/dg/
                API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-
                EnableSageMakerMetricsTimeSeries>`_.
                (Default: None).
            profiler_config (Optional[:class:`~sagemaker.debugger.ProfilerConfig`]):
                Configuration for how SageMaker Debugger collects
                monitoring and profiling information from your training job.
                If not specified, Debugger will be configured with
                a default configuration and will save system and framework metrics
                the estimator's default ``output_path`` in Amazon S3.
                Use :class:`~sagemaker.debugger.ProfilerConfig` to configure this parameter.
                To disable SageMaker Debugger monitoring and profiling, set the
                ``disable_profiler`` parameter to ``True``. (Default: None).
            disable_profiler (Optional[bool]): Specifies whether Debugger monitoring and profiling
                will be disabled. (Default: None).
            environment (Optional[Union[dict[str, str], dict[str, PipelineVariable]]]):
                Environment variables to be set for use during training job. (Default: None).
            max_retry_attempts (Optional[Union[int, PipelineVariable]]): The number of times
                to move a job to the STARTING status. You can specify between 1 and 30 attempts.
                If the value of attempts is greater than zero,
                the job is retried on InternalServerFailure
                the same number of attempts as the value.
                You can cap the total duration for your job by setting ``max_wait`` and ``max_run``.
                (Default: None).
            source_dir (Optional[Union[str, PipelineVariable]]): The absolute, relative, or
                S3 URI Path to a directory with any other training source code dependencies
                aside from the entry point file. If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory is preserved
                when training on Amazon SageMaker. If 'git_config' is provided,
                'source_dir' should be a relative location to a directory in the Git
                repo.
                (Default: None).

                .. admonition:: Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- train.py
                    >>>         |----- test.py

                    if you need 'train.py'
                    as the entry point and 'test.py' as the training source code, you can assign
                    entry_point='train.py', source_dir='src'.
            git_config (Optional[dict[str, str]]): Git configurations used for cloning
                files, including ``repo``, ``branch``, ``commit``,
                ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                ``repo`` field is required. All other fields are optional.
                ``repo`` specifies the Git repository where your training script
                is stored. If you don't provide ``branch``, the default value
                'master' is used. If you don't provide ``commit``, the latest
                commit in the specified branch is used.

                ``2FA_enabled``, ``username``, ``password`` and ``token`` are
                used for authentication. For GitHub (or other Git) accounts, set
                ``2FA_enabled`` to 'True' if two-factor authentication is
                enabled for the account, otherwise set it to 'False'. If you do
                not provide a value for ``2FA_enabled``, a default value of
                'False' is used. CodeCommit does not support two-factor
                authentication, so do not provide "2FA_enabled" with CodeCommit
                repositories.

                For GitHub and other Git repos, when SSH URLs are provided, it
                doesn't matter whether 2FA is enabled or disabled. You should
                either have no passphrase for the SSH key pairs or have the
                ssh-agent configured so that you will not be prompted for the SSH
                passphrase when you run the 'git clone' command with SSH URLs. When
                HTTPS URLs are provided, if 2FA is disabled, then either ``token``
                or ``username`` and ``password`` are be used for authentication if provided.
                ``Token`` is prioritized. If 2FA is enabled, only ``token`` is used
                for authentication if provided. If required authentication info
                is not provided, the SageMaker Python SDK attempts to use local credentials
                to authenticate. If that fails, an error message is thrown.

                For CodeCommit repos, 2FA is not supported, so ``2FA_enabled``
                should not be provided. There is no token in CodeCommit, so
                ``token`` should also not be provided. When ``repo`` is an SSH URL,
                the requirements are the same as GitHub  repos. When ``repo``
                is an HTTPS URL, ``username`` and ``password`` are used for
                authentication if they are provided. If they are not provided,
                the SageMaker Python SDK attempts to use either the CodeCommit
                credential helper or local credential storage for authentication.
                (Default: None).

                .. admonition:: Example
                    The following config:

                    >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                    >>>               'branch': 'test-branch-git-config',
                    >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

                    results in cloning the repo specified in 'repo', then
                    checking out the 'master' branch, and checking out the specified
                    commit.
            container_log_level (Optional[Union[int, PipelineVariable]]): The log level to use
                within the container. Valid values are defined in the Python logging module.
                (Default: None).
            code_location (Optional[str]): The S3 prefix URI where custom code is
                uploaded (Default: None). You must not include a trailing slash because
                a string prepended with a "/" is appended to ``code_location``. The code
                file uploaded to S3 is 'code_location/job-name/source/sourcedir.tar.gz'.
                If not specified, the default ``code location`` is 's3://output_bucket/job-name/'.
                (Default: None).
            entry_point (Optional[Union[str, PipelineVariable]]): The absolute or relative path
                to the local Python source file that should be executed as the entry point to
                training. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo.
                (Default: None).

                .. admonition:: Example
                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- train.py
                    >>>         |----- test.py

                    You can assign entry_point='src/train.py'.
            dependencies (Optional[list[str]]): A list of absolute or relative paths to directories
                with any additional libraries that should be exported
                to the container. The library folders are
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo. This is not supported with "local code"
                in Local Mode. (Default: None).

                .. admonition:: Example
                    The following Estimator call:

                    >>> Estimator(entry_point='train.py',
                    ...           dependencies=['my/libs/common', 'virtual-env'])

                    results in the following structure inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ train.py
                    >>>     |------ common
                    >>>     |------ virtual-env
            instance_groups (Optional[list[:class:`sagemaker.instance_group.InstanceGroup`]]):
                A list of ``InstanceGroup`` objects for launching a training job with a
                heterogeneous cluster. For example:

                .. code:: python

                    instance_groups=[
                        sagemaker.InstanceGroup(
                            'instance_group_name_1', 'ml.p3dn.24xlarge', 64),
                        sagemaker.InstanceGroup(
                            'instance_group_name_2', 'ml.c5n.18xlarge', 64)]

                For instructions on how to use ``InstanceGroup`` objects
                to configure a heterogeneous cluster
                through the SageMaker generic and framework estimator classes, see
                `Train Using a Heterogeneous Cluster
                <https://docs.aws.amazon.com/sagemaker/latest/dg/train-heterogeneous-cluster.html>`_
                in the *Amazon SageMaker developer guide*. (Default: None).
            training_repository_access_mode (Optional[str]): Specifies how SageMaker accesses the
                Docker image that contains the training algorithm (Default: None).
                Set this to one of the following values:
                * 'Platform' - The training image is hosted in Amazon ECR.
                * 'Vpc' - The training image is hosted in a private Docker registry in your VPC.
                When it's default to None, its behavior will be same as 'Platform' - image is hosted
                in ECR. (Default: None).
            training_repository_credentials_provider_arn (Optional[str]): The Amazon Resource Name
                (ARN) of an AWS Lambda function that provides credentials to authenticate to the
                private Docker registry where your training image is hosted (Default: None).
                When it's set to None, SageMaker will not do authentication before pulling the image
                in the private Docker registry. (Default: None).
            container_entry_point (Optional[List[str]]): The entrypoint script for a Docker
                container used to run a training job. This script takes precedence over
                the default train processing instructions.
            container_arguments (Optional[List[str]]): The arguments for a container used to run
                a training job.
            disable_output_compression (Optional[bool]): When set to true, Model is uploaded
                to Amazon S3 without compression after training finishes.

        Raises:
            ValueError: If the model ID is not recognized by JumpStart.
        """

        def _is_valid_model_id_hook():
            return is_valid_model_id(
                model_id=model_id,
                model_version=model_version,
                region=region,
                script=JumpStartScriptScope.TRAINING,
                sagemaker_session=sagemaker_session,
            )

        if not _is_valid_model_id_hook():
            JumpStartModelsAccessor.reset_cache()
            if not _is_valid_model_id_hook():
                raise ValueError(INVALID_MODEL_ID_ERROR_MSG.format(model_id=model_id))

        estimator_init_kwargs = get_init_kwargs(
            model_id=model_id,
            model_version=model_version,
            tolerate_vulnerable_model=tolerate_vulnerable_model,
            tolerate_deprecated_model=tolerate_deprecated_model,
            role=role,
            region=region,
            instance_count=instance_count,
            instance_type=instance_type,
            keep_alive_period_in_seconds=keep_alive_period_in_seconds,
            volume_size=volume_size,
            volume_kms_key=volume_kms_key,
            max_run=max_run,
            input_mode=input_mode,
            output_path=output_path,
            output_kms_key=output_kms_key,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            tags=tags,
            subnets=subnets,
            security_group_ids=security_group_ids,
            model_uri=model_uri,
            model_channel_name=model_channel_name,
            metric_definitions=metric_definitions,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
            rules=rules,
            debugger_hook_config=debugger_hook_config,
            tensorboard_output_config=tensorboard_output_config,
            enable_sagemaker_metrics=enable_sagemaker_metrics,
            enable_network_isolation=enable_network_isolation,
            profiler_config=profiler_config,
            disable_profiler=disable_profiler,
            environment=environment,
            max_retry_attempts=max_retry_attempts,
            source_dir=source_dir,
            git_config=git_config,
            hyperparameters=hyperparameters,
            container_log_level=container_log_level,
            code_location=code_location,
            entry_point=entry_point,
            dependencies=dependencies,
            instance_groups=instance_groups,
            training_repository_access_mode=training_repository_access_mode,
            training_repository_credentials_provider_arn=(
                training_repository_credentials_provider_arn
            ),
            image_uri=image_uri,
            container_entry_point=container_entry_point,
            container_arguments=container_arguments,
            disable_output_compression=disable_output_compression,
        )

        self.model_id = estimator_init_kwargs.model_id
        self.model_version = estimator_init_kwargs.model_version
        self.instance_type = estimator_init_kwargs.instance_type
        self.tolerate_deprecated_model = estimator_init_kwargs.tolerate_deprecated_model
        self.tolerate_vulnerable_model = estimator_init_kwargs.tolerate_vulnerable_model
        self.instance_count = estimator_init_kwargs.instance_count
        self.region = estimator_init_kwargs.region
        self.orig_predictor_cls = None
        self.role = estimator_init_kwargs.role
        self.sagemaker_session = estimator_init_kwargs.sagemaker_session
        self._enable_network_isolation = estimator_init_kwargs.enable_network_isolation

        super(JumpStartEstimator, self).__init__(**estimator_init_kwargs.to_kwargs_dict())

    def fit(
        self,
        inputs: Optional[Union[str, Dict, TrainingInput, FileSystemInput]] = None,
        wait: Optional[bool] = True,
        logs: Optional[str] = None,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
    ) -> None:
        """Start training job by calling base ``Estimator`` class ``fit`` method.

        Any field set to ``None`` does not get passed to the parent class method.

        Args:
            inputs (Optional[Union[str, dict, sagemaker.inputs.TrainingInput, sagemaker.inputs.FileSystemInput]]):
                Information about the training data. This can be one of four types:

                * (str) the S3 location where training data is saved, or a file:// path in
                    local mode.
                * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput] or
                    dict[str, sagemaker.inputs.FileSystemInput]) If using multiple channels for
                    training data, you can specify a dict mapping channel names to strings or
                    :func:`~sagemaker.inputs.TrainingInput` objects or
                    :func:`~sagemaker.inputs.FileSystemInput` objects.
                * (sagemaker.inputs.TrainingInput) - channel configuration for S3 data sources
                    that can provide additional information as well as the path to the training
                    dataset.
                    See :func:`sagemaker.inputs.TrainingInput` for full details.
                * (sagemaker.inputs.FileSystemInput) - channel configuration for
                    a file system data source that can provide additional information as well as
                    the path to the training dataset.


                (Default: None).
            wait (Optional[bool]): Whether the call should wait until the job completes.
                (Default: True).
            logs (Optional[List[str]]): A list of strings specifying which logs to print. Acceptable
                strings are "All", "None", "Training", or "Rules". To maintain backwards
                compatibility, boolean values are also accepted and converted to strings.
                Only meaningful when wait is True. (Default: None).
            job_name (Optional[str]): Training job name. If not specified, the estimator generates
                a default job name based on the training image name and current timestamp.
                (Default: None).
            experiment_config (Optional[dict[str, str]]): Experiment management configuration.
                Optionally, the dict can contain four keys:
                'ExperimentName', 'TrialName', 'TrialComponentDisplayName' and 'RunName'..
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
                * Both `ExperimentName` and `TrialName` will be ignored if the Estimator instance
                is built with :class:`~sagemaker.workflow.pipeline_context.PipelineSession`.
                However, the value of `TrialComponentDisplayName` is honored for display in Studio.
                (Default: None).
        """

        estimator_fit_kwargs = get_fit_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            inputs=inputs,
            wait=wait,
            logs=logs,
            job_name=job_name,
            experiment_config=experiment_config,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
            sagemaker_session=self.sagemaker_session,
        )

        return super(JumpStartEstimator, self).fit(**estimator_fit_kwargs.to_kwargs_dict())

    @classmethod
    def attach(
        cls,
        training_job_name: str,
        model_id: str,
        model_version: str = "*",
        sagemaker_session: session.Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
        model_channel_name: str = "model",
    ) -> "JumpStartEstimator":
        """Attach to an existing training job.

        Create a JumpStartEstimator bound to an existing training job.
        After attaching, if the training job has a Complete status,
        it can be ``deploy()`` ed to create a SageMaker Endpoint and return
        a ``Predictor``.

        If the training job is in progress, attach will block until the training job
        completes, but logs of the training job will not display. To see the logs
        content, please call ``logs()``

        Examples:
            >>> my_estimator.fit(wait=False)
            >>> training_job_name = my_estimator.latest_training_job.name
            Later on:
            >>> attached_estimator = JumpStartEstimator.attach(training_job_name, model_id)
            >>> attached_estimator.logs()
            >>> attached_estimator.deploy()

        Args:
            training_job_name (str): The name of the training job to attach to.
            model_id (str): The name of the JumpStart model id associated with the
                training job.
            model_version (str): Optional. The version of the JumpStart model id
                associated with the training job. (Default: "*").
            sagemaker_session (sagemaker.session.Session): Optional. Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
                (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
            model_channel_name (str): Optional. Name of the channel where pre-trained
                model data will be downloaded (default: 'model'). If no channel
                with the same name exists in the training job, this option will
                be ignored.

        Returns:
            Instance of the calling ``JumpStartEstimator`` Class with the attached
            training job.
        """

        return cls._attach(
            training_job_name=training_job_name,
            sagemaker_session=sagemaker_session,
            model_channel_name=model_channel_name,
            additional_kwargs={"model_id": model_id, "model_version": model_version},
        )

    def deploy(
        self,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        serializer: Optional[BaseSerializer] = None,
        deserializer: Optional[BaseDeserializer] = None,
        accelerator_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        tags: List[Dict[str, str]] = None,
        kms_key: Optional[str] = None,
        wait: Optional[bool] = True,
        data_capture_config: Optional[DataCaptureConfig] = None,
        async_inference_config: Optional[AsyncInferenceConfig] = None,
        serverless_inference_config: Optional[ServerlessInferenceConfig] = None,
        volume_size: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        inference_recommendation_id: Optional[str] = None,
        explainer_config: Optional[ExplainerConfig] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        model_name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None,
        sagemaker_session: Optional[session.Session] = None,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Optional[Union[int, PipelineVariable]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        use_compiled_model: bool = False,
    ) -> PredictorBase:
        """Creates endpoint from training job.

        Calls base ``Estimator`` class ``deploy`` method.

        Any field set to ``None`` does not get passed to the parent class method.

        Args:
            initial_instance_count (Optional[int]): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. If not using
                serverless inference or the model has not called ``right_size()``,
                then it need to be a number larger or equals
                to 1. (Default: None)
            instance_type (Optional[str]): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference or the model has not called ``right_size()``,
                then it is required to deploy a model.
                (Default: None)
            serializer (Optional[:class:`~sagemaker.serializers.BaseSerializer`]): A
                serializer object, used to encode data for an inference endpoint
                (Default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``. (Default: None).
            deserializer (Optional[:class:`~sagemaker.deserializers.BaseDeserializer`]): A
                deserializer object, used to decode data from an inference
                endpoint (Default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``. (Default: None).
            accelerator_type (Optional[str]): Type of Elastic Inference accelerator to
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
                (Default: None).
            endpoint_name (Optional[str]): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
                (Default: None).
            tags (Optional[List[dict[str, str]]]): The list of tags to attach to this
                specific endpoint. (Default: None).
            kms_key (Optional[str]): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint. (Default: None).
            wait (Optional[bool]): Whether the call should wait until the deployment of
                this model completes. (Default: True).
            data_capture_config (Optional[sagemaker.model_monitor.DataCaptureConfig]): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. (Default: None).
            async_inference_config (Optional[sagemaker.model_monitor.AsyncInferenceConfig]):
                Specifies configuration related to async endpoint. Use this configuration when
                trying to create async endpoint and make async inference. If empty config object
                passed through, will use default config to deploy async endpoint. Deploy a
                real-time endpoint if it's None. (Default: None)
            serverless_inference_config (Optional[sagemaker.serverless.ServerlessInferenceConfig]):
                Specifies configuration related to serverless endpoint. Use this configuration
                when trying to create serverless endpoint and make serverless inference. If
                empty object passed through, will use pre-defined values in
                ``ServerlessInferenceConfig`` class to deploy serverless endpoint. Deploy an
                instance based endpoint if it's None. (Default: None)
            volume_size (Optional[int]): The size, in GB, of the ML storage volume attached to
                individual inference instance associated with the production variant.
                Currenly only Amazon EBS gp2 storage volumes are supported. (Default: None).
            model_data_download_timeout (Optional[int]): The timeout value, in seconds, to download
                and extract model data from Amazon S3 to the individual inference instance
                associated with this production variant. (Default: None).
            container_startup_health_check_timeout (Optional[int]): The timeout value, in seconds,
                for your inference container to pass health check by SageMaker Hosting. For more
                information about health check see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-algo-ping-requests
                (Default: None).
            inference_recommendation_id (Optional[str]): The recommendation id which specifies the
                recommendation you picked from inference recommendation job results and
                would like to deploy the model and endpoint with recommended parameters.
                (Default: None).
            explainer_config (Optional[sagemaker.explainer.ExplainerConfig]): Specifies online
                explainability configuration for use with Amazon SageMaker Clarify. (Default: None).
            image_uri (Optional[Union[str, PipelineVariable]]): A Docker image URI. (Default: None).
            role (Optional[str]): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role if it needs to access some AWS resources.
                It can be null if this is being used to create a Model to pass
                to a ``PipelineModel`` which has its own Role field. (Default:
                None).
            predictor_cls (Optional[callable[string, sagemaker.session.Session]]): A
                function to call to create a predictor (Default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name. (Default: None).
            env (Optional[dict[str, str] or dict[str, PipelineVariable]]): Environment variables
                to run with ``image_uri`` when hosted in SageMaker. (Default: None).
            model_name (Optional[str]): The model name. If None, a default model name will be
                selected on each ``deploy``. (Default: None).
            vpc_config (Optional[Union[dict[str, list[str]],dict[str, list[PipelineVariable]]]]):
                The VpcConfig set on the model (Default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids. (Default: None).
            sagemaker_session (Optional[sagemaker.session.Session]): A SageMaker Session
                object, used for SageMaker interactions (Default: None). If not
                specified, one is created using the default AWS configuration
                chain. (Default: None).
            enable_network_isolation (Optional[Union[bool, PipelineVariable]]): If True,
                enables network isolation in the endpoint, isolating the model
                container. No inbound or outbound network calls can be made to
                or from the model container. (Default: None).
            model_kms_key (Optional[str]): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked. (Default: None).
            image_config (Optional[Union[dict[str, str], dict[str, PipelineVariable]]]): Specifies
                whether the image of model container is pulled from ECR, or private
                registry in your VPC. By default it is set to pull model container
                image from ECR. (Default: None).
            source_dir (Optional[str]): The absolute, relative, or S3 URI Path to a directory
                with any other training source code dependencies aside from the entry
                point file (Default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory is preserved
                when training on Amazon SageMaker. If 'git_config' is provided,
                'source_dir' should be a relative location to a directory in the Git repo.
                If the directory points to S3, no code is uploaded and the S3 location
                is used instead. (Default: None).

                .. admonition:: Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='inference.py', source_dir='src'.
            code_location (Optional[str]): Name of the S3 bucket where custom code is
                uploaded (Default: None). If not specified, the default bucket
                created by ``sagemaker.session.Session`` is used. (Default: None).
            entry_point (Optional[str]): The absolute or relative path to the local Python
                source file that should be executed as the entry point to
                model hosting. (Default: None). If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo.

                Example:
                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='src/inference.py'.

                (Default: None).
            container_log_level (Optional[Union[int, PipelineVariable]]): Log level to use within
                the container. Valid values are defined in the Python logging module.
                (Default: None).
            dependencies (Optional[list[str]]): A list of absolute or relative paths to directories
                with any additional libraries that should be exported
                to the container (default: []). The library folders are
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo. If the ```source_dir``` points
                to S3, code will be uploaded and the S3 location will be used
                instead.

                .. admonition:: Example

                    The following call

                    >>> Model(entry_point='inference.py',
                    ...       dependencies=['my/libs/common', 'virtual-env'])

                    results in the following structure inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ inference.py
                    >>>     |------ common
                    >>>     |------ virtual-env

                This is not supported with "local code" in Local Mode.
                (Default: None).
            git_config (Optional[dict[str, str]]): Git configurations used for cloning
                files, including ``repo``, ``branch``, ``commit``,
                ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                ``repo`` field is required. All other fields are optional.
                ``repo`` specifies the Git repository where your training script
                is stored. If you don't provide ``branch``, the default value
                'master' is used. If you don't provide ``commit``, the latest
                commit in the specified branch is used.

                .. admonition:: Example

                    The following config:

                    >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                    >>>               'branch': 'test-branch-git-config',
                    >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

                    results in cloning the repo specified in 'repo', then
                    checking out the 'master' branch, and checking out the specified
                    commit.

                ``2FA_enabled``, ``username``, ``password`` and ``token`` are
                used for authentication. For GitHub (or other Git) accounts, set
                ``2FA_enabled`` to 'True' if two-factor authentication is
                enabled for the account, otherwise set it to 'False'. If you do
                not provide a value for ``2FA_enabled``, a default value of
                'False' is used. CodeCommit does not support two-factor
                authentication, so do not provide "2FA_enabled" with CodeCommit
                repositories.

                For GitHub and other Git repos, when SSH URLs are provided, it
                doesn't matter whether 2FA is enabled or disabled. You should
                either have no passphrase for the SSH key pairs or have the
                ssh-agent configured so that you will not be prompted for the SSH
                passphrase when you run the 'git clone' command with SSH URLs. When
                HTTPS URLs are provided, if 2FA is disabled, then either ``token``
                or ``username`` and ``password`` are be used for authentication if provided.
                ``Token`` is prioritized. If 2FA is enabled, only ``token`` is used
                for authentication if provided. If required authentication info
                is not provided, the SageMaker Python SDK attempts to use local credentials
                to authenticate. If that fails, an error message is thrown.

                For CodeCommit repos, 2FA is not supported, so ``2FA_enabled``
                should not be provided. There is no token in CodeCommit, so
                ``token`` should also not be provided. When ``repo`` is an SSH URL,
                the requirements are the same as GitHub  repos. When ``repo``
                is an HTTPS URL, ``username`` and ``password`` are used for
                authentication if they are provided. If they are not provided,
                the SageMaker Python SDK attempts to use either the CodeCommit
                credential helper or local credential storage for authentication.
                (Default: None).
            use_compiled_model (bool): Flag to select whether to use compiled
                (optimized) model. (Default: False).
        """

        self.orig_predictor_cls = predictor_cls

        sagemaker_session = sagemaker_session or self.sagemaker_session
        role = resolve_model_sagemaker_config_field(
            field_name="role",
            field_val=role,
            sagemaker_session=sagemaker_session,
            default_value=self.role,
        )

        estimator_deploy_kwargs = get_deploy_kwargs(
            model_id=self.model_id,
            model_version=self.model_version,
            region=self.region,
            tolerate_vulnerable_model=self.tolerate_vulnerable_model,
            tolerate_deprecated_model=self.tolerate_deprecated_model,
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            serializer=serializer,
            deserializer=deserializer,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
            tags=tags,
            kms_key=kms_key,
            wait=wait,
            data_capture_config=data_capture_config,
            async_inference_config=async_inference_config,
            serverless_inference_config=serverless_inference_config,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
            inference_recommendation_id=inference_recommendation_id,
            explainer_config=explainer_config,
            image_uri=image_uri,
            role=role,
            predictor_cls=predictor_cls,
            env=env,
            model_name=model_name,
            vpc_config=vpc_config,
            sagemaker_session=sagemaker_session,
            enable_network_isolation=enable_network_isolation,
            model_kms_key=model_kms_key,
            image_config=image_config,
            source_dir=source_dir,
            code_location=code_location,
            entry_point=entry_point,
            container_log_level=container_log_level,
            dependencies=dependencies,
            git_config=git_config,
            use_compiled_model=use_compiled_model,
        )

        predictor = super(JumpStartEstimator, self).deploy(
            **estimator_deploy_kwargs.to_kwargs_dict()
        )

        # If no predictor class was passed, add defaults to predictor
        if self.orig_predictor_cls is None and async_inference_config is None:
            return get_default_predictor(
                predictor=predictor,
                model_id=self.model_id,
                model_version=self.model_version,
                region=self.region,
                tolerate_deprecated_model=self.tolerate_deprecated_model,
                tolerate_vulnerable_model=self.tolerate_vulnerable_model,
                sagemaker_session=self.sagemaker_session,
            )

        # If a predictor class was passed, do not mutate predictor
        return predictor

    def __str__(self) -> str:
        """Overriding str(*) method to make more human-readable."""
        return stringify_object(self)
