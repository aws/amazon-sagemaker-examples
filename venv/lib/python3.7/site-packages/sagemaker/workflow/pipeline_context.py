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
"""The pipeline context for workflow"""
from __future__ import absolute_import

import warnings
import inspect
from functools import wraps
from typing import Dict, Optional, Callable

from sagemaker.session import Session, SessionSettings
from sagemaker.local import LocalSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig


class _StepArguments:
    """Step arguments entity for `Step`"""

    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, caller_name: str = None, func: Callable = None, *func_args, **func_kwargs):
        """Create a `_StepArguments`

        Args:
            caller_name (str): The name of the caller function which is intercepted by the
                PipelineSession to get the step arguments.
            func (Callable): The job class function that generates the step arguments used
                when creating the job ( fit() for a training job )
            *func_args: The args for func
            **func_kwargs: The kwargs for func
        """
        self.caller_name = caller_name
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs


class _JobStepArguments(_StepArguments):
    """Step arguments entity for job step types

    Job step types include: TrainingStep, ProcessingStep, TuningStep, TransformStep
    """

    def __init__(self, caller_name: str, args: dict):
        """Create a `_JobStepArguments`

        Args:
            caller_name (str): The name of the caller function which is intercepted by the
                PipelineSession to get the step arguments.
            args (dict): The arguments to be used for composing the SageMaker API request.
        """
        super(_JobStepArguments, self).__init__(caller_name)
        self.args = args


class _ModelStepArguments(_StepArguments):
    """Step arguments entity for `ModelStep`"""

    def __init__(self, model):
        """Create a `_ModelStepArguments`

        Args:
            model (Model or PipelineModel): A `sagemaker.model.Model`
                or `sagemaker.pipeline.PipelineModel` instance
        """
        super(_ModelStepArguments, self).__init__()
        self.model = model
        self.create_model_package_request = None
        self.create_model_request = None
        self.need_runtime_repack = set()
        self.runtime_repack_output_prefix = None


class _PipelineConfig:
    """Config object that associates a step with its containing pipeline

    Args:
        pipeline_name (str): pipeline name
        step_name (str): step name
        code_hash (str): a hash of the code artifact for the particular step
        config_hash (str): a hash of the config artifact for the particular step (Processing)
        pipeline_definition_config (PipelineDefinitionConfig): a configuration used to toggle
            feature flags persistent in a pipeline definition
    """

    def __init__(
        self,
        pipeline_name: str,
        step_name: str,
        code_hash: str,
        config_hash: str,
        pipeline_definition_config: PipelineDefinitionConfig,
    ):
        self.pipeline_name = pipeline_name
        self.step_name = step_name
        self.code_hash = code_hash
        self.config_hash = config_hash
        self.pipeline_definition_config = pipeline_definition_config


class PipelineSession(Session):
    """Managing interactions with SageMaker APIs and AWS services needed under Pipeline Context

    This class inherits the SageMaker session, it provides convenient methods
    for manipulating entities and resources that Amazon SageMaker uses,
    such as training jobs, endpoints, and input datasets in S3. When composing
    SageMaker Model-Building Pipeline, PipelineSession is recommended over
    regular SageMaker Session
    """

    def __init__(
        self,
        boto_session=None,
        sagemaker_client=None,
        default_bucket=None,
        settings=SessionSettings(),
        sagemaker_config: dict = None,
        default_bucket_prefix: str = None,
    ):
        """Initialize a ``PipelineSession``.

        Args:
            boto_session (boto3.session.Session): The underlying Boto3 session which AWS service
                calls are delegated to (default: None). If not provided, one is created with
                default AWS configuration chain.
            sagemaker_client (boto3.SageMaker.Client): Client which makes Amazon SageMaker service
                calls other than ``InvokeEndpoint`` (default: None). Estimators created using this
                ``Session`` use this client. If not provided, one will be created using this
                instance's ``boto_session``.
            default_bucket (str): The default Amazon S3 bucket to be used by this session.
                This will be created the next time an Amazon S3 bucket is needed (by calling
                :func:`default_bucket`).
                If not provided, a default bucket will be created based on the following format:
                "sagemaker-{region}-{aws-account-id}".
                Example: "sagemaker-my-custom-bucket".
            settings (sagemaker.session_settings.SessionSettings): Optional. Set of optional
                parameters to apply to the session.
            sagemaker_config: A dictionary containing default values for the
                SageMaker Python SDK. (default: None). The dictionary must adhere to the schema
                defined at `~sagemaker.config.config_schema.SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA`.
                If sagemaker_config is not provided and configuration files exist (at the default
                paths for admins and users, or paths set through the environment variables
                SAGEMAKER_ADMIN_CONFIG_OVERRIDE and SAGEMAKER_USER_CONFIG_OVERRIDE),
                a new dictionary will be generated from those configuration files. Alternatively,
                this dictionary can be generated by calling
                :func:`~sagemaker.config.load_sagemaker_config` and then be provided to the
                Session.
            default_bucket_prefix (str): The default prefix to use for S3 Object Keys. When
                objects are saved to the Session's default_bucket, the Object Key used will
                start with the default_bucket_prefix. If not provided here or within
                sagemaker_config, no additional prefix will be added.
        """
        super().__init__(
            boto_session=boto_session,
            sagemaker_client=sagemaker_client,
            default_bucket=default_bucket,
            settings=settings,
            sagemaker_config=sagemaker_config,
            default_bucket_prefix=default_bucket_prefix,
        )
        self._context = None

    @property
    def context(self):
        """Hold contextual information useful to the session"""
        return self._context

    @context.setter
    def context(self, value: Optional[_StepArguments] = None):
        self._context = value

    def _intercept_create_request(self, request: Dict, create, func_name: str = None):
        """This function intercepts the create job request

        Args:
            request (dict): the create job request
            create (functor): a functor calls the sagemaker client create method
            func_name (str): the name of the function needed intercepting
        """
        if func_name == self.create_model.__name__:
            self.context.create_model_request = request
            self.context.caller_name = func_name
        elif func_name == self.create_model_package_from_containers.__name__:
            self.context.create_model_package_request = request
            self.context.caller_name = func_name
        else:
            self.context = _JobStepArguments(func_name, request)

    def init_model_step_arguments(self, model):
        """Create a `_ModelStepArguments` (if not exist) as pipeline context

        Args:
            model (Model or PipelineModel): A `sagemaker.model.Model`
                or `sagemaker.pipeline.PipelineModel` instance
        """
        if not isinstance(self._context, _ModelStepArguments):
            self._context = _ModelStepArguments(model)


class LocalPipelineSession(LocalSession, PipelineSession):
    """Managing a session that executes Sagemaker pipelines and jobs locally in a pipeline context.

    This class inherits from the LocalSession and PipelineSession classes.
    When running Sagemaker pipelines locally, this class is preferred over LocalSession.
    """

    def __init__(
        self,
        boto_session=None,
        default_bucket=None,
        s3_endpoint_url=None,
        disable_local_code=False,
        default_bucket_prefix=None,
    ):
        """Initialize a ``LocalPipelineSession``.

        Args:
            boto_session (boto3.session.Session): The underlying Boto3 session which AWS service
                calls are delegated to (default: None). If not provided, one is created with
                default AWS configuration chain.
            default_bucket (str): The default Amazon S3 bucket to be used by this session.
                This will be created the next time an Amazon S3 bucket is needed (by calling
                :func:`default_bucket`).
                If not provided, a default bucket will be created based on the following format:
                "sagemaker-{region}-{aws-account-id}".
                Example: "sagemaker-my-custom-bucket".
            s3_endpoint_url (str): Override the default endpoint URL for Amazon S3,
                if set (default: None).
            disable_local_code (bool): Set to True to override the default AWS configuration chain
                to disable the `local.local_code` setting, which may not be supported for some SDK
                features (default: False).
            default_bucket_prefix (str): The default prefix to use for S3 Object Keys. When
                objects are saved to the Session's default_bucket, the Object Key used will
                start with the default_bucket_prefix. If not provided here or within
                sagemaker_config, no additional prefix will be added.
        """

        super().__init__(
            boto_session=boto_session,
            default_bucket=default_bucket,
            s3_endpoint_url=s3_endpoint_url,
            disable_local_code=disable_local_code,
            default_bucket_prefix=default_bucket_prefix,
        )


def runnable_by_pipeline(run_func):
    """A convenient Decorator

    This is a decorator designed to annotate, during pipeline session,
    the methods that downstream managed to
        1. preprocess user inputs, outputs, and configurations
        2. generate the create request
        3. start the job.
    For instance, `Processor.run`, `Estimator.fit`, or `Transformer.transform`.
    This decorator will essentially run 1, and capture the request shape from 2,
    then instead of starting a new job in 3, it will return request shape from 2
    to `sagemaker.workflow.steps.Step`. The request shape will be used to construct
    the arguments needed to compose that particular step as part of the pipeline.
    The job will be started during pipeline execution.
    """

    @wraps(run_func)
    def wrapper(*args, **kwargs):
        self_instance = args[0]
        if isinstance(self_instance.sagemaker_session, PipelineSession):
            run_func_params = inspect.signature(run_func).parameters
            arg_list = list(args)

            override_wait, override_logs = False, False
            for i, (arg_name, _) in enumerate(run_func_params.items()):
                if i >= len(arg_list):
                    break
                if arg_name == "wait":
                    override_wait = True
                    arg_list[i] = False
                elif arg_name == "logs":
                    override_logs = True
                    arg_list[i] = False

            args = tuple(arg_list)

            if not override_wait and "wait" in run_func_params.keys():
                kwargs["wait"] = False
            if not override_logs and "logs" in run_func_params.keys():
                kwargs["logs"] = False

            warnings.warn(
                "Running within a PipelineSession, there will be No Wait, "
                "No Logs, and No Job being started.",
                UserWarning,
            )
            if run_func.__name__ in ["register", "create"]:
                self_instance.sagemaker_session.init_model_step_arguments(self_instance)
                run_func(*args, **kwargs)
                context = self_instance.sagemaker_session.context
                self_instance.sagemaker_session.context = None
                return context

            return _StepArguments(retrieve_caller_name(self_instance), run_func, *args, **kwargs)

        return run_func(*args, **kwargs)

    return wrapper


def retrieve_caller_name(job_instance):
    """Convenience method for runnable_by_pipeline decorator

    This function takes an instance of a job class and maps it
    to the pipeline session function that creates the job request.

    Args:
        job_instance: A job class instance, one of the following
            imported types
    """

    from sagemaker.processing import Processor
    from sagemaker.estimator import EstimatorBase
    from sagemaker.transformer import Transformer
    from sagemaker.tuner import HyperparameterTuner
    from sagemaker.automl.automl import AutoML

    if isinstance(job_instance, Processor):
        return "process"
    if isinstance(job_instance, EstimatorBase):
        return "train"
    if isinstance(job_instance, Transformer):
        return "transform"
    if isinstance(job_instance, HyperparameterTuner):
        return "create_tuning_job"
    if isinstance(job_instance, AutoML):
        return "auto_ml"

    return None
