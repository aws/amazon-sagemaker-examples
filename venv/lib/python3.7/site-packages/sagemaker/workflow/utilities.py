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
"""Utilities to support workflow."""
from __future__ import absolute_import

import inspect
import logging
from functools import wraps
from pathlib import Path
from typing import List, Sequence, Union, Set, TYPE_CHECKING
import hashlib
from urllib.parse import unquote, urlparse
from contextlib import contextmanager
from _hashlib import HASH as Hash

from sagemaker.utils import base_from_name
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.pipeline_context import _StepArguments, _PipelineConfig
from sagemaker.workflow.entities import (
    Entity,
    RequestType,
)
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig

logger = logging.getLogger(__name__)

DEF_CONFIG_WARN_MSG_TEMPLATE = (
    "Popping out '%s' from the pipeline definition by default "
    "since it will be overridden at pipeline execution time. Please utilize "
    "the PipelineDefinitionConfig to persist this field in the pipeline definition "
    "if desired."
)

if TYPE_CHECKING:
    from sagemaker.workflow.step_collections import StepCollection

BUF_SIZE = 65536  # 64KiB
_pipeline_config: _PipelineConfig = None


def list_to_request(entities: Sequence[Union[Entity, "StepCollection"]]) -> List[RequestType]:
    """Get the request structure for list of entities.

    Args:
        entities (Sequence[Entity]): A list of entities.
    Returns:
        list: A request structure for a workflow service call.
    """
    from sagemaker.workflow.step_collections import StepCollection

    request_dicts = []
    for entity in entities:
        if isinstance(entity, Entity):
            request_dicts.append(entity.to_request())
        elif isinstance(entity, StepCollection):
            request_dicts.extend(entity.request_dicts())
    return request_dicts


@contextmanager
def _pipeline_config_manager(
    pipeline_name: str,
    step_name: str,
    code_hash: str,
    config_hash: str,
    pipeline_definition_config: PipelineDefinitionConfig,
):
    """Expose static _pipeline_config variable to other modules

    Args:
        pipeline_name (str): pipeline name
        step_name (str): step name
        code_hash (str): a hash of the code artifact for the particular step
        config_hash (str): a hash of the config artifact for the particular step (Processing)
    """

    # pylint: disable=W0603
    global _pipeline_config
    _pipeline_config = _PipelineConfig(
        pipeline_name, step_name, code_hash, config_hash, pipeline_definition_config
    )
    try:
        yield
    finally:
        _pipeline_config = None


def build_steps(
    steps: Sequence[Entity],
    pipeline_name: str,
    pipeline_definition_config: PipelineDefinitionConfig,
):
    """Get the request structure for list of steps, with _pipeline_config_manager

    Args:
        steps (Sequence[Entity]): A list of steps, (Entity type because Step causes circular import)
        pipeline_name (str): The name of the pipeline, passed down from pipeline.to_request()
        pipeline_definition_config (PipelineDefinitionConfig): A pipeline definition configuration
            for a pipeline containing feature flag toggles
    Returns:
        list: A request structure object for a service call for the list of pipeline steps
    """
    from sagemaker.workflow.step_collections import StepCollection

    request_dicts = []
    for step in steps:
        with _pipeline_config_manager(
            pipeline_name,
            step.name,
            get_code_hash(step),
            get_config_hash(step),
            pipeline_definition_config,
        ):
            if isinstance(step, StepCollection):
                request_dicts.extend(step.request_dicts())
            else:
                request_dicts.append(step.to_request())
    return request_dicts


def get_code_hash(step: Entity) -> str:
    """Get the hash of the code artifact(s) for the given step

    Args:
        step (Entity): A pipeline step object (Entity type because Step causes circular import)
    Returns:
        str: A hash string representing the unique code artifact(s) for the step
    """
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

    if isinstance(step, ProcessingStep) and step.step_args:
        kwargs = step.step_args.func_kwargs
        source_dir = kwargs.get("source_dir")
        submit_class = kwargs.get("submit_class")
        dependencies = get_processing_dependencies(
            [
                kwargs.get("dependencies"),
                kwargs.get("submit_py_files"),
                [submit_class] if submit_class else None,
                kwargs.get("submit_jars"),
                kwargs.get("submit_files"),
            ]
        )
        code = kwargs.get("submit_app") or kwargs.get("code")

        return get_processing_code_hash(code, source_dir, dependencies)

    if isinstance(step, TrainingStep) and step.step_args:
        job_obj = step.step_args.func_args[0]
        source_dir = job_obj.source_dir
        dependencies = job_obj.dependencies
        entry_point = job_obj.entry_point

        return get_training_code_hash(entry_point, source_dir, dependencies)
    return None


def get_processing_dependencies(dependency_args: List[List[str]]) -> List[str]:
    """Get the Processing job dependencies from the processor run kwargs

    Args:
        dependency_args: A list of dependency args from processor.run()
    Returns:
        List[str]: A list of code dependencies for the job
    """

    dependencies = []
    for arg in dependency_args:
        if arg:
            dependencies += arg

    return dependencies


def get_processing_code_hash(code: str, source_dir: str, dependencies: List[str]) -> str:
    """Get the hash of a processing step's code artifact(s).

    Args:
        code (str): Path to a file with the processing script to run
        source_dir (str): Path to a directory with any other processing
                source code dependencies aside from the entry point file
        dependencies (str): A list of paths to directories (absolute
                or relative) with any additional libraries that will be exported
                to the container
    Returns:
        str: A hash string representing the unique code artifact(s) for the step
    """

    # FrameworkProcessor
    if source_dir:
        source_dir_url = urlparse(source_dir)
        if source_dir_url.scheme == "" or source_dir_url.scheme == "file":
            # Include code in the hash when possible
            if code:
                code_url = urlparse(code)
                if code_url.scheme == "" or code_url.scheme == "file":
                    return hash_files_or_dirs([code, source_dir] + dependencies)
            return hash_files_or_dirs([source_dir] + dependencies)
    # Other Processors - Spark, Script, Base, etc.
    if code:
        code_url = urlparse(code)
        if code_url.scheme == "" or code_url.scheme == "file":
            return hash_files_or_dirs([code] + dependencies)
    return None


def get_training_code_hash(entry_point: str, source_dir: str, dependencies: List[str]) -> str:
    """Get the hash of a training step's code artifact(s).

    Args:
        entry_point (str): The absolute or relative path to the local Python
                source file that should be executed as the entry point to
                training
        source_dir (str): Path to a directory with any other training source
                code dependencies aside from the entry point file
        dependencies (str): A list of paths to directories (absolute
                or relative) with any additional libraries that will be exported
                to the container
    Returns:
        str: A hash string representing the unique code artifact(s) for the step
    """
    from sagemaker.workflow import is_pipeline_variable

    if not is_pipeline_variable(source_dir) and not is_pipeline_variable(entry_point):
        if source_dir:
            source_dir_url = urlparse(source_dir)
            if source_dir_url.scheme == "" or source_dir_url.scheme == "file":
                return hash_files_or_dirs([source_dir] + dependencies)
        elif entry_point:
            entry_point_url = urlparse(entry_point)
            if entry_point_url.scheme == "" or entry_point_url.scheme == "file":
                return hash_files_or_dirs([entry_point] + dependencies)
    return None


def get_config_hash(step: Entity):
    """Get the hash of the config artifact(s) for the given step

    Args:
        step (Entity): A pipeline step object (Entity type because Step causes circular import)
    Returns:
        str: A hash string representing the unique config artifact(s) for the step
    """
    from sagemaker.workflow.steps import ProcessingStep

    if isinstance(step, ProcessingStep) and step.step_args:
        config = step.step_args.func_kwargs.get("configuration")
        if config:
            return hash_object(config)
    return None


def hash_object(obj) -> str:
    """Get the MD5 hash of an object.

    Args:
        obj (dict): The object
    Returns:
        str: The MD5 hash of the object
    """
    return hashlib.md5(str(obj).encode()).hexdigest()


def hash_file(path: str) -> str:
    """Get the MD5 hash of a file.

    Args:
        path (str): The local path for the file.
    Returns:
        str: The MD5 hash of the file.
    """
    return _hash_file(path, hashlib.md5()).hexdigest()


def hash_files_or_dirs(paths: List[str]) -> str:
    """Get the MD5 hash of the contents of a list of files or directories.

    Hash is changed if:
       * input list is changed
       * new nested directories/files are added to any directory in the input list
       * nested directory/file names are changed for any of the inputted directories
       * content of files is edited

    Args:
        paths: List of file or directory paths
    Returns:
        str: The MD5 hash of the list of files or directories.
    """
    md5 = hashlib.md5()
    for path in sorted(paths):
        md5 = _hash_file_or_dir(path, md5)
    return md5.hexdigest()


def _hash_file_or_dir(path: str, md5: Hash) -> Hash:
    """Updates the inputted Hash with the contents of the current path.

    Args:
        path: path of file or directory
    Returns:
        str: The MD5 hash of the file or directory
    """
    if isinstance(path, str) and path.lower().startswith("file://"):
        path = unquote(urlparse(path).path)
    md5.update(path.encode())
    if Path(path).is_dir():
        md5 = _hash_dir(path, md5)
    elif Path(path).is_file():
        md5 = _hash_file(path, md5)
    return md5


def _hash_dir(directory: Union[str, Path], md5: Hash) -> Hash:
    """Updates the inputted Hash with the contents of the current path.

    Args:
        directory: path of the directory
    Returns:
        str: The MD5 hash of the directory
    """
    if not Path(directory).is_dir():
        raise ValueError(str(directory) + " is not a valid directory")
    for path in sorted(Path(directory).iterdir()):
        md5.update(path.name.encode())
        if path.is_file():
            md5 = _hash_file(path, md5)
        elif path.is_dir():
            md5 = _hash_dir(path, md5)
    return md5


def _hash_file(file: Union[str, Path], md5: Hash) -> Hash:
    """Updates the inputted Hash with the contents of the current path.

    Args:
        file: path of the file
    Returns:
        str: The MD5 hash of the file
    """
    if isinstance(file, str) and file.lower().startswith("file://"):
        file = unquote(urlparse(file).path)
    if not Path(file).is_file():
        raise ValueError(str(file) + " is not a valid file")
    with open(file, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5


def validate_step_args_input(
    step_args: _StepArguments, expected_caller: Set[str], error_message: str
):
    """Validate the `_StepArguments` object which is passed into a pipeline step

    Args:
        step_args (_StepArguments): A `_StepArguments` object to be used for composing
            a pipeline step.
        expected_caller (Set[str]): The expected name of the caller function which is
            intercepted by the PipelineSession to get the step arguments.
        error_message (str): The error message to be thrown if the validation fails.
    """
    if not isinstance(step_args, _StepArguments):
        raise TypeError(error_message)
    if step_args.caller_name not in expected_caller:
        raise ValueError(error_message)


def override_pipeline_parameter_var(func):
    """A decorator to override pipeline Parameters passed into a function

    This is a temporary decorator to override pipeline Parameter objects with their default value
    and display warning information to instruct users to update their code.

    This decorator can help to give a grace period for users to update their code when
    we make changes to explicitly prevent passing any pipeline variables to a function.

    We should remove this decorator after the grace period.
    """
    warning_msg_template = (
        "The input argument %s of function (%s) is a pipeline variable (%s), "
        "which is interpreted in pipeline execution time only. "
        "As the function needs to evaluate the argument value in SDK compile time, "
        "the default_value of this Parameter object will be used to override it. "
        "Please make sure the default_value is valid."
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = "{}.{}".format(func.__module__, func.__name__)
        params = inspect.signature(func).parameters
        args = list(args)
        for i, (arg_name, _) in enumerate(params.items()):
            if i >= len(args):
                break
            if isinstance(args[i], Parameter):
                logger.warning(warning_msg_template, arg_name, func_name, type(args[i]))
                args[i] = args[i].default_value
        args = tuple(args)

        for arg_name, value in kwargs.items():
            if isinstance(value, Parameter):
                logger.warning(warning_msg_template, arg_name, func_name, type(value))
                kwargs[arg_name] = value.default_value
        return func(*args, **kwargs)

    return wrapper


def execute_job_functions(step_args: _StepArguments):
    """Execute the job class functions during pipeline definition construction

    Executes the job functions such as run(), fit(), or transform() that have been
    delayed until the pipeline gets built, for steps built with a PipelineSession.

    Handles multiple functions in instances where job functions are chained
    together from the inheritance of different job classes (e.g. PySparkProcessor,
    ScriptProcessor, and Processor).

    Args:
        step_args (_StepArguments): A `_StepArguments` object to be used for composing
            a pipeline step, contains the necessary function information
    """

    chained_args = step_args.func(*step_args.func_args, **step_args.func_kwargs)
    if isinstance(chained_args, _StepArguments):
        execute_job_functions(chained_args)


def trim_request_dict(request_dict, job_key, config):
    """Trim request_dict for unwanted fields to not persist them in step arguments

    Trim the job_name field off request_dict in cases where we do not want to include it
    in the pipeline definition.

    Args:
        request_dict (dict): A dictionary used to build the arguments for a pipeline step,
            containing fields that will be passed to job client during orchestration.
        job_key (str): The key in a step's arguments to look up the base_job_name if it
            exists
        config (_pipeline_config) The config intercepted and set for a pipeline via the
            context manager
    """

    if not config or not config.pipeline_definition_config.use_custom_job_prefix:
        logger.warning(DEF_CONFIG_WARN_MSG_TEMPLATE, job_key)
        request_dict.pop(job_key, None)  # safely return null in case of KeyError
    else:
        if job_key in request_dict:
            request_dict[job_key] = base_from_name(request_dict[job_key])  # trim timestamp

    return request_dict
