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
"""Placeholder docstring"""
from __future__ import absolute_import

import importlib
import inspect
import json
import logging

from enum import Enum
from typing import Union, Dict, Optional, List, Set

import sagemaker
from sagemaker.amazon.amazon_estimator import (
    RecordSet,
    AmazonAlgorithmEstimatorBase,
    FileSystemRecordSet,
)
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from sagemaker.deprecations import removed_function
from sagemaker.estimator import Framework, EstimatorBase
from sagemaker.inputs import TrainingInput, FileSystemInput
from sagemaker.job import _Job
from sagemaker.jumpstart.utils import (
    add_jumpstart_tags,
    get_jumpstart_base_name_if_jumpstart_model,
)
from sagemaker.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
    ParameterRange,
)
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.pipeline_context import runnable_by_pipeline

from sagemaker.session import Session
from sagemaker.utils import (
    base_from_name,
    base_name_from_image,
    name_from_base,
    to_string,
)

AMAZON_ESTIMATOR_MODULE = "sagemaker"
AMAZON_ESTIMATOR_CLS_NAMES = {
    "factorization-machines": "FactorizationMachines",
    "kmeans": "KMeans",
    "lda": "LDA",
    "linear-learner": "LinearLearner",
    "ntm": "NTM",
    "randomcutforest": "RandomCutForest",
    "knn": "KNN",
    "object2vec": "Object2Vec",
}
HYPERPARAMETER_TUNING_JOB_NAME = "HyperParameterTuningJobName"
PARENT_HYPERPARAMETER_TUNING_JOBS = "ParentHyperParameterTuningJobs"
WARM_START_TYPE = "WarmStartType"
HYPERBAND_STRATEGY_CONFIG = "HyperbandStrategyConfig"
HYPERBAND_MIN_RESOURCE = "MinResource"
HYPERBAND_MAX_RESOURCE = "MaxResource"
GRID_SEARCH = "Grid"
MAX_NUMBER_OF_TRAINING_JOBS_NOT_IMPROVING = "MaxNumberOfTrainingJobsNotImproving"
BEST_OBJECTIVE_NOT_IMPROVING = "BestObjectiveNotImproving"
CONVERGENCE_DETECTED = "ConvergenceDetected"
COMPLETE_ON_CONVERGENCE_DETECTED = "CompleteOnConvergence"
TARGET_OBJECTIVE_METRIC_VALUE = "TargetObjectiveMetricValue"
MAX_RUNTIME_IN_SECONDS = "MaxRuntimeInSeconds"

logger = logging.getLogger(__name__)


class WarmStartTypes(Enum):
    """Warm Start Configuration type.

    There can be two types of warm start jobs:

    * IdenticalDataAndAlgorithm: Type of warm start that allows users to reuse
    training results from existing tuning jobs that have the same algorithm code
    and datasets.
    * TransferLearning: Type of warm start that allows users to reuse training
    results from existing tuning jobs that have similar algorithm code and
    datasets.
    """

    IDENTICAL_DATA_AND_ALGORITHM = "IdenticalDataAndAlgorithm"
    TRANSFER_LEARNING = "TransferLearning"


class WarmStartConfig(object):
    """Warm Start Configuration which defines the nature of the warm start.

    This warm start configuration is provided to the ``HyperparameterTuner``,
    with type and parents for warm start.

    Examples:
        >>> warm_start_config = WarmStartConfig(
        >>>                         type=WarmStartTypes.TransferLearning, parents={"p1","p2"})
        >>> warm_start_config.type
        "TransferLearning"
        >>> warm_start_config.parents
        {"p1","p2"}
    """

    def __init__(
        self,
        warm_start_type: WarmStartTypes,
        parents: Set[Union[str, PipelineVariable]],
    ):
        """Creates a ``WarmStartConfig`` with provided ``WarmStartTypes`` and parents.

        Args:
            warm_start_type (sagemaker.tuner.WarmStartTypes): This should be one
                of the supported warm start types in WarmStartType
            parents (set[str] or set[PipelineVariable]): Set of parent tuning jobs which
                will be used to warm start the new tuning job.
        """

        if warm_start_type not in list(WarmStartTypes):
            raise ValueError(
                "Invalid type: {}, valid warm start types are: {}".format(
                    warm_start_type, list(WarmStartTypes)
                )
            )

        if not parents:
            raise ValueError(
                "Invalid parents: {}, parents should not be None/empty".format(parents)
            )

        self.type = warm_start_type
        self.parents = set(parents)

    @classmethod
    def from_job_desc(cls, warm_start_config):
        """Creates a ``WarmStartConfig`` from a warm start configuration response.

        This is the warm start configuration from the DescribeTrainingJob response.

        Examples:
            >>> warm_start_config = WarmStartConfig.from_job_desc(warm_start_config={
            >>>    "WarmStartType":"TransferLearning",
            >>>    "ParentHyperParameterTuningJobs": [
            >>>        {'HyperParameterTuningJobName': "p1"},
            >>>        {'HyperParameterTuningJobName': "p2"},
            >>>    ]
            >>>})
            >>> warm_start_config.type
            "TransferLearning"
            >>> warm_start_config.parents
            ["p1","p2"]

        Args:
            warm_start_config (dict): The expected format of the
                ``warm_start_config`` contains two first-class

        Returns:
            sagemaker.tuner.WarmStartConfig: De-serialized instance of
            WarmStartConfig containing the type and parents provided as part of
            ``warm_start_config``.
        """
        if (
            not warm_start_config
            or WARM_START_TYPE not in warm_start_config
            or PARENT_HYPERPARAMETER_TUNING_JOBS not in warm_start_config
        ):
            return None

        parents = []
        for parent in warm_start_config[PARENT_HYPERPARAMETER_TUNING_JOBS]:
            parents.append(parent[HYPERPARAMETER_TUNING_JOB_NAME])

        return cls(
            warm_start_type=WarmStartTypes(warm_start_config[WARM_START_TYPE]),
            parents=parents,
        )

    def to_input_req(self):
        """Converts the ``self`` instance to the desired input request format.

        Examples:
            >>> warm_start_config = WarmStartConfig
            (
                warm_start_type=WarmStartTypes.TransferLearning,parents=["p1,p2"]
            )
            >>> warm_start_config.to_input_req()
            {
                "WarmStartType":"TransferLearning",
                "ParentHyperParameterTuningJobs": [
                    {'HyperParameterTuningJobName': "p1"},
                    {'HyperParameterTuningJobName': "p2"},
                ]
            }

        Returns:
            dict: Containing the "WarmStartType" and
            "ParentHyperParameterTuningJobs" as the first class fields.
        """
        return {
            WARM_START_TYPE: self.type.value,
            PARENT_HYPERPARAMETER_TUNING_JOBS: [
                {HYPERPARAMETER_TUNING_JOB_NAME: parent} for parent in self.parents
            ],
        }


class HyperbandStrategyConfig(object):
    """The configuration for Hyperband, a multi-fidelity based hyperparameter tuning strategy.

    Hyperband uses the final and intermediate results of a training job to dynamically allocate
    resources to hyperparameter configurations being evaluated while automatically stopping
    under-performing configurations. This parameter should be provided only if Hyperband is
    selected as the Strategy under the HyperParameterTuningJobConfig.

    Examples:
        >>> hyperband_strategy_config = HyperbandStrategyConfig(
        >>>                                 max_resource=10, min_resource = 1)
        >>> hyperband_strategy_config.max_resource
        10
        >>> hyperband_strategy_config.min_resource
        1
    """

    def __init__(self, max_resource: int, min_resource: int):
        """Creates a ``HyperbandStrategyConfig`` with provided `min_resource`` and ``max_resource``.

        Args:
            max_resource (int): The maximum number of resources (such as epochs) that can be used
            by a training job launched by a hyperparameter tuning job.
                Once a job reaches the MaxResource value, it is stopped.
                If a value for MaxResource is not provided, and Hyperband is selected as the
                hyperparameter tuning strategy, HyperbandTrainingJ attempts to infer MaxResource
                from the following keys (if present) in StaticsHyperParameters:
                    epochs
                    numepochs
                    n-epochs
                    n_epochs
                    num_epochs
                If HyperbandStrategyConfig is unable to infer a value for MaxResource, it generates
                a validation error.
                The maximum value is 20,000 epochs. All metrics that correspond to an objective
                metric are used to derive early stopping decisions.
                For distributed training jobs, ensure that duplicate metrics are not printed in the
                logs across the individual nodes in a training job.
                If multiple nodes are publishing duplicate or incorrect metrics, hyperband
                optimisation algorithm may make an incorrect stopping decision and stop the job
                prematurely.
            min_resource (int): The minimum number of resources (such as epochs)
                that can be used by a training job launched by a hyperparameter tuning job.
                If the value for MinResource has not been reached, the training job will not be
                stopped by Hyperband.
        """
        self.min_resource = min_resource
        self.max_resource = max_resource

    @classmethod
    def from_job_desc(cls, hyperband_strategy_config):
        """Creates a ``HyperbandStrategyConfig`` from a hyperband strategy configuration response.

        This is the Hyperband strategy configuration from the DescribeTuningJob response.

        Examples:
            >>> hyperband_strategy_config =
            >>>     HyperbandStrategyConfig.from_job_desc(hyperband_strategy_config={
            >>>         "MaxResource": 10,
            >>>         "MinResource": 1
            >>>     })
            >>> hyperband_strategy_config.max_resource
            10
            >>> hyperband_strategy_config.min_resource
            1

        Args:
            hyperband_strategy_config (dict): The expected format of the
                ``hyperband_strategy_config`` contains two first-class fields

        Returns:
            sagemaker.tuner.HyperbandStrategyConfig: De-serialized instance of
                ``HyperbandStrategyConfig`` containing the max_resource
                and min_resource provided as part of ``hyperband_strategy_config``.
        """
        return cls(
            min_resource=hyperband_strategy_config[HYPERBAND_MIN_RESOURCE],
            max_resource=hyperband_strategy_config[HYPERBAND_MAX_RESOURCE],
        )

    def to_input_req(self):
        """Converts the ``self`` instance to the desired input request format.

        Examples:
            >>> hyperband_strategy_config = HyperbandStrategyConfig (
                max_resource=10,
                min_resource=1
            )
            >>> hyperband_strategy_config.to_input_req()
            {
                "MaxResource":10,
                "MinResource": 1
            }

        Returns:
            dict: Containing the "MaxResource" and
                "MinResource" as the first class fields.
        """
        return {
            HYPERBAND_MIN_RESOURCE: self.min_resource,
            HYPERBAND_MAX_RESOURCE: self.max_resource,
        }


class StrategyConfig(object):
    """The configuration for a training job launched by a hyperparameter tuning job.

    Choose Bayesian for Bayesian optimization, and Random for random search optimization.
    For more advanced use cases, use Hyperband, which evaluates objective metrics for training jobs
    after every epoch.
    """

    def __init__(
        self,
        hyperband_strategy_config: HyperbandStrategyConfig,
    ):
        """Creates a ``StrategyConfig`` with provided ``HyperbandStrategyConfig``.

        Args:
            hyperband_strategy_config (sagemaker.tuner.HyperbandStrategyConfig): The configuration
                for the object that specifies the Hyperband strategy.
                This parameter is only supported for the Hyperband selection for Strategy within
                the HyperParameterTuningJobConfig.
        """

        self.hyperband_strategy_config = hyperband_strategy_config

    @classmethod
    def from_job_desc(cls, strategy_config):
        """Creates a ``HyperbandStrategyConfig`` from a hyperband strategy configuration response.

        This is the hyper band strategy configuration from the DescribeTuningJob response.

        Args:
            strategy_config (dict): The expected format of the
                ``strategy_config`` contains one first-class field

        Returns:
            sagemaker.tuner.StrategyConfig: De-serialized instance of
            StrategyConfig containing the strategy configuration.
        """
        return cls(
            hyperband_strategy_config=HyperbandStrategyConfig.from_job_desc(
                strategy_config[HYPERBAND_STRATEGY_CONFIG]
            )
        )

    def to_input_req(self):
        """Converts the ``self`` instance to the desired input request format.

        Examples:
            >>> strategy_config = StrategyConfig(
                HyperbandStrategyConfig(
                    max_resource=10,
                    min_resource=1
                )
            )
            >>> strategy_config.to_input_req()
            {
                "HyperbandStrategyConfig": {
                    "MaxResource":10,
                    "MinResource": 1
                }
            }

        Returns:
            dict: Containing the strategy configurations.
        """
        return {
            HYPERBAND_STRATEGY_CONFIG: self.hyperband_strategy_config.to_input_req(),
        }


class InstanceConfig:
    """Instance configuration for training jobs started by hyperparameter tuning.

    Contains the configuration(s) for one or more resources for processing hyperparameter jobs.
    These resources include compute instances and storage volumes to use in model training jobs
    launched by hyperparameter tuning jobs.
    """

    def __init__(
        self,
        instance_count: Union[int, PipelineVariable] = None,
        instance_type: Union[str, PipelineVariable] = None,
        volume_size: Union[int, PipelineVariable] = 30,
    ):
        """Creates a ``InstanceConfig`` instance.

        It takes instance configuration information for training
        jobs that are created as the result of a hyperparameter tuning job.

        Args:
            * instance_count (str or PipelineVariable): The number of compute instances of type
            InstanceType to use. For distributed training, select a value greater than 1.
            * instance_type (str or PipelineVariable):
            The instance type used to run hyperparameter optimization tuning jobs.
            * volume_size (int or PipelineVariable): The volume size in GB of the data to be
            processed for hyperparameter optimization
        """
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.volume_size = volume_size

    @classmethod
    def from_job_desc(cls, instance_config):
        """Creates a ``InstanceConfig`` from an instance configuration response.

        This is the instance configuration from the DescribeTuningJob response.

        Args:
            instance_config (dict): The expected format of the
                ``instance_config`` contains one first-class field

        Returns:
            sagemaker.tuner.InstanceConfig: De-serialized instance of
            InstanceConfig containing the strategy configuration.
        """
        return cls(
            instance_count=instance_config["InstanceCount"],
            instance_type=instance_config[" InstanceType "],
            volume_size=instance_config["VolumeSizeInGB"],
        )

    def to_input_req(self):
        """Converts the ``self`` instance to the desired input request format.

        Examples:
            >>> strategy_config = InstanceConfig(
                instance_count=1,
                instance_type='ml.m4.xlarge',
                volume_size=30
            )
            >>> strategy_config.to_input_req()
            {
                "InstanceCount":1,
                "InstanceType":"ml.m4.xlarge",
                "VolumeSizeInGB":30
            }

        Returns:
            dict: Containing the instance configurations.
        """
        return {
            "InstanceCount": self.instance_count,
            "InstanceType": self.instance_type,
            "VolumeSizeInGB": self.volume_size,
        }


class TuningJobCompletionCriteriaConfig(object):
    """The configuration for a job completion criteria."""

    def __init__(
        self,
        max_number_of_training_jobs_not_improving: int = None,
        complete_on_convergence: bool = None,
        target_objective_metric_value: float = None,
    ):
        """Creates a ``TuningJobCompletionCriteriaConfig`` with provided criteria.

        Args:
            max_number_of_training_jobs_not_improving (int): The number of training jobs that do not
                improve the best objective after which tuning job will stop.
            complete_on_convergence (bool): A flag to stop your hyperparameter tuning job if
                automatic model tuning (AMT) has detected that your model has converged as evaluated
                against your objective function.
            target_objective_metric_value (float): The value of the objective metric.
        """

        self.max_number_of_training_jobs_not_improving = max_number_of_training_jobs_not_improving
        self.complete_on_convergence = complete_on_convergence
        self.target_objective_metric_value = target_objective_metric_value

    @classmethod
    def from_job_desc(cls, completion_criteria_config):
        """Creates a ``TuningJobCompletionCriteriaConfig`` from a configuration response.

        This is the completion criteria configuration from the DescribeTuningJob response.
        Args:
            completion_criteria_config (dict): The expected format of the
                ``completion_criteria_config`` contains three first-class fields

        Returns:
            sagemaker.tuner.TuningJobCompletionCriteriaConfig: De-serialized instance of
            TuningJobCompletionCriteriaConfig containing the completion criteria.
        """
        complete_on_convergence = None
        if CONVERGENCE_DETECTED in completion_criteria_config:
            if completion_criteria_config[CONVERGENCE_DETECTED][COMPLETE_ON_CONVERGENCE_DETECTED]:
                complete_on_convergence = bool(
                    completion_criteria_config[CONVERGENCE_DETECTED][
                        COMPLETE_ON_CONVERGENCE_DETECTED
                    ]
                    == "Enabled"
                )

        max_number_of_training_jobs_not_improving = None
        if BEST_OBJECTIVE_NOT_IMPROVING in completion_criteria_config:
            if completion_criteria_config[BEST_OBJECTIVE_NOT_IMPROVING][
                MAX_NUMBER_OF_TRAINING_JOBS_NOT_IMPROVING
            ]:
                max_number_of_training_jobs_not_improving = completion_criteria_config[
                    BEST_OBJECTIVE_NOT_IMPROVING
                ][MAX_NUMBER_OF_TRAINING_JOBS_NOT_IMPROVING]

        target_objective_metric_value = None
        if TARGET_OBJECTIVE_METRIC_VALUE in completion_criteria_config:
            target_objective_metric_value = completion_criteria_config[
                TARGET_OBJECTIVE_METRIC_VALUE
            ]

        return cls(
            max_number_of_training_jobs_not_improving=max_number_of_training_jobs_not_improving,
            complete_on_convergence=complete_on_convergence,
            target_objective_metric_value=target_objective_metric_value,
        )

    def to_input_req(self):
        """Converts the ``self`` instance to the desired input request format.

        Examples:
            >>> completion_criteria_config = TuningJobCompletionCriteriaConfig(
                max_number_of_training_jobs_not_improving=5
                complete_on_convergence = True,
                target_objective_metric_value = 0.42
            )
            >>> completion_criteria_config.to_input_req()
            {
                "BestObjectiveNotImproving": {
                    "MaxNumberOfTrainingJobsNotImproving":5
                },
                "ConvergenceDetected": {
                    "CompleteOnConvergence": "Enabled",
                },
                "TargetObjectiveMetricValue": 0.42
            }

        Returns:
            dict: Containing the completion criteria configurations.
        """
        completion_criteria_config = {}
        if self.max_number_of_training_jobs_not_improving is not None:
            completion_criteria_config[BEST_OBJECTIVE_NOT_IMPROVING] = {}
            completion_criteria_config[BEST_OBJECTIVE_NOT_IMPROVING][
                MAX_NUMBER_OF_TRAINING_JOBS_NOT_IMPROVING
            ] = self.max_number_of_training_jobs_not_improving

        if self.target_objective_metric_value is not None:
            completion_criteria_config[
                TARGET_OBJECTIVE_METRIC_VALUE
            ] = self.target_objective_metric_value

        if self.complete_on_convergence is not None:
            completion_criteria_config[CONVERGENCE_DETECTED] = {}
            completion_criteria_config[CONVERGENCE_DETECTED][COMPLETE_ON_CONVERGENCE_DETECTED] = (
                "Enabled" if self.complete_on_convergence else "Disabled"
            )

        return completion_criteria_config


class HyperparameterTuner(object):
    """Defines interaction with Amazon SageMaker hyperparameter tuning jobs.

    It also supports deploying the resulting models.
    """

    TUNING_JOB_NAME_MAX_LENGTH = 32

    SAGEMAKER_ESTIMATOR_MODULE = "sagemaker_estimator_module"
    SAGEMAKER_ESTIMATOR_CLASS_NAME = "sagemaker_estimator_class_name"

    DEFAULT_ESTIMATOR_MODULE = "sagemaker.estimator"
    DEFAULT_ESTIMATOR_CLS_NAME = "Estimator"

    def __init__(
        self,
        estimator: EstimatorBase,
        objective_metric_name: Union[str, PipelineVariable],
        hyperparameter_ranges: Dict[str, ParameterRange],
        metric_definitions: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        strategy: Union[str, PipelineVariable] = "Bayesian",
        objective_type: Union[str, PipelineVariable] = "Maximize",
        max_jobs: Union[int, PipelineVariable] = None,
        max_parallel_jobs: Union[int, PipelineVariable] = 1,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        base_tuning_job_name: Optional[str] = None,
        warm_start_config: Optional[WarmStartConfig] = None,
        strategy_config: Optional[StrategyConfig] = None,
        completion_criteria_config: Optional[TuningJobCompletionCriteriaConfig] = None,
        early_stopping_type: Union[str, PipelineVariable] = "Off",
        estimator_name: Optional[str] = None,
        random_seed: Optional[int] = None,
        autotune: bool = False,
        hyperparameters_to_keep_static: Optional[List[str]] = None,
    ):
        """Creates a ``HyperparameterTuner`` instance.

        It takes an estimator to obtain configuration information for training
        jobs that are created as the result of a hyperparameter tuning job.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): An estimator object
                that has been initialized with the desired configuration. There
                does not need to be a training job associated with this
                instance.
            objective_metric_name (str or PipelineVariable): Name of the metric for evaluating
                training jobs.
            hyperparameter_ranges (dict[str, sagemaker.parameter.ParameterRange]): Dictionary of
                parameter ranges. These parameter ranges can be one
                of three types: Continuous, Integer, or Categorical. The keys of
                the dictionary are the names of the hyperparameter, and the
                values are the appropriate parameter range class to represent
                the range.
            metric_definitions (list[dict[str, str] or list[dict[str, PipelineVariable]]): A list of
                dictionaries that defines the metric(s) used to evaluate the training jobs (default:
                None). Each dictionary contains two keys: 'Name' for the name of
                the metric, and 'Regex' for the regular expression used to
                extract the metric from the logs. This should be defined only
                for hyperparameter tuning jobs that don't use an Amazon
                algorithm.
            strategy (str or PipelineVariable): Strategy to be used for hyperparameter estimations
                (default: 'Bayesian').
            objective_type (str or PipelineVariable): The type of the objective metric for
                evaluating training jobs. This value can be either 'Minimize' or
                'Maximize' (default: 'Maximize').
            max_jobs (int or PipelineVariable): Maximum total number of training jobs to start for
                the hyperparameter tuning job. The default value is unspecified fot the 'Grid'
                strategy and the default value is 1 for all others strategies (default: None).
            max_parallel_jobs (int or PipelineVariable): Maximum number of parallel training jobs to
                start (default: 1).
            max_runtime_in_seconds (int or PipelineVariable): The maximum time in seconds
                 that a hyperparameter tuning job can run.
            tags (list[dict[str, str] or list[dict[str, PipelineVariable]]): List of tags for
                labeling the tuning job (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            base_tuning_job_name (str): Prefix for the hyperparameter tuning job
                name when the :meth:`~sagemaker.tuner.HyperparameterTuner.fit`
                method launches. If not specified, a default job name is
                generated, based on the training image name and current
                timestamp.
            warm_start_config (sagemaker.tuner.WarmStartConfig): A
                ``WarmStartConfig`` object that has been initialized with the
                configuration defining the nature of warm start tuning job.
            strategy_config (sagemaker.tuner.StrategyConfig): A configuration for "Hyperparameter"
                tuning job optimisation strategy.
            completion_criteria_config (sagemaker.tuner.TuningJobCompletionCriteriaConfig): A
                 configuration for the completion criteria.
            early_stopping_type (str or PipelineVariable): Specifies whether early stopping is
                enabled for the job. Can be either 'Auto' or 'Off' (default:
                'Off'). If set to 'Off', early stopping will not be attempted.
                If set to 'Auto', early stopping of some training jobs may
                happen, but is not guaranteed to.
            estimator_name (str): A unique name to identify an estimator within the
                hyperparameter tuning job, when more than one estimator is used with
                the same tuning job (default: None).
            random_seed (int): An initial value used to initialize a pseudo-random number generator.
                Setting a random seed will make the hyperparameter tuning search strategies to
                produce more consistent configurations for the same tuning job.
            autotune (bool): Whether the parameter ranges or other unset settings of a tuning job
                should be chosen automatically (default: False).
            hyperparameters_to_keep_static: list[str]: Names of hyperparameters that will be kept
                static and will not be assigned a tunable range with Autotune functionality.
                (default: None).
        """
        if hyperparameter_ranges is None or len(hyperparameter_ranges) == 0:
            if not autotune:
                raise ValueError("Need to specify hyperparameter ranges or set autotune=True.")

        if not autotune and hyperparameters_to_keep_static is not None:
            raise ValueError(
                "hyperparameters_to_keep_static parameter is set, however Autotune mode is not "
                "enabled. Either do not set value for hyperparameters_to_keep_static parameter, "
                "or enable Autotune mode by setting autotune=True."
            )

        if hyperparameters_to_keep_static is not None:
            if len(hyperparameters_to_keep_static) != len(set(hyperparameters_to_keep_static)):
                raise ValueError("Please remove duplicate names in hyperparameters_to_keep_static.")

        if estimator_name is not None:
            self.estimator = None
            self.objective_metric_name = None
            self._hyperparameter_ranges = None
            self.metric_definitions = None
            self.estimator_dict = {estimator_name: estimator}
            self.objective_metric_name_dict = {estimator_name: objective_metric_name}
            self._hyperparameter_ranges_dict = {estimator_name: hyperparameter_ranges}
            self.metric_definitions_dict = (
                {estimator_name: metric_definitions} if metric_definitions is not None else {}
            )
            self.static_hyperparameters = None
            self.auto_parameters = None
            self.auto_parameters_dict = None
            self.hyperparameters_to_keep_static = None
            self.hyperparameters_to_keep_static_dict = {
                estimator_name: hyperparameters_to_keep_static
            }
        else:
            self.estimator = estimator
            self.objective_metric_name = objective_metric_name
            self._hyperparameter_ranges = hyperparameter_ranges
            self.metric_definitions = metric_definitions
            self.estimator_dict = None
            self.objective_metric_name_dict = None
            self._hyperparameter_ranges_dict = None
            self.metric_definitions_dict = None
            self.static_hyperparameters_dict = None
            self.auto_parameters = None
            self.auto_parameters_dict = None
            self.hyperparameters_to_keep_static = hyperparameters_to_keep_static
            self.hyperparameters_to_keep_static_dict = None

        self._validate_parameter_ranges(estimator, hyperparameter_ranges)

        self.strategy = strategy
        self.strategy_config = strategy_config
        self.completion_criteria_config = completion_criteria_config
        self.objective_type = objective_type
        # For the GridSearch strategy we expect the max_jobs equals None and recalculate it later.
        # For all other strategies for the backward compatibility we keep
        # the default value as 1 (previous default value).
        self.max_jobs = max_jobs
        if max_jobs is None and strategy != GRID_SEARCH:
            self.max_jobs = 1
        self.max_parallel_jobs = max_parallel_jobs
        self.max_runtime_in_seconds = max_runtime_in_seconds

        self.tags = tags
        self.base_tuning_job_name = base_tuning_job_name
        self._current_job_name = None
        self.latest_tuning_job = None
        self.warm_start_config = warm_start_config
        self.early_stopping_type = early_stopping_type
        self.random_seed = random_seed
        self.instance_configs_dict = None
        self.instance_configs = None
        self.autotune = autotune

    def override_resource_config(
        self, instance_configs: Union[List[InstanceConfig], Dict[str, List[InstanceConfig]]]
    ):
        """Override the instance configuration of the estimators used by the tuner.

        Args:
            instance_configs (List[InstanceConfig] or Dict[str, List[InstanceConfig]):
                The InstanceConfigs to use as an override for the instance configuration
                of the estimator. ``None`` will remove the override.
        """
        if isinstance(instance_configs, dict):
            self._validate_dict_argument(
                name="instance_configs",
                value=instance_configs,
                allowed_keys=list(self.estimator_dict.keys()),
            )
            self.instance_configs_dict = instance_configs
        else:
            self.instance_configs = instance_configs
            if self.estimator_dict is not None and self.estimator_dict.keys():
                estimator_names = list(self.estimator_dict.keys())
                self.instance_configs_dict = {estimator_names[0]: instance_configs}

    def _prepare_for_tuning(self, job_name=None, include_cls_metadata=False):
        """Prepare the tuner instance for tuning (fit)."""
        self._prepare_job_name_for_tuning(job_name=job_name)
        self._prepare_static_hyperparameters_for_tuning(include_cls_metadata=include_cls_metadata)
        self._prepare_auto_parameters_for_tuning()
        self._prepare_tags_for_tuning()

    def _get_model_uri(
        self,
        estimator,
    ):
        """Return the model artifact URI used by the Estimator instance.

        This attribute can live in multiple places, and accessing the attribute can
        raise a TypeError, which needs to be handled.
        """
        try:
            return getattr(estimator, "model_data", None)
        except TypeError:
            return getattr(estimator, "model_uri", None)

    def _prepare_tags_for_tuning(self):
        """Add tags to tuning job (from Estimator and JumpStart tags)."""

        # Add tags from Estimator class
        estimator = self.estimator or self.estimator_dict[sorted(self.estimator_dict.keys())[0]]

        estimator_tags = getattr(estimator, "tags", []) or []

        if self.tags is None and len(estimator_tags) > 0:
            self.tags = []

        for tag in estimator_tags:
            if tag not in self.tags:
                self.tags.append(tag)

        self.tags = add_jumpstart_tags(
            tags=self.tags,
            training_script_uri=getattr(estimator, "source_dir", None),
            training_model_uri=self._get_model_uri(estimator),
        )

    def _prepare_job_name_for_tuning(self, job_name=None):
        """Set current job name before starting tuning."""
        if job_name is not None:
            self._current_job_name = job_name
        else:
            base_name = self.base_tuning_job_name
            if base_name is None:
                estimator = (
                    self.estimator or self.estimator_dict[sorted(self.estimator_dict.keys())[0]]
                )
                base_name = base_name_from_image(
                    estimator.training_image_uri(),
                    default_base_name=EstimatorBase.JOB_CLASS_NAME,
                )

                jumpstart_base_name = get_jumpstart_base_name_if_jumpstart_model(
                    getattr(estimator, "source_dir", None),
                    self._get_model_uri(estimator),
                )
                base_name = jumpstart_base_name or base_name
            self._current_job_name = name_from_base(
                base_name, max_length=self.TUNING_JOB_NAME_MAX_LENGTH, short=True
            )

    def _prepare_static_hyperparameters_for_tuning(self, include_cls_metadata=False):
        """Prepare static hyperparameters for all estimators before tuning."""
        self.static_hyperparameters = None
        if self.estimator is not None:
            self.static_hyperparameters = self._prepare_static_hyperparameters(
                self.estimator, self._hyperparameter_ranges, include_cls_metadata
            )

        self.static_hyperparameters_dict = None
        if self.estimator_dict is not None:
            self.static_hyperparameters_dict = {
                estimator_name: self._prepare_static_hyperparameters(
                    estimator,
                    self._hyperparameter_ranges_dict[estimator_name],
                    include_cls_metadata.get(estimator_name, False)
                    if isinstance(include_cls_metadata, dict)
                    else include_cls_metadata,
                )
                for (estimator_name, estimator) in self.estimator_dict.items()
            }

    def _prepare_auto_parameters_for_tuning(self):
        """Prepare auto parameters for all estimators before tuning."""
        self.auto_parameters = None
        if self.estimator is not None:
            self.static_hyperparameters, self.auto_parameters = self._prepare_auto_parameters(
                self.static_hyperparameters, self.hyperparameters_to_keep_static
            )

        self.auto_parameters_dict = None
        if self.estimator_dict is not None:
            static_auto_parameters_dict = {
                estimator_name: self._prepare_auto_parameters(
                    self.static_hyperparameters_dict[estimator_name],
                    self.hyperparameters_to_keep_static_dict.get(estimator_name, None)
                    if self.hyperparameters_to_keep_static_dict
                    else None,
                )
                for estimator_name in sorted(self.estimator_dict.keys())
            }

            self.static_hyperparameters_dict = {}
            self.auto_parameters_dict = {}
            for estimator_name, (
                static_hyperparameters,
                auto_parameters,
            ) in static_auto_parameters_dict.items():
                self.static_hyperparameters_dict[estimator_name] = static_hyperparameters
                self.auto_parameters_dict[estimator_name] = auto_parameters

    @classmethod
    def _prepare_static_hyperparameters(
        cls, estimator, hyperparameter_ranges, include_cls_metadata
    ):
        """Prepare static hyperparameters for one estimator before tuning."""
        # Remove any hyperparameter that will be tuned
        static_hyperparameters = {
            str(k): to_string(v) for (k, v) in estimator.hyperparameters().items()
        }
        if hyperparameter_ranges is not None:
            for hyperparameter_name in hyperparameter_ranges.keys():
                static_hyperparameters.pop(hyperparameter_name, None)

        # For attach() to know what estimator to use for frameworks
        # (other algorithms may not accept extra hyperparameters)
        if include_cls_metadata or isinstance(estimator, Framework):
            static_hyperparameters[cls.SAGEMAKER_ESTIMATOR_CLASS_NAME] = json.dumps(
                estimator.__class__.__name__
            )
            static_hyperparameters[cls.SAGEMAKER_ESTIMATOR_MODULE] = json.dumps(
                estimator.__module__
            )

        return static_hyperparameters

    def _prepare_auto_parameters(self, static_hyperparameters, hyperparameters_to_keep_static):
        """Prepare auto parameters for one estimator before tuning."""
        if not self.autotune:
            return static_hyperparameters, None

        if hyperparameters_to_keep_static is None:
            hyperparameters_to_keep_static = {}

        if not set(hyperparameters_to_keep_static).issubset(set(static_hyperparameters.keys())):
            raise ValueError(
                "Names in hyperparameters_to_keep_static must be members of estimator's "
                "hyperparameters."
            )

        new_static_hyperparameters = {
            k: v for k, v in static_hyperparameters.items() if k in hyperparameters_to_keep_static
        }
        auto_parameters = {
            k: v
            for k, v in static_hyperparameters.items()
            if k not in hyperparameters_to_keep_static
        }

        return new_static_hyperparameters, auto_parameters

    @runnable_by_pipeline
    def fit(
        self,
        inputs: Optional[
            Union[
                str,
                Dict,
                List,
                TrainingInput,
                FileSystemInput,
                RecordSet,
                FileSystemRecordSet,
            ]
        ] = None,
        job_name: Optional[str] = None,
        include_cls_metadata: Union[bool, Dict[str, bool]] = False,
        estimator_kwargs: Optional[Dict[str, dict]] = None,
        wait: bool = True,
        **kwargs
    ):
        """Start a hyperparameter tuning job.

        Args:
            inputs: Information about the training data. Please refer to the
                ``fit()`` method of the associated estimator, as this can take
                any of the following forms:

                * (str) - The S3 location where training data is saved.
                * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) -
                    If using multiple channels for training data, you can specify
                    a dict mapping channel names to strings or
                    :func:`~sagemaker.inputs.TrainingInput` objects.
                * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources
                    that can provide additional information about the training dataset.
                    See :func:`sagemaker.inputs.TrainingInput` for full details.
                * (sagemaker.session.FileSystemInput) - channel configuration for
                    a file system data source that can provide additional information as well as
                    the path to the training dataset.
                * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                    Amazon :class:~`Record` objects serialized and stored in S3.
                    For use with an estimator for an Amazon algorithm.
                * (sagemaker.amazon.amazon_estimator.FileSystemRecordSet) -
                    Amazon SageMaker channel configuration for a file system data source for
                    Amazon algorithms.
                * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
                    where each instance is a different channel of training data.
                * (list[sagemaker.amazon.amazon_estimator.FileSystemRecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.FileSystemRecordSet` objects,
                    where each instance is a different channel of training data.

            job_name (str): Tuning job name. If not specified, the tuner
                generates a default job name, based on the training image name
                and current timestamp.
            include_cls_metadata: It can take one of the following two forms.

                * (bool) - Whether or not the hyperparameter tuning job should include information
                    about the estimator class (default: False). This information is passed as a
                    hyperparameter, so if the algorithm you are using cannot handle unknown
                    hyperparameters (e.g. an Amazon SageMaker built-in algorithm that does not
                    have a custom estimator in the Python SDK), then set ``include_cls_metadata``
                    to ``False``.
                * (dict[str, bool]) - This version should be used for tuners created via the
                    factory method create(), to specify the flag for each estimator provided in
                    the estimator_dict argument of the method. The keys would be the same
                    estimator names as in estimator_dict. If one estimator doesn't need the flag
                    set, then no need to include it in the dictionary.

            estimator_kwargs (dict[str, dict]): Dictionary for other arguments needed for
                training. Should be used only for tuners created via the factory method create().
                The keys are the estimator names for the estimator_dict argument of create()
                method. Each value is a dictionary for the other arguments needed for training
                of the corresponding estimator.
            wait (bool): Whether the call should wait until the job completes (default: ``True``).
            **kwargs: Other arguments needed for training. Please refer to the
                ``fit()`` method of the associated estimator to see what other
                arguments are needed.
        """
        if self.estimator is not None:
            self._fit_with_estimator(inputs, job_name, include_cls_metadata, **kwargs)
        else:
            self._fit_with_estimator_dict(inputs, job_name, include_cls_metadata, estimator_kwargs)

        if wait:
            self.latest_tuning_job.wait()

    def _fit_with_estimator(self, inputs, job_name, include_cls_metadata, **kwargs):
        """Start tuning for tuner instances that have the ``estimator`` field set."""
        self._prepare_estimator_for_tuning(self.estimator, inputs, job_name, **kwargs)
        self._prepare_for_tuning(job_name=job_name, include_cls_metadata=include_cls_metadata)
        self.latest_tuning_job = _TuningJob.start_new(self, inputs)

    def _fit_with_estimator_dict(self, inputs, job_name, include_cls_metadata, estimator_kwargs):
        """Start tuning for tuner instances that have the ``estimator_dict`` field set."""
        estimator_names = sorted(self.estimator_dict.keys())
        self._validate_dict_argument(name="inputs", value=inputs, allowed_keys=estimator_names)
        self._validate_dict_argument(
            name="include_cls_metadata",
            value=include_cls_metadata,
            allowed_keys=estimator_names,
        )
        self._validate_dict_argument(
            name="estimator_kwargs",
            value=estimator_kwargs,
            allowed_keys=estimator_names,
        )

        for (estimator_name, estimator) in self.estimator_dict.items():
            ins = inputs.get(estimator_name, None) if inputs is not None else None
            args = estimator_kwargs.get(estimator_name, {}) if estimator_kwargs is not None else {}
            self._prepare_estimator_for_tuning(estimator, ins, job_name, **args)

        inc_cls_metadata = include_cls_metadata if include_cls_metadata is not None else {}
        self._prepare_for_tuning(job_name=job_name, include_cls_metadata=inc_cls_metadata)

        self.latest_tuning_job = _TuningJob.start_new(self, inputs)

    @classmethod
    def _prepare_estimator_for_tuning(cls, estimator, inputs, job_name, **kwargs):
        """Prepare one estimator before starting tuning."""
        if isinstance(inputs, (list, RecordSet, FileSystemRecordSet)):
            estimator._prepare_for_training(inputs, **kwargs)
        else:
            estimator._prepare_for_training(job_name)

    @classmethod
    def attach(
        cls,
        tuning_job_name,
        sagemaker_session=None,
        job_details=None,
        estimator_cls=None,
    ):
        """Attach to an existing hyperparameter tuning job.

        Create a HyperparameterTuner bound to an existing hyperparameter
        tuning job. After attaching, if there exists a best training job (or any
        other completed training job), that can be deployed to create an Amazon
        SageMaker Endpoint and return a ``Predictor``.

        The ``HyperparameterTuner`` instance could be created in one of the following two forms.

            * If the 'TrainingJobDefinition' field is present in tuning job description, the tuner
                will be created using the default constructor with a single estimator.
            * If the 'TrainingJobDefinitions' field (list) is present in tuning job description,
                the tuner will be created using the factory method ``create()`` with one or
                several estimators. Each estimator corresponds to one item in the
                'TrainingJobDefinitions' field, while the estimator names would come from the
                'DefinitionName' field of items in the 'TrainingJobDefinitions' field. For more
                details on how tuners are created from multiple estimators, see ``create()``
                documentation.

        For more details on 'TrainingJobDefinition' and 'TrainingJobDefinitions' fields in tuning
        job description, see
        https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job

        Args:
            tuning_job_name (str): The name of the hyperparameter tuning job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which manages
                interactions with Amazon SageMaker APIs and any other AWS services needed.
                If not specified, one is created using the default AWS configuration chain.
            job_details (dict): The response to a ``DescribeHyperParameterTuningJob`` call.
                If not specified, the ``HyperparameterTuner`` will perform one such call with
                the provided hyperparameter tuning job name.
            estimator_cls: It can take one of the following two forms.

                (str): The estimator class name associated with the training jobs, e.g.
                    'sagemaker.estimator.Estimator'. If not specified, the ``HyperparameterTuner``
                    will try to derive the correct estimator class from training job metadata,
                    defaulting to :class:~`sagemaker.estimator.Estimator` if it is unable to
                    determine a more specific class.
                (dict[str, str]): This form should be used only when the 'TrainingJobDefinitions'
                    field (list) is present in tuning job description. In this scenario training
                    jobs could be created from different training job definitions in the
                    'TrainingJobDefinitions' field, each of which would be mapped to a different
                    estimator after the ``attach()`` call. The ``estimator_cls`` should then be a
                    dictionary to specify estimator class names for individual estimators as
                    needed. The keys should be the 'DefinitionName' value of items in
                    'TrainingJobDefinitions', which would be used as estimator names in the
                    resulting tuner instance.

        Examples:
            Example #1 - assuming we have the following tuning job description, which has the
            'TrainingJobDefinition' field present using a SageMaker built-in algorithm (i.e. PCA),
            and ``attach()`` can derive the estimator class from the training image.
            So ``estimator_cls`` would not be needed.

            .. code:: python

                {
                    'BestTrainingJob': 'best_training_job_name',
                    'TrainingJobDefinition': {
                        'AlgorithmSpecification': {
                            'TrainingImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1,
                        },
                    },
                }

            >>> my_tuner.fit()
            >>> job_name = my_tuner.latest_tuning_job.name
            Later on:
            >>> attached_tuner = HyperparameterTuner.attach(job_name)
            >>> attached_tuner.deploy()

            Example #2 - assuming we have the following tuning job description, which has a 2-item
            list for the 'TrainingJobDefinitions' field. In this case 'estimator_cls' is only
            needed for the 2nd item since the 1st item uses a SageMaker built-in algorithm
            (i.e. PCA).

            .. code:: python

                {
                    'BestTrainingJob': 'best_training_job_name',
                    'TrainingJobDefinitions': [
                        {
                            'DefinitionName': 'estimator_pca',
                            'AlgorithmSpecification': {
                                'TrainingImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1,
                            },
                        },
                        {
                            'DefinitionName': 'estimator_byoa',
                            'AlgorithmSpecification': {
                                'TrainingImage': '123456789012.dkr.ecr.us-west-2.amazonaws.com/byoa:latest,
                            },
                        }
                    ]
                }

            >>> my_tuner.fit()
            >>> job_name = my_tuner.latest_tuning_job.name
            Later on:
            >>> attached_tuner = HyperparameterTuner.attach(
            >>>     job_name,
            >>>     estimator_cls={
            >>>         'estimator_byoa': 'org.byoa.Estimator'
            >>>     })
            >>> attached_tuner.deploy()


        Returns:
            sagemaker.tuner.HyperparameterTuner: A ``HyperparameterTuner``
            instance with the attached hyperparameter tuning job.
        """
        sagemaker_session = sagemaker_session or Session()

        if job_details is None:
            job_details = sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name
            )

        if "TrainingJobDefinition" in job_details:
            tuner = cls._attach_with_training_details(sagemaker_session, estimator_cls, job_details)
        else:
            tuner = cls._attach_with_training_details_list(
                sagemaker_session, estimator_cls, job_details
            )

        tuner.latest_tuning_job = _TuningJob(
            sagemaker_session=sagemaker_session, job_name=tuning_job_name
        )
        tuner._current_job_name = tuning_job_name

        return tuner

    @classmethod
    def _attach_with_training_details(cls, sagemaker_session, estimator_cls, job_details):
        """Create a HyperparameterTuner bound to an existing hyperparameter tuning job.

        The tuning job has the ``TrainingJobDefinition`` field set in this case.
        """
        estimator = cls._prepare_estimator(
            estimator_cls=estimator_cls,
            training_details=job_details["TrainingJobDefinition"],
            parameter_ranges=job_details["HyperParameterTuningJobConfig"]["ParameterRanges"],
            sagemaker_session=sagemaker_session,
        )
        init_params = cls._prepare_init_params_from_job_description(job_details)

        return cls(estimator=estimator, **init_params)

    @classmethod
    def _attach_with_training_details_list(cls, sagemaker_session, estimator_cls, job_details):
        """Create a HyperparameterTuner bound to an existing hyperparameter tuning job.

        The tuning job has the ``TrainingJobDefinitions`` field set in this case.
        """
        estimator_names = sorted(
            [
                training_details["DefinitionName"]
                for training_details in job_details["TrainingJobDefinitions"]
            ]
        )
        cls._validate_dict_argument(
            name="estimator_cls", value=estimator_cls, allowed_keys=estimator_names
        )

        estimator_dict = {}
        objective_metric_name_dict = {}
        hyperparameter_ranges_dict = {}
        metric_definitions_dict = {}

        for training_details in job_details["TrainingJobDefinitions"]:
            estimator_name = training_details["DefinitionName"]

            estimator_dict[estimator_name] = cls._prepare_estimator(
                estimator_cls=estimator_cls.get(estimator_name) if estimator_cls else None,
                training_details=training_details,
                parameter_ranges=training_details["HyperParameterRanges"],
                sagemaker_session=sagemaker_session,
            )

            objective_metric_name_dict[estimator_name] = training_details["TuningObjective"][
                "MetricName"
            ]
            hyperparameter_ranges_dict[
                estimator_name
            ] = cls._prepare_parameter_ranges_from_job_description(  # noqa: E501 # pylint: disable=line-too-long
                training_details["HyperParameterRanges"]
            )

            metric_definitions = training_details["AlgorithmSpecification"].get(
                "MetricDefinitions", None
            )
            if metric_definitions is not None:
                metric_definitions_dict[estimator_name] = metric_definitions

        init_params = cls._prepare_init_params_from_job_description(job_details)

        return HyperparameterTuner.create(
            estimator_dict=estimator_dict,
            objective_metric_name_dict=objective_metric_name_dict,
            hyperparameter_ranges_dict=hyperparameter_ranges_dict,
            metric_definitions_dict=metric_definitions_dict,
            **init_params
        )

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        serializer=None,
        deserializer=None,
        accelerator_type=None,
        endpoint_name=None,
        wait=True,
        model_name=None,
        kms_key=None,
        data_capture_config=None,
        **kwargs
    ):
        """Deploy the best trained or user specified model to an Amazon SageMaker endpoint.

        And also return a ``sagemaker.Predictor`` object.

        For more information:
        http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html

        Args:
            initial_instance_count (int): Minimum number of EC2 instances to
                deploy to an endpoint for prediction.
            instance_type (str): Type of EC2 instance to deploy to an endpoint
                for prediction, for example, 'ml.c4.xlarge'.
            serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
                serializer object, used to encode data for an inference endpoint
                (default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``.
            deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
                deserializer object, used to decode data from an inference
                endpoint (default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``.
            accelerator_type (str): Type of Elastic Inference accelerator to
                attach to an endpoint for model loading and inference, for
                example, 'ml.eia1.medium'. If not specified, no Elastic
                Inference accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            endpoint_name (str): Name to use for creating an Amazon SageMaker
                endpoint. If not specified, the name of the training job is
                used.
            wait (bool): Whether the call should wait until the deployment of
                model completes (default: True).
            model_name (str): Name to use for creating an Amazon SageMaker
                model. If not specified, the name of the training job is used.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint.
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.
            **kwargs: Other arguments needed for deployment. Please refer to the
                ``create_model()`` method of the associated estimator to see
                what other arguments are needed.

        Returns:
            sagemaker.predictor.Predictor: A predictor that provides a ``predict()``
                method, which can be used to send requests to the Amazon SageMaker endpoint
                and obtain inferences.
        """
        best_training_job = self._get_best_training_job()
        best_estimator = self.best_estimator(best_training_job)

        return best_estimator.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            serializer=serializer,
            deserializer=deserializer,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name or best_training_job["TrainingJobName"],
            wait=wait,
            model_name=model_name,
            kms_key=kms_key,
            data_capture_config=data_capture_config,
            **kwargs
        )

    def stop_tuning_job(self):
        """Stop latest running hyperparameter tuning job."""
        self._ensure_last_tuning_job()
        self.latest_tuning_job.stop()

    def describe(self):
        """Returns a response from the DescribeHyperParameterTuningJob API call."""
        return self.sagemaker_session.describe_tuning_job(self._current_job_name)

    def wait(self):
        """Wait for latest hyperparameter tuning job to finish."""
        self._ensure_last_tuning_job()
        self.latest_tuning_job.wait()

    def best_estimator(self, best_training_job=None):
        """Return the estimator that has best training job attached.

        The trained model can then be deployed to an Amazon SageMaker endpoint and return a
        ``sagemaker.Predictor`` object.

        Args:
            best_training_job (dict): Dictionary containing "TrainingJobName" and
                "TrainingJobDefinitionName".

                Example:

                .. code:: python

                    {
                        "TrainingJobName": "my_training_job_name",
                        "TrainingJobDefinitionName": "my_training_job_definition_name"
                    }

        Returns:
            sagemaker.estimator.EstimatorBase: The estimator that has the best training job
                attached.

        Raises:
            Exception: If there is no best training job available for the hyperparameter tuning job.
        """
        if best_training_job is None:
            best_training_job = self._get_best_training_job()

        if self.estimator is not None:
            best_estimator = self.estimator
        else:
            best_estimator_name = best_training_job["TrainingJobDefinitionName"]
            best_estimator = self.estimator_dict[best_estimator_name]

        return best_estimator.attach(
            training_job_name=best_training_job["TrainingJobName"],
            sagemaker_session=self.sagemaker_session,
        )

    def best_training_job(self):
        """Return name of the best training job for the latest hyperparameter tuning job.

        Raises:
            Exception: If there is no best training job available for the
                hyperparameter tuning job.
        """
        return self._get_best_training_job()["TrainingJobName"]

    def _get_best_training_job(self):
        """Return the best training job for the latest hyperparameter tuning job.

        Raises:
            Exception: If there is no best training job available for the
                hyperparameter tuning job.
        """
        self._ensure_last_tuning_job()

        tuning_job_describe_result = self.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(  # noqa: E501 # pylint: disable=line-too-long
            HyperParameterTuningJobName=self.latest_tuning_job.name
        )

        try:
            return tuning_job_describe_result["BestTrainingJob"]
        except KeyError:
            raise Exception(
                "Best training job not available for tuning job: {}".format(
                    self.latest_tuning_job.name
                )
            )

    def _ensure_last_tuning_job(self):
        """Placeholder docstring"""
        if self.latest_tuning_job is None:
            raise ValueError("No tuning job available")

    @classmethod
    def _prepare_estimator(
        cls, estimator_cls, training_details, parameter_ranges, sagemaker_session
    ):
        """Attach an estimator from training job details"""
        estimator_cls = cls._prepare_estimator_cls(estimator_cls, training_details)
        return cls._prepare_estimator_from_job_description(
            estimator_cls, training_details, parameter_ranges, sagemaker_session
        )

    @classmethod
    def _prepare_estimator_cls(cls, estimator_cls, training_details):
        # Check for customer-specified estimator first
        """Placeholder docstring"""
        if estimator_cls is not None:
            module, cls_name = estimator_cls.rsplit(".", 1)
            return getattr(importlib.import_module(module), cls_name)

        # Then check for estimator class in hyperparameters
        hyperparameters = training_details["StaticHyperParameters"]
        if (
            cls.SAGEMAKER_ESTIMATOR_CLASS_NAME in hyperparameters
            and cls.SAGEMAKER_ESTIMATOR_MODULE in hyperparameters
        ):
            module = hyperparameters.get(cls.SAGEMAKER_ESTIMATOR_MODULE)
            cls_name = hyperparameters.get(cls.SAGEMAKER_ESTIMATOR_CLASS_NAME)
            return getattr(importlib.import_module(json.loads(module)), json.loads(cls_name))

        # Then try to derive the estimator from the image name for 1P algorithms
        image_uri = training_details["AlgorithmSpecification"]["TrainingImage"]
        algorithm = image_uri[image_uri.find("/") + 1 : image_uri.find(":")]
        if algorithm in AMAZON_ESTIMATOR_CLS_NAMES:
            cls_name = AMAZON_ESTIMATOR_CLS_NAMES[algorithm]
            return getattr(importlib.import_module(AMAZON_ESTIMATOR_MODULE), cls_name)

        # Default to the BYO estimator
        return getattr(
            importlib.import_module(cls.DEFAULT_ESTIMATOR_MODULE),
            cls.DEFAULT_ESTIMATOR_CLS_NAME,
        )

    @classmethod
    def _prepare_estimator_from_job_description(
        cls, estimator_cls, training_details, parameter_ranges, sagemaker_session
    ):
        """Placeholder docstring"""
        # Swap name for static hyperparameters to what an estimator would expect
        training_details["HyperParameters"] = training_details["StaticHyperParameters"]
        del training_details["StaticHyperParameters"]

        # Remove hyperparameter reserved by SageMaker for tuning jobs
        del training_details["HyperParameters"]["_tuning_objective_metric"]

        # Add missing hyperparameters defined in the hyperparameter ranges,
        # as potentially required in the Amazon algorithm estimator's constructor
        if issubclass(estimator_cls, AmazonAlgorithmEstimatorBase):
            additional_hyperparameters = cls._extract_hyperparameters_from_parameter_ranges(
                parameter_ranges
            )
            training_details["HyperParameters"].update(additional_hyperparameters)

        # Add items expected by the estimator (but aren't needed otherwise)
        training_details["TrainingJobName"] = ""
        if "KmsKeyId" not in training_details["OutputDataConfig"]:
            training_details["OutputDataConfig"]["KmsKeyId"] = ""

        estimator_init_params = estimator_cls._prepare_init_params_from_job_description(
            training_details
        )
        return estimator_cls(sagemaker_session=sagemaker_session, **estimator_init_params)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Placeholder docstring"""
        tuning_config = job_details["HyperParameterTuningJobConfig"]

        params = {
            "strategy": tuning_config["Strategy"],
            "max_jobs": tuning_config["ResourceLimits"]["MaxNumberOfTrainingJobs"],
            "max_parallel_jobs": tuning_config["ResourceLimits"]["MaxParallelTrainingJobs"],
            "warm_start_config": WarmStartConfig.from_job_desc(
                job_details.get("WarmStartConfig", None)
            ),
            "early_stopping_type": tuning_config["TrainingJobEarlyStoppingType"],
            "base_tuning_job_name": base_from_name(job_details["HyperParameterTuningJobName"]),
        }

        if "TuningJobCompletionCriteria" in tuning_config:
            params["completion_criteria_config"] = TuningJobCompletionCriteriaConfig.from_job_desc(
                tuning_config["TuningJobCompletionCriteria"]
            )

        if MAX_RUNTIME_IN_SECONDS in tuning_config["ResourceLimits"]:
            params["max_runtime_in_seconds"] = tuning_config["ResourceLimits"][
                MAX_RUNTIME_IN_SECONDS
            ]

        if "RandomSeed" in tuning_config:
            params["random_seed"] = tuning_config["RandomSeed"]

        if "HyperParameterTuningJobObjective" in tuning_config:
            params["objective_metric_name"] = tuning_config["HyperParameterTuningJobObjective"][
                "MetricName"
            ]
            params["objective_type"] = tuning_config["HyperParameterTuningJobObjective"]["Type"]

        if "ParameterRanges" in tuning_config:
            params["hyperparameter_ranges"] = cls._prepare_parameter_ranges_from_job_description(
                tuning_config["ParameterRanges"]
            )

        if "TrainingJobDefinition" in job_details:
            params["metric_definitions"] = job_details["TrainingJobDefinition"][
                "AlgorithmSpecification"
            ]["MetricDefinitions"]

        if "TrainingJobDefinitions" in job_details:
            params["objective_type"] = job_details["TrainingJobDefinitions"][0]["TuningObjective"][
                "Type"
            ]

        return params

    @classmethod
    def _prepare_parameter_ranges_from_job_description(cls, parameter_ranges):
        """Placeholder docstring"""
        ranges = {}

        for parameter in parameter_ranges["CategoricalParameterRanges"]:
            ranges[parameter["Name"]] = CategoricalParameter(parameter["Values"])

        for parameter in parameter_ranges["ContinuousParameterRanges"]:
            ranges[parameter["Name"]] = ContinuousParameter(
                float(parameter["MinValue"]), float(parameter["MaxValue"])
            )

        for parameter in parameter_ranges["IntegerParameterRanges"]:
            ranges[parameter["Name"]] = IntegerParameter(
                int(parameter["MinValue"]), int(parameter["MaxValue"])
            )

        return ranges

    @classmethod
    def _extract_hyperparameters_from_parameter_ranges(cls, parameter_ranges):
        """Placeholder docstring"""
        hyperparameters = {}

        for parameter in parameter_ranges["CategoricalParameterRanges"]:
            hyperparameters[parameter["Name"]] = parameter["Values"][0]

        for parameter in parameter_ranges["ContinuousParameterRanges"]:
            hyperparameters[parameter["Name"]] = float(parameter["MinValue"])

        for parameter in parameter_ranges["IntegerParameterRanges"]:
            hyperparameters[parameter["Name"]] = int(parameter["MinValue"])

        return hyperparameters

    def hyperparameter_ranges(self):
        """Return the hyperparameter ranges in a dictionary.

        Dictionary to be used as part of a request for creating a hyperparameter tuning job.
        """
        if self._hyperparameter_ranges is None:
            return None

        return self._prepare_parameter_ranges_for_tuning(
            self._hyperparameter_ranges, self.estimator
        )

    def hyperparameter_ranges_dict(self):
        """Return a dictionary of hyperparameter ranges for all estimators in ``estimator_dict``"""
        if self._hyperparameter_ranges_dict is None:
            return None

        return {
            estimator_name: self._prepare_parameter_ranges_for_tuning(
                self._hyperparameter_ranges_dict[estimator_name],
                self.estimator_dict[estimator_name],
            )
            for estimator_name in sorted(self.estimator_dict.keys())
        }

    @classmethod
    def _prepare_parameter_ranges_for_tuning(cls, parameter_ranges, estimator):
        """Prepare hyperparameter ranges for tuning"""
        processed_parameter_ranges = dict()
        for range_type in ParameterRange.__all_types__:
            hp_ranges = []
            for parameter_name, parameter in parameter_ranges.items():
                if parameter is not None and parameter.__name__ == range_type:
                    # Categorical parameters needed to be serialized as JSON for our framework
                    # containers
                    if isinstance(parameter, CategoricalParameter) and isinstance(
                        estimator, Framework
                    ):
                        tuning_range = parameter.as_json_range(parameter_name)
                    else:
                        tuning_range = parameter.as_tuning_range(parameter_name)
                    hp_ranges.append(tuning_range)
            processed_parameter_ranges[range_type + "ParameterRanges"] = hp_ranges
        return processed_parameter_ranges

    @property
    def sagemaker_session(self):
        """Convenience method for accessing the SageMaker session.

        It access :class:`~sagemaker.session.Session` object associated with the estimator
        for the ``HyperparameterTuner``.
        """
        estimator = self.estimator
        if estimator is None:
            first_estimator_name = sorted(self.estimator_dict.keys())[0]
            estimator = self.estimator_dict[first_estimator_name]
        return estimator.sagemaker_session

    def analytics(self):
        """An instance of HyperparameterTuningJobAnalytics for this latest tuning job of this tuner.

        Analytics olbject gives you access to tuning results summarized into a pandas dataframe.
        """
        return HyperparameterTuningJobAnalytics(self.latest_tuning_job.name, self.sagemaker_session)

    def _validate_parameter_ranges(self, estimator, hyperparameter_ranges):
        """Validate hyperparameter ranges for an estimator"""
        for kls in inspect.getmro(estimator.__class__)[::-1]:
            for _, value in kls.__dict__.items():
                if isinstance(value, hp):
                    try:
                        # The hyperparam names may not be the same as the class attribute that
                        # holds them, for instance: local_lloyd_init_method is called
                        # local_init_method. We need to map these and pass the correct name to
                        # the constructor.
                        parameter_range = hyperparameter_ranges[value.name]

                        if isinstance(parameter_range, ParameterRange):
                            self._validate_parameter_range(value, parameter_range)
                    except KeyError:
                        pass

    def _validate_parameter_range(self, value_hp, parameter_range):
        """Placeholder docstring"""
        for (
            parameter_range_key,
            parameter_range_value,
        ) in parameter_range.__dict__.items():
            if parameter_range_key == "scaling_type":
                continue

            # Categorical ranges
            if isinstance(parameter_range_value, list):
                for categorical_value in parameter_range_value:
                    value_hp.validate(categorical_value)
            # Continuous, Integer ranges
            else:
                value_hp.validate(parameter_range_value)

    def transfer_learning_tuner(self, additional_parents=None, estimator=None):
        """Creates a new ``HyperparameterTuner``.

        Creation is done by copying the request fields from the provided parent
        to the new instance of ``HyperparameterTuner``.
        Followed by addition of warm start configuration with the type as
        "TransferLearning" and parents as the union of provided list of
        ``additional_parents`` and the ``self``. Also, training image in the new
        tuner's estimator is updated with the provided ``training_image``.

        Examples:
            >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
            >>> transfer_learning_tuner = parent_tuner.transfer_learning_tuner(
            >>>                                             additional_parents={"parent-job-2"})
            Later On:
            >>> transfer_learning_tuner.fit(inputs={})

        Args:
            additional_parents (set{str}): Set of additional parents along with
                the self to be used in warm starting
            estimator (sagemaker.estimator.EstimatorBase): An estimator object
                that has been initialized with the desired configuration. There
                does not need to be a training job associated with this
                instance.

        Returns:
            sagemaker.tuner.HyperparameterTuner: ``HyperparameterTuner``
            instance which can be used to launch transfer learning tuning job.
        """

        return self._create_warm_start_tuner(
            additional_parents=additional_parents,
            warm_start_type=WarmStartTypes.TRANSFER_LEARNING,
            estimator=estimator,
        )

    def identical_dataset_and_algorithm_tuner(self, additional_parents=None):
        """Creates a new ``HyperparameterTuner``.

        Creation is done by copying the request fields from the provided parent to the new instance
        of ``HyperparameterTuner``.

        Followed by addition of warm start configuration with the type as
        "IdenticalDataAndAlgorithm" and parents as the union of provided list of
        ``additional_parents`` and the ``self``

        Examples:
            >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
            >>> identical_dataset_algo_tuner = parent_tuner.identical_dataset_and_algorithm_tuner(
            >>>                                                additional_parents={"parent-job-2"})
            Later On:
            >>> identical_dataset_algo_tuner.fit(inputs={})

        Args:
            additional_parents (set{str}): Set of additional parents along with
                the self to be used in warm starting

        Returns:
            sagemaker.tuner.HyperparameterTuner: HyperparameterTuner instance
            which can be used to launch identical dataset and algorithm tuning
            job.
        """

        return self._create_warm_start_tuner(
            additional_parents=additional_parents,
            warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
        )

    def _create_warm_start_tuner(self, additional_parents, warm_start_type, estimator=None):
        """Creates a new ``HyperparameterTuner`` with ``WarmStartConfig``.

        Where type will be equal to ``warm_start_type`` and``parents`` would be equal
        to union of ``additional_parents`` and self.

        Args:
            additional_parents (set{str}): Additional parents along with self,
                to be used for warm starting.
            warm_start_type (sagemaker.tuner.WarmStartTypes): Type of warm start
                job.
            estimator:

        Returns:
            sagemaker.tuner.HyperparameterTuner: Instance with the request
            fields copied from self along with the warm start configuration
        """
        all_parents = {self.latest_tuning_job.name}
        if additional_parents:
            all_parents = all_parents.union(additional_parents)

        if self.estimator is not None:
            return HyperparameterTuner(
                estimator=estimator if estimator else self.estimator,
                objective_metric_name=self.objective_metric_name,
                hyperparameter_ranges=self._hyperparameter_ranges,
                strategy=self.strategy,
                strategy_config=self.strategy_config,
                completion_criteria_config=self.completion_criteria_config,
                objective_type=self.objective_type,
                max_jobs=self.max_jobs,
                max_parallel_jobs=self.max_parallel_jobs,
                max_runtime_in_seconds=self.max_runtime_in_seconds,
                warm_start_config=WarmStartConfig(
                    warm_start_type=warm_start_type, parents=all_parents
                ),
                early_stopping_type=self.early_stopping_type,
                random_seed=self.random_seed,
            )

        if len(self.estimator_dict) > 1:
            raise ValueError(
                "Warm start is not supported currently for tuners with multiple estimators"
            )

        if estimator is not None:
            estimator_name = list(self.estimator_dict.keys())[0]
            estimator_dict = {estimator_name: estimator}
        else:
            estimator_dict = self.estimator_dict

        return HyperparameterTuner.create(
            estimator_dict=estimator_dict,
            objective_metric_name_dict=self.objective_metric_name_dict,
            hyperparameter_ranges_dict=self._hyperparameter_ranges_dict,
            metric_definitions_dict=self.metric_definitions_dict,
            strategy=self.strategy,
            strategy_config=self.strategy_config,
            completion_criteria_config=self.completion_criteria_config,
            objective_type=self.objective_type,
            max_jobs=self.max_jobs,
            max_parallel_jobs=self.max_parallel_jobs,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            warm_start_config=WarmStartConfig(warm_start_type=warm_start_type, parents=all_parents),
            early_stopping_type=self.early_stopping_type,
            random_seed=self.random_seed,
        )

    @classmethod
    def create(
        cls,
        estimator_dict,
        objective_metric_name_dict,
        hyperparameter_ranges_dict,
        metric_definitions_dict=None,
        base_tuning_job_name=None,
        strategy="Bayesian",
        strategy_config=None,
        completion_criteria_config=None,
        objective_type="Maximize",
        max_jobs=None,
        max_parallel_jobs=1,
        max_runtime_in_seconds=None,
        tags=None,
        warm_start_config=None,
        early_stopping_type="Off",
        random_seed=None,
        autotune=False,
        hyperparameters_to_keep_static_dict=None,
    ):
        """Factory method to create a ``HyperparameterTuner`` instance.

        It takes one or more estimators to obtain configuration information for training jobs
        that are created as the result of a hyperparameter tuning job. The estimators are provided
        through a  dictionary (i.e. ``estimator_dict``) with unique estimator names as the keys.
        For  individual estimators separate objective metric names and hyperparameter ranges
        should be provided in two dictionaries, i.e. ``objective_metric_name_dict`` and
        ``hyperparameter_ranges_dict``, with the same estimator names as the keys. Optional
        metrics definitions could also be provided for individual estimators via another dictionary
        ``metric_definitions_dict``.

        Args:
            estimator_dict (dict[str, sagemaker.estimator.EstimatorBase]): Dictionary of estimator
                instances that have been initialized with the desired configuration. There does not
                need to be a training job associated with the estimator instances. The keys of the
                dictionary would be referred to as "estimator names".
            objective_metric_name_dict (dict[str, str]): Dictionary of names of the objective
                metric for evaluating training jobs. The keys are the same set of estimator names
                as in ``estimator_dict``, and there must be one entry for each estimator in
                ``estimator_dict``.
            hyperparameter_ranges_dict (dict[str, dict[str, sagemaker.parameter.ParameterRange]]):
                Dictionary of tunable hyperparameter ranges. The keys are the same set of estimator
                names as in estimator_dict, and there must be one entry for each estimator in
                estimator_dict. Each value is a dictionary of sagemaker.parameter.ParameterRange
                instance, which can be one of three types: Continuous, Integer, or Categorical.
                The keys of each ParameterRange dictionaries are the names of the hyperparameter,
                and the values are the appropriate parameter range class to represent the range.
            metric_definitions_dict (dict(str, list[dict]]): Dictionary of metric definitions.
                The keys are the same set or a subset of estimator names as in estimator_dict,
                and there must be one entry for each estimator in estimator_dict. Each value is
                a list of dictionaries that defines the metric(s) used to evaluate the training
                jobs (default: None). Each of these dictionaries contains two keys: 'Name' for the
                name of the metric, and 'Regex' for the regular expression used to extract the
                metric from the logs. This should be defined only for hyperparameter tuning jobs
                that don't use an Amazon algorithm.
            base_tuning_job_name (str): Prefix for the hyperparameter tuning job name when the
                :meth:`~sagemaker.tuner.HyperparameterTuner.fit` method launches.
                If not specified, a default job name is generated,
                based on the training image name and current timestamp.
            strategy (str): Strategy to be used for hyperparameter estimations
                (default: 'Bayesian').
            strategy_config (dict): The configuration for a training job launched by a
                hyperparameter tuning job.
            completion_criteria_config (dict): The configuration for tuning job completion criteria.
            objective_type (str): The type of the objective metric for evaluating training jobs.
                This value can be either 'Minimize' or 'Maximize' (default: 'Maximize').
            max_jobs (int): Maximum total number of training jobs to start for the hyperparameter
                tuning job. The default value is unspecified fot the 'Grid' strategy
                and the value is 1 for all others strategies (default: None).
            max_parallel_jobs (int): Maximum number of parallel training jobs to start
                (default: 1).
            max_runtime_in_seconds (int): The maximum time in seconds
                 that a hyperparameter tuning job can run.
            tags (list[dict]): List of tags for labeling the tuning job (default: None). For more,
                see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            warm_start_config (sagemaker.tuner.WarmStartConfig): A ``WarmStartConfig`` object that
                has been initialized with the configuration defining the nature of warm start
                tuning job.
            early_stopping_type (str): Specifies whether early stopping is enabled for the job.
                Can be either 'Auto' or 'Off' (default: 'Off'). If set to 'Off', early stopping
                will not be attempted. If set to 'Auto', early stopping of some training jobs may
                happen, but is not guaranteed to.
            random_seed (int): An initial value used to initialize a pseudo-random number generator.
                Setting a random seed will make the hyperparameter tuning search strategies to
                produce more consistent configurations for the same tuning job.
            autotune (bool): Whether the parameter ranges or other unset settings of a tuning job
                should be chosen automatically (default: False).
            hyperparameters_to_keep_static_dict (dict(str, list[str]]): Dictionary of
                hyperparameter names that will be kept static. The keys are the same set or a subset
                of estimator names as in estimator_dict, and there must be one entry for each
                estimator in estimator_dict. Each value is a list of hyperparameter names that will
                be kept static and will not be assigned a tunable range with Autotune functionality
                (default: None).

        Returns:
            sagemaker.tuner.HyperparameterTuner: a new ``HyperparameterTuner`` object that can
            start a hyperparameter tuning job with one or more estimators.

        """

        cls._validate_create_tuner_inputs(
            estimator_dict,
            objective_metric_name_dict,
            hyperparameter_ranges_dict,
            metric_definitions_dict,
            hyperparameters_to_keep_static_dict,
        )

        estimator_names = sorted(estimator_dict.keys())
        first_estimator_name = estimator_names[0]

        metric_definitions = (
            metric_definitions_dict.get(first_estimator_name, None)
            if metric_definitions_dict is not None
            else None
        )

        hyperparameters_to_keep_static = (
            hyperparameters_to_keep_static_dict.get(first_estimator_name, None)
            if hyperparameters_to_keep_static_dict is not None
            else None
        )

        tuner = HyperparameterTuner(
            base_tuning_job_name=base_tuning_job_name,
            estimator_name=first_estimator_name,
            estimator=estimator_dict[first_estimator_name],
            objective_metric_name=objective_metric_name_dict[first_estimator_name],
            hyperparameter_ranges=hyperparameter_ranges_dict[first_estimator_name],
            metric_definitions=metric_definitions,
            strategy=strategy,
            strategy_config=strategy_config,
            completion_criteria_config=completion_criteria_config,
            objective_type=objective_type,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            max_runtime_in_seconds=max_runtime_in_seconds,
            tags=tags,
            warm_start_config=warm_start_config,
            early_stopping_type=early_stopping_type,
            random_seed=random_seed,
            autotune=autotune,
            hyperparameters_to_keep_static=hyperparameters_to_keep_static,
        )

        for estimator_name in estimator_names[1:]:
            metric_definitions = (
                metric_definitions_dict.get(estimator_name, None)
                if metric_definitions_dict is not None
                else None
            )
            hyperparameters_to_keep_static = (
                hyperparameters_to_keep_static_dict.get(estimator_name, None)
                if hyperparameters_to_keep_static_dict is not None
                else None
            )
            tuner._add_estimator(
                estimator_name=estimator_name,
                estimator=estimator_dict[estimator_name],
                objective_metric_name=objective_metric_name_dict[estimator_name],
                hyperparameter_ranges=hyperparameter_ranges_dict[estimator_name],
                metric_definitions=metric_definitions,
                hyperparameters_to_keep_static=hyperparameters_to_keep_static,
            )
        return tuner

    @classmethod
    def _validate_create_tuner_inputs(
        cls,
        estimator_dict,
        objective_metric_name_dict,
        hyperparameter_ranges_dict,
        metric_definitions_dict=None,
        hyperparameters_to_keep_static_dict=None,
    ):
        """Validate inputs for ``HyperparameterTuner.create()``"""
        cls._validate_estimator_dict(estimator_dict)

        estimator_names = sorted(estimator_dict.keys())

        cls._validate_dict_argument(
            name="objective_metric_name_dict",
            value=objective_metric_name_dict,
            allowed_keys=estimator_names,
            require_same_keys=True,
        )
        cls._validate_dict_argument(
            name="hyperparameter_ranges_dict",
            value=hyperparameter_ranges_dict,
            allowed_keys=estimator_names,
            require_same_keys=True,
        )
        cls._validate_dict_argument(
            name="metric_definitions_dict",
            value=metric_definitions_dict,
            allowed_keys=estimator_names,
        )
        cls._validate_dict_argument(
            name="hyperparameters_to_keep_static_dict",
            value=hyperparameters_to_keep_static_dict,
            allowed_keys=estimator_names,
        )

    @classmethod
    def _validate_estimator_dict(cls, estimator_dict):
        """Validate ``estimator_dict`` in inputs for ``HyperparameterTuner.create()``"""
        if estimator_dict is None or len(estimator_dict) == 0:
            raise ValueError("At least one estimator should be provided")
        if None in estimator_dict.keys():
            raise ValueError("Estimator names cannot be None")

    @classmethod
    def _validate_dict_argument(cls, name, value, allowed_keys, require_same_keys=False):
        """Check if an argument is an dictionary with correct key set."""
        if value is None:
            return

        if not isinstance(value, dict):
            raise ValueError(
                "Argument '{}' must be a dictionary using {} as keys".format(name, allowed_keys)
            )

        value_keys = sorted(value.keys())

        if require_same_keys:
            if value_keys != allowed_keys:
                raise ValueError(
                    "The keys of argument '{}' must be the same as {}".format(name, allowed_keys)
                )
        else:
            if not set(value_keys).issubset(set(allowed_keys)):
                raise ValueError(
                    "The keys of argument '{}' must be a subset of {}".format(name, allowed_keys)
                )

    def _add_estimator(
        self,
        estimator_name,
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions=None,
        hyperparameters_to_keep_static=None,
    ):
        """Add an estimator with corresponding attributes, if applicable.

        The objective metric name, parameter ranges and metric definitions are added to
        the estimator, if populated.
        """
        self.estimator_dict[estimator_name] = estimator
        self.objective_metric_name_dict[estimator_name] = objective_metric_name
        self._hyperparameter_ranges_dict[estimator_name] = hyperparameter_ranges
        if hyperparameters_to_keep_static is not None:
            self.hyperparameters_to_keep_static_dict[
                estimator_name
            ] = hyperparameters_to_keep_static
        if metric_definitions is not None:
            self.metric_definitions_dict[estimator_name] = metric_definitions

    delete_endpoint = removed_function("delete_endpoint")


class _TuningJob(_Job):
    """Placeholder docstring"""

    @classmethod
    def start_new(cls, tuner, inputs):
        """Create a new Amazon SageMaker HyperParameter Tuning job.

        The new HyperParameter Tuning job uses the provided `tuner` and `inputs`
        to start a new job.

        Args:
            tuner (sagemaker.tuner.HyperparameterTuner): HyperparameterTuner
                object created by the user.
            inputs (str): Parameters used when called
                :meth:`~sagemaker.estimator.EstimatorBase.fit`.

        Returns:
            sagemaker.tuner._TuningJob: Constructed object that captures all
            information about the started job.
        """
        tuner_args = cls._get_tuner_args(tuner, inputs)

        tuner.sagemaker_session.create_tuning_job(**tuner_args)

        return cls(tuner.sagemaker_session, tuner._current_job_name)

    @classmethod
    def _get_tuner_args(cls, tuner, inputs):
        """Gets a dict of arguments for a new Amazon SageMaker tuning job from the tuner

        Args:
            tuner (:class:`~sagemaker.tuner.HyperparameterTuner`):
                The ``HyperparameterTuner`` instance that started the job.
            inputs: Information about the training data. Please refer to the
            ``fit()`` method of the associated estimator.
        Returns:
            Dict: dict for `sagemaker.session.Session.tune` method
        """
        warm_start_config_req = None
        if tuner.warm_start_config:
            warm_start_config_req = tuner.warm_start_config.to_input_req()

        tuning_config = {
            "strategy": tuner.strategy,
            "max_jobs": tuner.max_jobs,
            "max_parallel_jobs": tuner.max_parallel_jobs,
            "early_stopping_type": tuner.early_stopping_type,
        }

        if tuner.max_runtime_in_seconds is not None:
            tuning_config["max_runtime_in_seconds"] = tuner.max_runtime_in_seconds

        if tuner.random_seed is not None:
            tuning_config["random_seed"] = tuner.random_seed

        if tuner.strategy_config is not None:
            tuning_config["strategy_config"] = tuner.strategy_config.to_input_req()

        if tuner.objective_metric_name is not None:
            tuning_config["objective_type"] = tuner.objective_type
            tuning_config["objective_metric_name"] = tuner.objective_metric_name

        parameter_ranges = tuner.hyperparameter_ranges()
        if parameter_ranges is not None:
            tuning_config["parameter_ranges"] = parameter_ranges

        if tuner.auto_parameters is not None:
            tuning_config["auto_parameters"] = tuner.auto_parameters

        if tuner.completion_criteria_config is not None:
            tuning_config[
                "completion_criteria_config"
            ] = tuner.completion_criteria_config.to_input_req()

        tuner_args = {
            "job_name": tuner._current_job_name,
            "tuning_config": tuning_config,
            "tags": tuner.tags,
            "warm_start_config": warm_start_config_req,
            "autotune": tuner.autotune,
        }

        if tuner.estimator is not None:
            tuner_args["training_config"] = cls._prepare_training_config(
                inputs=inputs,
                estimator=tuner.estimator,
                static_hyperparameters=tuner.static_hyperparameters,
                metric_definitions=tuner.metric_definitions,
                instance_configs=tuner.instance_configs,
            )

        if tuner.estimator_dict is not None:
            tuner_args["training_config_list"] = [
                cls._prepare_training_config(
                    inputs.get(estimator_name, None) if inputs is not None else None,
                    tuner.estimator_dict[estimator_name],
                    tuner.static_hyperparameters_dict[estimator_name],
                    tuner.metric_definitions_dict.get(estimator_name, None),
                    estimator_name,
                    tuner.objective_type,
                    tuner.objective_metric_name_dict[estimator_name],
                    tuner.hyperparameter_ranges_dict()[estimator_name],
                    tuner.instance_configs_dict.get(estimator_name, None)
                    if tuner.instance_configs_dict is not None
                    else None,
                    tuner.auto_parameters_dict.get(estimator_name, None)
                    if tuner.auto_parameters_dict is not None
                    else None,
                )
                for estimator_name in sorted(tuner.estimator_dict.keys())
            ]

        return tuner_args

    @staticmethod
    def _prepare_hp_resource_config(
        instance_configs: List[InstanceConfig],
        instance_count: int,
        instance_type: str,
        volume_size: int,
        volume_kms_key: str,
    ):
        """Placeholder hpo resource config for one estimator of the tuner."""
        resource_config = {}
        if volume_kms_key is not None:
            resource_config["VolumeKmsKeyId"] = volume_kms_key
        if instance_configs is None:
            resource_config["InstanceCount"] = instance_count
            resource_config["InstanceType"] = instance_type
            resource_config["VolumeSizeInGB"] = volume_size
        else:
            resource_config["InstanceConfigs"] = _TuningJob._prepare_instance_configs(
                instance_configs
            )
        return resource_config

    @staticmethod
    def _prepare_instance_configs(instance_configs: List[InstanceConfig]):
        """Prepare instance config for create tuning request."""
        return [config.to_input_req() for config in instance_configs]

    @staticmethod
    def _prepare_training_config(
        inputs,
        estimator,
        static_hyperparameters,
        metric_definitions,
        estimator_name=None,
        objective_type=None,
        objective_metric_name=None,
        parameter_ranges=None,
        instance_configs=None,
        auto_parameters=None,
    ):
        """Prepare training config for one estimator."""
        training_config = _Job._load_config(inputs, estimator)

        del training_config["resource_config"]
        training_config["hpo_resource_config"] = _TuningJob._prepare_hp_resource_config(
            instance_configs,
            estimator.instance_count,
            estimator.instance_type,
            estimator.volume_size,
            estimator.volume_kms_key,
        )

        training_config["input_mode"] = estimator.input_mode
        training_config["metric_definitions"] = metric_definitions

        if isinstance(inputs, TrainingInput):
            if "InputMode" in inputs.config:
                logger.debug(
                    "Selecting TrainingInput's input_mode (%s) for TrainingInputMode.",
                    inputs.config["InputMode"],
                )
                training_config["input_mode"] = inputs.config["InputMode"]

        if isinstance(estimator, sagemaker.algorithm.AlgorithmEstimator):
            training_config["algorithm_arn"] = estimator.algorithm_arn
        else:
            training_config["image_uri"] = estimator.training_image_uri()

        training_config["enable_network_isolation"] = estimator.enable_network_isolation()
        training_config[
            "encrypt_inter_container_traffic"
        ] = estimator.encrypt_inter_container_traffic

        training_config["use_spot_instances"] = estimator.use_spot_instances
        training_config["checkpoint_s3_uri"] = estimator.checkpoint_s3_uri
        training_config["checkpoint_local_path"] = estimator.checkpoint_local_path

        training_config["static_hyperparameters"] = static_hyperparameters

        if estimator_name is not None:
            training_config["estimator_name"] = estimator_name

        if objective_type is not None:
            training_config["objective_type"] = objective_type

        if objective_metric_name is not None:
            training_config["objective_metric_name"] = objective_metric_name

        if parameter_ranges is not None:
            training_config["parameter_ranges"] = parameter_ranges

        if auto_parameters is not None:
            training_config["auto_parameters"] = auto_parameters

        if estimator.max_retry_attempts is not None:
            training_config["max_retry_attempts"] = estimator.max_retry_attempts

        if estimator.environment is not None:
            training_config["environment"] = estimator.environment

        return training_config

    def stop(self):
        """Placeholder docstring."""
        self.sagemaker_session.stop_tuning_job(name=self.name)

    def wait(self):
        """Placeholder docstring."""
        self.sagemaker_session.wait_for_tuning_job(self.name)


def create_identical_dataset_and_algorithm_tuner(
    parent, additional_parents=None, sagemaker_session=None
):
    """Creates a new tuner with an identical dataset and algorithm.

    It does this identical creation by copying the request fields from the
    provided parent to the new instance of ``HyperparameterTuner`` followed
    by addition of warm start configuration with the type as
    "IdenticalDataAndAlgorithm" and ``parents`` as the union of provided list
    of ``additional_parents`` and the ``parent``.

    Args:
        parent (str): Primary parent tuning job's name from which the Tuner and
            Estimator configuration has to be copied
        additional_parents (set{str}): Set of additional parent tuning job's
            names along with the primary parent tuning job name to be used in
            warm starting the transfer learning tuner.
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other AWS
            services needed. If not specified, one is created using the default
            AWS configuration chain.

    Returns:
        sagemaker.tuner.HyperparameterTuner: a new ``HyperparameterTuner``
        object for the warm-started hyperparameter tuning job
    """

    parent_tuner = HyperparameterTuner.attach(
        tuning_job_name=parent, sagemaker_session=sagemaker_session
    )
    return parent_tuner.identical_dataset_and_algorithm_tuner(additional_parents=additional_parents)


def create_transfer_learning_tuner(
    parent, additional_parents=None, estimator=None, sagemaker_session=None
):
    """Creates a new ``HyperParameterTuner`` instance from the parent.

    It creates the new tuner by copying the request fields from the provided
    parent to the new instance of ``HyperparameterTuner`` followed by addition
    of warm start configuration with the type as "TransferLearning" and
    ``parents`` as the union of provided list of ``additional_parents`` and
    the ``parent``.

    Args:
        parent (str): Primary parent tuning job's name from which the Tuner and
            Estimator configuration has to be copied
        additional_parents (set{str}): Set of additional parent tuning job's
            names along with the primary parent tuning job name to be used in
            warm starting the identical dataset and algorithm tuner.
        estimator (sagemaker.estimator.EstimatorBase): An estimator object that
            has been initialized with the desired configuration. There does not
            need to be a training job associated with this instance.
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other AWS
            services needed. If not specified, one is created using the default
            AWS configuration chain.

    Returns:
        sagemaker.tuner.HyperparameterTuner: New instance of warm started
        HyperparameterTuner
    """

    parent_tuner = HyperparameterTuner.attach(
        tuning_job_name=parent, sagemaker_session=sagemaker_session
    )
    return parent_tuner.transfer_learning_tuner(
        additional_parents=additional_parents, estimator=estimator
    )
