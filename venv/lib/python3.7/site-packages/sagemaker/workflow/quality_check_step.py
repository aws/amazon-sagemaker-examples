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
"""The step definitions for workflow."""
from __future__ import absolute_import

from abc import ABC
from typing import List, Union, Optional
import os
import pathlib
import attr

from sagemaker import s3
from sagemaker.model_monitor import ModelMonitor
from sagemaker.processing import ProcessingOutput, ProcessingJob, Processor, ProcessingInput
from sagemaker.workflow import is_pipeline_variable

from sagemaker.workflow.entities import RequestType, PipelineVariable
from sagemaker.workflow.properties import (
    Properties,
)
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.steps import Step, StepTypeEnum, CacheConfig
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.utilities import trim_request_dict

_CONTAINER_BASE_PATH = "/opt/ml/processing"
_CONTAINER_INPUT_PATH = "input"
_CONTAINER_OUTPUT_PATH = "output"
_BASELINE_DATASET_INPUT_NAME = "baseline_dataset_input"
_RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME = "record_preprocessor_script_input"
_POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME = "post_analytics_processor_script_input"
_MODEL_MONITOR_S3_PATH = "model-monitor"
_BASELINING_S3_PATH = "baselining"
_RESULTS_S3_PATH = "results"
_DEFAULT_OUTPUT_NAME = "quality_check_output"
_MODEL_QUALITY_TYPE = "MODEL_QUALITY"
_DATA_QUALITY_TYPE = "DATA_QUALITY"


@attr.s
class QualityCheckConfig(ABC):
    """Quality Check Config.

    Attributes:
        baseline_dataset (str or PipelineVariable): The path to the
            baseline_dataset file. This can be a local path or an S3 uri string
        dataset_format (dict): The format of the baseline_dataset.
        output_s3_uri (str or PipelineVariable): Desired S3 destination of
            the constraint_violations and statistics json files (default: None).
            If not specified an auto generated path will be used:
            "s3://<default_session_bucket>/model-monitor/baselining/<job_name>/results"
        post_analytics_processor_script (str): The path to the record post-analytics
            processor script (default: None). This can be a local path or an S3 uri string
            but CANNOT be any type of the PipelineVariable.
    """

    baseline_dataset: Union[str, PipelineVariable] = attr.ib()
    dataset_format: dict = attr.ib()
    output_s3_uri: Union[str, PipelineVariable] = attr.ib(kw_only=True, default=None)
    post_analytics_processor_script: str = attr.ib(kw_only=True, default=None)


@attr.s
class DataQualityCheckConfig(QualityCheckConfig):
    """Data Quality Check Config.

    Attributes:
        record_preprocessor_script (str): The path to the record preprocessor script
            (default: None).
            This can be a local path or an S3 uri string
            but CANNOT be any type of the PipelineVariable.
    """

    record_preprocessor_script: str = attr.ib(default=None)


@attr.s
class ModelQualityCheckConfig(QualityCheckConfig):
    """Model Quality Check Config.

    Attributes:
        problem_type (str or PipelineVariable): The type of problem of this model
            quality monitoring.
            Valid values are "Regression", "BinaryClassification", "MulticlassClassification".
        inference_attribute (str or PipelineVariable): Index or JSONpath to
            locate predicted label(s) (default: None).
        probability_attribute (str or PipelineVariable): Index or JSONpath to
            locate probabilities (default: None).
        ground_truth_attribute (str or PipelineVariable: Index or JSONpath to
            locate actual label(s) (default: None).
        probability_threshold_attribute (str or PipelineVariable): Threshold to
            convert probabilities to binaries (default: None).
    """

    problem_type: Union[str, PipelineVariable] = attr.ib()
    inference_attribute: Union[str, PipelineVariable] = attr.ib(default=None)
    probability_attribute: Union[str, PipelineVariable] = attr.ib(default=None)
    ground_truth_attribute: Union[str, PipelineVariable] = attr.ib(default=None)
    probability_threshold_attribute: Union[str, PipelineVariable] = attr.ib(default=None)


class QualityCheckStep(Step):
    """QualityCheck step for workflow."""

    def __init__(
        self,
        name: str,
        quality_check_config: QualityCheckConfig,
        check_job_config: CheckJobConfig,
        skip_check: Union[bool, PipelineVariable] = False,
        fail_on_violation: Union[bool, PipelineVariable] = True,
        register_new_baseline: Union[bool, PipelineVariable] = False,
        model_package_group_name: Union[str, PipelineVariable] = None,
        supplied_baseline_statistics: Union[str, PipelineVariable] = None,
        supplied_baseline_constraints: Union[str, PipelineVariable] = None,
        display_name: str = None,
        description: str = None,
        cache_config: CacheConfig = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Constructs a QualityCheckStep.

        Args:
            name (str): The name of the QualityCheckStep step.
            quality_check_config (QualityCheckConfig): A QualityCheckConfig instance.
            check_job_config (CheckJobConfig): A CheckJobConfig instance.
            skip_check (bool or PipelineVariable): Whether the check
                should be skipped (default: False).
            fail_on_violation (bool or PipelineVariable): Whether to fail the step
                if violation detected (default: True).
            register_new_baseline (bool or PipelineVariable): Whether
                the new baseline should be registered (default: False).
            model_package_group_name (str or PipelineVariable): The name of a
                registered model package group, among which the baseline will be fetched
                from the latest approved model (default: None).
            supplied_baseline_statistics (str or PipelineVariable): The S3 path
                to the supplied statistics object representing the statistics JSON file
                which will be used for drift to check (default: None).
            supplied_baseline_constraints (str or PipelineVariable): The S3 path
                to the supplied constraints object representing the constraints JSON file
                which will be used for drift to check (default: None).
            display_name (str): The display name of the QualityCheckStep step (default: None).
            description (str): The description of the QualityCheckStep step (default: None).
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance
                (default: None).
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `QualityCheckStep`
                depends on (default: None).
        """
        if not isinstance(quality_check_config, DataQualityCheckConfig) and not isinstance(
            quality_check_config, ModelQualityCheckConfig
        ):
            raise RuntimeError(
                "The quality_check_config can only be object of "
                + "DataQualityCheckConfig or ModelQualityCheckConfig"
            )

        super(QualityCheckStep, self).__init__(
            name, display_name, description, StepTypeEnum.QUALITY_CHECK, depends_on
        )
        self.skip_check = skip_check
        self.fail_on_violation = fail_on_violation
        self.register_new_baseline = register_new_baseline
        self.check_job_config = check_job_config
        self.quality_check_config = quality_check_config
        self.model_package_group_name = model_package_group_name
        self.supplied_baseline_statistics = supplied_baseline_statistics
        self.supplied_baseline_constraints = supplied_baseline_constraints
        self.cache_config = cache_config

        if isinstance(self.quality_check_config, DataQualityCheckConfig):
            self._model_monitor = self.check_job_config._generate_model_monitor(
                "DefaultModelMonitor"
            )
        else:
            self._model_monitor = self.check_job_config._generate_model_monitor(
                "ModelQualityMonitor"
            )
        self._model_monitor.latest_baselining_job_name = (
            self._model_monitor._generate_baselining_job_name()
        )

        baseline_job_inputs_with_nones = self._generate_baseline_job_inputs()
        self._baseline_job_inputs = [
            baseline_job_input
            for baseline_job_input in baseline_job_inputs_with_nones.values()
            if baseline_job_input is not None
        ]
        self._baseline_output = self._generate_baseline_output()
        self._baselining_processor = self._generate_baseline_processor(
            baseline_dataset_input=baseline_job_inputs_with_nones["baseline_dataset_input"],
            baseline_output=self._baseline_output,
            post_processor_script_input=baseline_job_inputs_with_nones[
                "post_processor_script_input"
            ],
            record_preprocessor_script_input=baseline_job_inputs_with_nones[
                "record_preprocessor_script_input"
            ],
        )

        root_prop = Properties(step_name=name)
        root_prop.__dict__["CalculatedBaselineConstraints"] = Properties(
            step_name=name, path="CalculatedBaselineConstraints"
        )
        root_prop.__dict__["CalculatedBaselineStatistics"] = Properties(
            step_name=name, path="CalculatedBaselineStatistics"
        )
        root_prop.__dict__["BaselineUsedForDriftCheckStatistics"] = Properties(
            step_name=name, path="BaselineUsedForDriftCheckStatistics"
        )
        root_prop.__dict__["BaselineUsedForDriftCheckConstraints"] = Properties(
            step_name=name, path="BaselineUsedForDriftCheckConstraints"
        )
        self._properties = root_prop

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to define the QualityCheck step."""
        from sagemaker.workflow.utilities import _pipeline_config

        normalized_inputs, normalized_outputs = self._baselining_processor._normalize_args(
            inputs=self._baseline_job_inputs,
            outputs=[self._baseline_output],
        )
        process_args = ProcessingJob._get_process_args(
            self._baselining_processor,
            normalized_inputs,
            normalized_outputs,
            experiment_config=dict(),
        )
        request_dict = self._baselining_processor.sagemaker_session._get_process_request(
            **process_args
        )
        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "ProcessingJobName", _pipeline_config)

        return request_dict

    @property
    def properties(self):
        """A Properties object representing the output parameters of the QualityCheck step."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration etc."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        if isinstance(self.quality_check_config, DataQualityCheckConfig):
            request_dict["CheckType"] = _DATA_QUALITY_TYPE
        else:
            request_dict["CheckType"] = _MODEL_QUALITY_TYPE

        request_dict["ModelPackageGroupName"] = self.model_package_group_name
        request_dict["SkipCheck"] = self.skip_check
        request_dict["FailOnViolation"] = self.fail_on_violation
        request_dict["RegisterNewBaseline"] = self.register_new_baseline
        request_dict["SuppliedBaselineStatistics"] = self.supplied_baseline_statistics
        request_dict["SuppliedBaselineConstraints"] = self.supplied_baseline_constraints

        return request_dict

    def _generate_baseline_job_inputs(self):
        """Generates a dict with ProcessingInput objects

        Generates a dict with three ProcessingInput objects: baseline_dataset_input,
            post_processor_script_input and record_preprocessor_script_input

        Returns:
            dict: with three ProcessingInput objects as baseline job inputs
        """
        baseline_dataset = self.quality_check_config.baseline_dataset
        baseline_dataset_des = str(
            pathlib.PurePosixPath(
                _CONTAINER_BASE_PATH, _CONTAINER_INPUT_PATH, _BASELINE_DATASET_INPUT_NAME
            )
        )
        if is_pipeline_variable(baseline_dataset):
            baseline_dataset_input = ProcessingInput(
                source=self.quality_check_config.baseline_dataset,
                destination=baseline_dataset_des,
                input_name=_BASELINE_DATASET_INPUT_NAME,
            )
        else:
            baseline_dataset_input = self._model_monitor._upload_and_convert_to_processing_input(
                source=self.quality_check_config.baseline_dataset,
                destination=baseline_dataset_des,
                name=_BASELINE_DATASET_INPUT_NAME,
            )

        post_processor_script_input = self._model_monitor._upload_and_convert_to_processing_input(
            source=self.quality_check_config.post_analytics_processor_script,
            destination=str(
                pathlib.PurePosixPath(
                    _CONTAINER_BASE_PATH,
                    _CONTAINER_INPUT_PATH,
                    _POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME,
                )
            ),
            name=_POST_ANALYTICS_PROCESSOR_SCRIPT_INPUT_NAME,
        )

        record_preprocessor_script_input = None
        if isinstance(self.quality_check_config, DataQualityCheckConfig):
            record_preprocessor_script_input = (
                self._model_monitor._upload_and_convert_to_processing_input(
                    source=self.quality_check_config.record_preprocessor_script,
                    destination=str(
                        pathlib.PurePosixPath(
                            _CONTAINER_BASE_PATH,
                            _CONTAINER_INPUT_PATH,
                            _RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME,
                        )
                    ),
                    name=_RECORD_PREPROCESSOR_SCRIPT_INPUT_NAME,
                )
            )
        return dict(
            baseline_dataset_input=baseline_dataset_input,
            post_processor_script_input=post_processor_script_input,
            record_preprocessor_script_input=record_preprocessor_script_input,
        )

    def _generate_baseline_output(self):
        """Generates a ProcessingOutput object

        Returns:
            sagemaker.processing.ProcessingOutput: The normalized ProcessingOutput object.
        """
        s3_uri = self.quality_check_config.output_s3_uri or s3.s3_path_join(
            "s3://",
            self._model_monitor.sagemaker_session.default_bucket(),
            self._model_monitor.sagemaker_session.default_bucket_prefix,
            _MODEL_MONITOR_S3_PATH,
            _BASELINING_S3_PATH,
            self._model_monitor.latest_baselining_job_name,
            _RESULTS_S3_PATH,
        )
        return ProcessingOutput(
            source=str(pathlib.PurePosixPath(_CONTAINER_BASE_PATH, _CONTAINER_OUTPUT_PATH)),
            destination=s3_uri,
            output_name=_DEFAULT_OUTPUT_NAME,
        )

    def _generate_baseline_processor(
        self,
        baseline_dataset_input,
        baseline_output,
        post_processor_script_input=None,
        record_preprocessor_script_input=None,
    ):
        """Generates a baseline processor

        Args:
            baseline_dataset_input (ProcessingInput): A ProcessingInput instance for baseline
                dataset input.
            baseline_output (ProcessingOutput): A ProcessingOutput instance for baseline
                dataset output.
            post_processor_script_input (ProcessingInput): A ProcessingInput instance for
                post processor script input.
            record_preprocessor_script_input (ProcessingInput): A ProcessingInput instance for
                record preprocessor script input.

        Returns:
            sagemaker.processing.Processor: The baseline processor
        """
        quality_check_cfg = self.quality_check_config
        # Unlike other input, dataset must be a directory for the Monitoring image.
        baseline_dataset_container_path = baseline_dataset_input.destination

        post_processor_script_container_path = None
        if post_processor_script_input is not None:
            post_processor_script_container_path = str(
                pathlib.PurePosixPath(
                    post_processor_script_input.destination,
                    os.path.basename(quality_check_cfg.post_analytics_processor_script),
                )
            )

        record_preprocessor_script_container_path = None
        if isinstance(quality_check_cfg, DataQualityCheckConfig):
            if record_preprocessor_script_input is not None:
                record_preprocessor_script_container_path = str(
                    pathlib.PurePosixPath(
                        record_preprocessor_script_input.destination,
                        os.path.basename(quality_check_cfg.record_preprocessor_script),
                    )
                )
            normalized_env = ModelMonitor._generate_env_map(
                env=self._model_monitor.env,
                dataset_format=quality_check_cfg.dataset_format,
                output_path=baseline_output.source,
                enable_cloudwatch_metrics=False,  # Only supported for monitoring schedules
                dataset_source_container_path=baseline_dataset_container_path,
                record_preprocessor_script_container_path=record_preprocessor_script_container_path,
                post_processor_script_container_path=post_processor_script_container_path,
            )
        else:
            inference_attribute = (
                str(quality_check_cfg.inference_attribute)
                if quality_check_cfg.inference_attribute is not None
                else None
            )
            probability_attribute = (
                str(quality_check_cfg.probability_attribute)
                if quality_check_cfg.probability_attribute is not None
                else None
            )
            ground_truth_attribute = (
                str(quality_check_cfg.ground_truth_attribute)
                if quality_check_cfg.ground_truth_attribute is not None
                else None
            )
            probability_threshold_attr = (
                str(quality_check_cfg.probability_threshold_attribute)
                if quality_check_cfg.probability_threshold_attribute is not None
                else None
            )
            normalized_env = ModelMonitor._generate_env_map(
                env=self._model_monitor.env,
                dataset_format=quality_check_cfg.dataset_format,
                output_path=baseline_output.source,
                enable_cloudwatch_metrics=False,  # Only supported for monitoring schedules
                dataset_source_container_path=baseline_dataset_container_path,
                post_processor_script_container_path=post_processor_script_container_path,
                analysis_type=_MODEL_QUALITY_TYPE,
                problem_type=quality_check_cfg.problem_type,
                inference_attribute=inference_attribute,
                probability_attribute=probability_attribute,
                ground_truth_attribute=ground_truth_attribute,
                probability_threshold_attribute=probability_threshold_attr,
            )

        return Processor(
            role=self._model_monitor.role,
            image_uri=self._model_monitor.image_uri,
            instance_count=self._model_monitor.instance_count,
            instance_type=self._model_monitor.instance_type,
            entrypoint=self._model_monitor.entrypoint,
            volume_size_in_gb=self._model_monitor.volume_size_in_gb,
            volume_kms_key=self._model_monitor.volume_kms_key,
            output_kms_key=self._model_monitor.output_kms_key,
            max_runtime_in_seconds=self._model_monitor.max_runtime_in_seconds,
            base_job_name=self._model_monitor.base_job_name,
            sagemaker_session=self._model_monitor.sagemaker_session,
            env=normalized_env,
            tags=self._model_monitor.tags,
            network_config=self._model_monitor.network_config,
        )
