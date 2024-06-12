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

import copy
import json
import os
import tempfile
from abc import ABC
from typing import List, Union, Optional

import attr

from sagemaker import s3
from sagemaker.clarify import (
    DataConfig,
    BiasConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
    ProcessingOutputHandler,
    _upload_analysis_config,
    SageMakerClarifyProcessor,
    _set,
)
from sagemaker.model_monitor import BiasAnalysisConfig, ExplainabilityAnalysisConfig
from sagemaker.model_monitor.model_monitoring import _MODEL_MONITOR_S3_PATH
from sagemaker.processing import ProcessingInput, ProcessingOutput, ProcessingJob
from sagemaker.utils import name_from_base
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import RequestType, PipelineVariable
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.steps import Step, StepTypeEnum, CacheConfig
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.utilities import trim_request_dict

_DATA_BIAS_TYPE = "DATA_BIAS"
_MODEL_BIAS_TYPE = "MODEL_BIAS"
_MODEL_EXPLAINABILITY_TYPE = "MODEL_EXPLAINABILITY"
_BIAS_MONITORING_CFG_BASE_NAME = "bias-monitoring"
_EXPLAINABILITY_MONITORING_CFG_BASE_NAME = "model-explainability-monitoring"


@attr.s
class ClarifyCheckConfig(ABC):
    """Clarify Check Config

    Attributes:
        data_config (DataConfig): Config of the input/output data.
        kms_key (str): The ARN of the KMS key that is used to encrypt the
            user code file (default: None).
            This field CANNOT be any type of the `PipelineVariable`.
        monitoring_analysis_config_uri: (str): The uri of monitoring analysis config.
            This field does not take input.
            It will be generated once uploading the created analysis config file.
    """

    data_config: DataConfig = attr.ib()
    kms_key: str = attr.ib(kw_only=True, default=None)
    monitoring_analysis_config_uri: str = attr.ib(kw_only=True, default=None)


@attr.s
class DataBiasCheckConfig(ClarifyCheckConfig):
    """Data Bias Check Config

    Attributes:
        data_bias_config (BiasConfig): Config of sensitive groups.
        methods (str or list[str]): Selector of a subset of potential metrics:
            ["`CI <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-bias-metric-class-imbalance.html>`_",
            "`DPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-true-label-imbalance.html>`_",
            "`KL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kl-divergence.html>`_",
            "`JS <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-jensen-shannon-divergence.html>`_",
            "`LP <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-lp-norm.html>`_",
            "`TVD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-total-variation-distance.html>`_",
            "`KS <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kolmogorov-smirnov.html>`_",
            "`CDDL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-cddl.html>`_"].
            Defaults to computing all.
            This field CANNOT be any type of the `PipelineVariable`.
    """  # noqa E501

    data_bias_config: BiasConfig = attr.ib()
    methods: Union[str, List[str]] = attr.ib(default="all")


@attr.s
class ModelBiasCheckConfig(ClarifyCheckConfig):
    """Model Bias Check Config

    Attributes:
        data_bias_config (BiasConfig): Config of sensitive groups.
        model_config (ModelConfig): Config of the model and its endpoint to be created.
        model_predicted_label_config (ModelPredictedLabelConfig): Config of how to
            extract the predicted label from the model output.
        methods (str or list[str]): Selector of a subset of potential metrics:
            ["`DPPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dppl.html>`_"
            , "`DI <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-di.html>`_",
            "`DCA <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dca.html>`_",
            "`DCR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dcr.html>`_",
            "`RD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-rd.html>`_",
            "`DAR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dar.html>`_",
            "`DRR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-drr.html>`_",
            "`AD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-ad.html>`_",
            "`CDDPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-cddpl.html>`_
            ", "`TE <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-te.html>`_",
            "`FT <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-ft.html>`_"].
            Defaults to computing all.
            This field CANNOT be any type of the `PipelineVariable`.
    """

    data_bias_config: BiasConfig = attr.ib()
    model_config: ModelConfig = attr.ib()
    model_predicted_label_config: ModelPredictedLabelConfig = attr.ib()
    methods: Union[str, List[str]] = attr.ib(default="all")


@attr.s
class ModelExplainabilityCheckConfig(ClarifyCheckConfig):
    """Model Explainability Check Config

    Attributes:
        model_config (ModelConfig): Config of the model and its endpoint to be created.
        explainability_config (SHAPConfig): Config of the specific explainability method.
            Currently, only SHAP is supported.
        model_scores (str or int or ModelPredictedLabelConfig): Index or JMESPath expression
            to locate the predicted scores in the model output (default: None).
            This is not required if the model output is a single score. Alternatively,
            an instance of ModelPredictedLabelConfig can be provided
            but this field CANNOT be any type of the `PipelineVariable`.
    """

    model_config: ModelConfig = attr.ib()
    explainability_config: SHAPConfig = attr.ib()
    model_scores: Union[str, int, ModelPredictedLabelConfig] = attr.ib(default=None)


class ClarifyCheckStep(Step):
    """ClarifyCheckStep step for workflow."""

    def __init__(
        self,
        name: str,
        clarify_check_config: ClarifyCheckConfig,
        check_job_config: CheckJobConfig,
        skip_check: Union[bool, PipelineVariable] = False,
        fail_on_violation: Union[bool, PipelineVariable] = True,
        register_new_baseline: Union[bool, PipelineVariable] = False,
        model_package_group_name: Union[str, PipelineVariable] = None,
        supplied_baseline_constraints: Union[str, PipelineVariable] = None,
        display_name: str = None,
        description: str = None,
        cache_config: CacheConfig = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Constructs a ClarifyCheckStep.

        Args:
            name (str): The name of the ClarifyCheckStep step.
            clarify_check_config (ClarifyCheckConfig): A ClarifyCheckConfig instance.
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
            supplied_baseline_constraints (str or PipelineVariable): The S3 path
                to the supplied constraints object representing the constraints JSON file
                which will be used for drift to check (default: None).
            display_name (str): The display name of the ClarifyCheckStep step (default: None).
            description (str): The description of the ClarifyCheckStep step (default: None).
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance
                (default: None).
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `ClarifyCheckStep`
                depends on (default: None).
        """
        if (
            not isinstance(clarify_check_config, DataBiasCheckConfig)
            and not isinstance(clarify_check_config, ModelBiasCheckConfig)
            and not isinstance(clarify_check_config, ModelExplainabilityCheckConfig)
        ):
            raise RuntimeError(
                "The clarify_check_config can only be object of "
                + "DataBiasCheckConfig, ModelBiasCheckConfig or ModelExplainabilityCheckConfig"
            )

        if is_pipeline_variable(clarify_check_config.data_config.s3_analysis_config_output_path):
            raise RuntimeError(
                "s3_analysis_config_output_path cannot be of type "
                + "ExecutionVariable/Expression/Parameter/Properties"
            )

        if (
            not clarify_check_config.data_config.s3_analysis_config_output_path
            and is_pipeline_variable(clarify_check_config.data_config.s3_output_path)
        ):
            raise RuntimeError(
                "`s3_output_path` cannot be of type ExecutionVariable/Expression/Parameter"
                + "/Properties if `s3_analysis_config_output_path` is none or empty "
            )

        super(ClarifyCheckStep, self).__init__(
            name, display_name, description, StepTypeEnum.CLARIFY_CHECK, depends_on
        )
        self.skip_check = skip_check
        self.fail_on_violation = fail_on_violation
        self.register_new_baseline = register_new_baseline
        self.clarify_check_config = clarify_check_config
        self.check_job_config = check_job_config
        self.model_package_group_name = model_package_group_name
        self.supplied_baseline_constraints = supplied_baseline_constraints
        self.cache_config = cache_config

        if isinstance(self.clarify_check_config, ModelExplainabilityCheckConfig):
            self._model_monitor = self.check_job_config._generate_model_monitor(
                "ModelExplainabilityMonitor"
            )
        else:
            self._model_monitor = self.check_job_config._generate_model_monitor("ModelBiasMonitor")

        self.clarify_check_config.monitoring_analysis_config_uri = (
            self._upload_monitoring_analysis_config()
        )
        self._baselining_processor = self._model_monitor._create_baselining_processor()
        self._processing_params = self._generate_processing_job_parameters(
            self._generate_processing_job_analysis_config(), self._baselining_processor
        )

        root_prop = Properties(step_name=name)
        root_prop.__dict__["CalculatedBaselineConstraints"] = Properties(
            step_name=name, path="CalculatedBaselineConstraints"
        )
        root_prop.__dict__["BaselineUsedForDriftCheckConstraints"] = Properties(
            step_name=name, path="BaselineUsedForDriftCheckConstraints"
        )
        self._properties = root_prop

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to define the ClarifyCheck step."""
        from sagemaker.workflow.utilities import _pipeline_config

        normalized_inputs, normalized_outputs = self._baselining_processor._normalize_args(
            inputs=[self._processing_params["config_input"], self._processing_params["data_input"]],
            outputs=[self._processing_params["result_output"]],
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
        """A Properties object representing the output parameters of the ClarifyCheck step."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration etc."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        if isinstance(self.clarify_check_config, DataBiasCheckConfig):
            request_dict["CheckType"] = _DATA_BIAS_TYPE
        elif isinstance(self.clarify_check_config, ModelBiasCheckConfig):
            request_dict["CheckType"] = _MODEL_BIAS_TYPE
        else:
            request_dict["CheckType"] = _MODEL_EXPLAINABILITY_TYPE

        request_dict["ModelPackageGroupName"] = self.model_package_group_name
        request_dict["SkipCheck"] = self.skip_check
        request_dict["FailOnViolation"] = self.fail_on_violation
        request_dict["RegisterNewBaseline"] = self.register_new_baseline
        request_dict["SuppliedBaselineConstraints"] = self.supplied_baseline_constraints
        if isinstance(
            self.clarify_check_config, (ModelBiasCheckConfig, ModelExplainabilityCheckConfig)
        ):
            request_dict[
                "ModelName"
            ] = self.clarify_check_config.model_config.get_predictor_config()["model_name"]
        return request_dict

    def _generate_processing_job_analysis_config(self) -> dict:
        """Generate the clarify processing job analysis config

        Returns:
            dict: processing job analysis config dictionary.
        """
        analysis_config = self.clarify_check_config.data_config.get_config()
        if isinstance(self.clarify_check_config, DataBiasCheckConfig):
            analysis_config.update(self.clarify_check_config.data_bias_config.get_config())
            analysis_config["methods"] = {
                "pre_training_bias": {"methods": self.clarify_check_config.methods}
            }
        elif isinstance(self.clarify_check_config, ModelBiasCheckConfig):
            analysis_config.update(self.clarify_check_config.data_bias_config.get_config())
            (
                probability_threshold,
                predictor_config,
            ) = self.clarify_check_config.model_predicted_label_config.get_predictor_config()
            predictor_config.update(self.clarify_check_config.model_config.get_predictor_config())
            if "model_name" in predictor_config:
                predictor_config.pop("model_name")
            analysis_config["methods"] = {
                "post_training_bias": {"methods": self.clarify_check_config.methods}
            }
            analysis_config["predictor"] = predictor_config
            _set(probability_threshold, "probability_threshold", analysis_config)
        else:
            predictor_config = self.clarify_check_config.model_config.get_predictor_config()
            if "model_name" in predictor_config:
                predictor_config.pop("model_name")
            model_scores = self.clarify_check_config.model_scores
            if isinstance(model_scores, ModelPredictedLabelConfig):
                probability_threshold, predicted_label_config = model_scores.get_predictor_config()
                _set(probability_threshold, "probability_threshold", analysis_config)
                predictor_config.update(predicted_label_config)
            else:
                _set(model_scores, "label", predictor_config)
            analysis_config[
                "methods"
            ] = self.clarify_check_config.explainability_config.get_explainability_config()
            analysis_config["predictor"] = predictor_config
        return analysis_config

    def _generate_processing_job_parameters(
        self, analysis_config: dict, baselining_processor: SageMakerClarifyProcessor
    ) -> dict:
        """Generates input and output parameters for the clarify processing job

        Args:
            analysis_config (dict): A clarify processing job analysis config
            baselining_processor (SageMakerClarifyProcessor): A SageMakerClarifyProcessor instance

        Returns:
            dict: with two ProcessingInput objects as the clarify processing job inputs and
                a ProcessingOutput object as the clarify processing job output parameter
        """
        data_config = self.clarify_check_config.data_config
        analysis_config["methods"]["report"] = {"name": "report", "title": "Analysis Report"}

        with tempfile.TemporaryDirectory() as tmpdirname:
            analysis_config_file = os.path.join(tmpdirname, "analysis_config.json")
            with open(analysis_config_file, "w") as f:
                json.dump(analysis_config, f)
            s3_analysis_config_file = _upload_analysis_config(
                analysis_config_file,
                data_config.s3_analysis_config_output_path or data_config.s3_output_path,
                baselining_processor.sagemaker_session,
                self.clarify_check_config.kms_key,
            )
            config_input = ProcessingInput(
                input_name="analysis_config",
                source=s3_analysis_config_file,
                destination=SageMakerClarifyProcessor._CLARIFY_CONFIG_INPUT,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_compression_type="None",
            )
            data_input = ProcessingInput(
                input_name="dataset",
                source=data_config.s3_data_input_path,
                destination=SageMakerClarifyProcessor._CLARIFY_DATA_INPUT,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type=data_config.s3_data_distribution_type,
                s3_compression_type=data_config.s3_compression_type,
            )
            result_output = ProcessingOutput(
                source=SageMakerClarifyProcessor._CLARIFY_OUTPUT,
                destination=data_config.s3_output_path,
                output_name="analysis_result",
                s3_upload_mode=ProcessingOutputHandler.get_s3_upload_mode(analysis_config),
            )
        return dict(config_input=config_input, data_input=data_input, result_output=result_output)

    def _upload_monitoring_analysis_config(self) -> str:
        """Generate and upload monitoring schedule analysis config to s3

        Returns:
            str: The S3 uri of the uploaded monitoring schedule analysis config
        """

        output_s3_uri = self._get_s3_base_uri_for_monitoring_analysis_config()

        if isinstance(self.clarify_check_config, ModelExplainabilityCheckConfig):
            # Explainability analysis doesn't need label
            headers = copy.deepcopy(self.clarify_check_config.data_config.headers)
            if headers and self.clarify_check_config.data_config.label in headers:
                headers.remove(self.clarify_check_config.data_config.label)
            explainability_analysis_config = ExplainabilityAnalysisConfig(
                explainability_config=self.clarify_check_config.explainability_config,
                model_config=self.clarify_check_config.model_config,
                headers=headers,
            )
            analysis_config = explainability_analysis_config._to_dict()
            if "predictor" in analysis_config and "model_name" in analysis_config["predictor"]:
                analysis_config["predictor"].pop("model_name")
            job_definition_name = name_from_base(
                f"{_EXPLAINABILITY_MONITORING_CFG_BASE_NAME}-config"
            )

        else:
            bias_analysis_config = BiasAnalysisConfig(
                bias_config=self.clarify_check_config.data_bias_config,
                headers=self.clarify_check_config.data_config.headers,
                label=self.clarify_check_config.data_config.label,
            )
            analysis_config = bias_analysis_config._to_dict()
            job_definition_name = name_from_base(f"{_BIAS_MONITORING_CFG_BASE_NAME}-config")

        return self._model_monitor._upload_analysis_config(
            analysis_config, output_s3_uri, job_definition_name, self.clarify_check_config.kms_key
        )

    def _get_s3_base_uri_for_monitoring_analysis_config(self) -> str:
        """Generate s3 base uri for monitoring schedule analysis config

        Returns:
            str: The S3 base uri of the monitoring schedule analysis config
        """
        s3_analysis_config_output_path = (
            self.clarify_check_config.data_config.s3_analysis_config_output_path
        )
        monitoring_cfg_base_name = f"{_BIAS_MONITORING_CFG_BASE_NAME}-configuration"
        if isinstance(self.clarify_check_config, ModelExplainabilityCheckConfig):
            monitoring_cfg_base_name = f"{_EXPLAINABILITY_MONITORING_CFG_BASE_NAME}-configuration"

        if s3_analysis_config_output_path:
            return s3.s3_path_join(
                s3_analysis_config_output_path,
                monitoring_cfg_base_name,
            )
        return s3.s3_path_join(
            "s3://",
            self._model_monitor.sagemaker_session.default_bucket(),
            self._model_monitor.sagemaker_session.default_bucket_prefix,
            _MODEL_MONITOR_S3_PATH,
            monitoring_cfg_base_name,
        )
