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
"""The `AutoMLStep` definition for SageMaker Pipelines Workflows"""
from __future__ import absolute_import

from typing import Union, Optional, List

from sagemaker import Session, Model
from sagemaker.exceptions import AutoMLStepInvalidModeError
from sagemaker.workflow.entities import RequestType

from sagemaker.workflow.pipeline_context import _JobStepArguments
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.retry import RetryPolicy
from sagemaker.workflow.steps import ConfigurableRetryStep, CacheConfig, Step, StepTypeEnum
from sagemaker.workflow.utilities import validate_step_args_input, trim_request_dict
from sagemaker.workflow.step_collections import StepCollection


class AutoMLStep(ConfigurableRetryStep):
    """`AutoMLStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments,
        display_name: str = None,
        description: str = None,
        cache_config: CacheConfig = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        """Construct a `AutoMLStep`, given a `AutoML` instance.

        In addition to the `AutoML` instance, the other arguments are those
        that are supplied to the `fit` method of the `sagemaker.automl.automl.AutoML`.

        Args:
            name (str): The name of the `AutoMLStep`.
            step_args (_JobStepArguments): The arguments for the `AutoMLStep` definition.
            display_name (str): The display name of the `AutoMLStep`.
            description (str): The description of the `AutoMLStep`.
            cache_config (CacheConfig): A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `AutoMLStep`
                depends on.
            retry_policies (List[RetryPolicy]): A list of retry policies.
        """
        super(AutoMLStep, self).__init__(
            name, StepTypeEnum.AUTOML, display_name, description, depends_on, retry_policies
        )

        validate_step_args_input(
            step_args=step_args,
            expected_caller={Session.auto_ml.__name__},
            error_message="The step_args of AutoMLStep must be obtained " "from automl.fit().",
        )

        self.step_args = step_args
        self.cache_config = cache_config

        root_property = Properties(step_name=name, shape_name="DescribeAutoMLJobResponse")

        best_candidate_properties = Properties(step_name=name, path="BestCandidateProperties")
        best_candidate_properties.__dict__["ModelInsightsJsonReportPath"] = Properties(
            step_name=name, path="BestCandidateProperties.ModelInsightsJsonReportPath"
        )
        best_candidate_properties.__dict__["ExplainabilityJsonReportPath"] = Properties(
            step_name=name, path="BestCandidateProperties.ExplainabilityJsonReportPath"
        )

        root_property.__dict__["BestCandidateProperties"] = best_candidate_properties
        self._properties = root_property

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_auto_ml_job`.

        NOTE: The `CreateAutoMLJob` request is not quite the
            args list that workflow needs.

        `ModelDeployConfig` and `GenerateCandidateDefinitionsOnly`
            attribute cannot be included.
        """
        from sagemaker.workflow.utilities import execute_job_functions
        from sagemaker.workflow.utilities import _pipeline_config

        # execute fit function in AutoML with saved parameters,
        # and store args in PipelineSession's _context
        execute_job_functions(self.step_args)

        # populate request dict with args
        auto_ml = self.step_args.func_args[0]
        request_dict = auto_ml.sagemaker_session.context.args

        if "AutoMLJobConfig" not in request_dict:
            raise AutoMLStepInvalidModeError()
        if (
            "Mode" not in request_dict["AutoMLJobConfig"]
            or request_dict["AutoMLJobConfig"]["Mode"] != "ENSEMBLING"
        ):
            raise AutoMLStepInvalidModeError()

        if "ModelDeployConfig" in request_dict:
            request_dict.pop("ModelDeployConfig", None)
        if "GenerateCandidateDefinitionsOnly" in request_dict:
            request_dict.pop("GenerateCandidateDefinitionsOnly", None)
        # Continue to pop job name if not explicitly opted-in via config
        # AutoML Trims to AutoMLJo-2023-06-23-22-57-39-083
        request_dict = trim_request_dict(request_dict, "AutoMLJobName", _pipeline_config)

        return request_dict

    @property
    def properties(self):
        """A `Properties` object representing the `DescribeAutoMLJobResponse` data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict

    def get_best_auto_ml_model(self, role, sagemaker_session=None):
        """Get the best candidate model artifacts, image uri and env variables for the best model.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker AutoML jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions.
                If the best model will be used as part of ModelStep, then sagemaker_session
                should be class:`~sagemaker.workflow.pipeline_context.PipelineSession`. Example::
                     model = Model(sagemaker_session=PipelineSession())
                     model_step = ModelStep(step_args=model.register())
        """
        inference_container = self.properties.BestCandidate.InferenceContainers[0]
        inference_container_environment = inference_container.Environment
        image = inference_container.Image
        model_data = inference_container.ModelDataUrl
        model = Model(
            image_uri=image,
            model_data=model_data,
            env={
                "MODEL_NAME": inference_container_environment["MODEL_NAME"],
                "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT": inference_container_environment[
                    "SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT"
                ],
                "SAGEMAKER_SUBMIT_DIRECTORY": inference_container_environment[
                    "SAGEMAKER_SUBMIT_DIRECTORY"
                ],
                "SAGEMAKER_INFERENCE_SUPPORTED": inference_container_environment[
                    "SAGEMAKER_INFERENCE_SUPPORTED"
                ],
                "SAGEMAKER_INFERENCE_OUTPUT": inference_container_environment[
                    "SAGEMAKER_INFERENCE_OUTPUT"
                ],
                "SAGEMAKER_PROGRAM": inference_container_environment["SAGEMAKER_PROGRAM"],
            },
            sagemaker_session=sagemaker_session,
            role=role,
        )

        return model
