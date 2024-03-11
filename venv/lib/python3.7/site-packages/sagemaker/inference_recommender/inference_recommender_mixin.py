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

import logging
import re

from typing import List, Dict, Optional
import sagemaker
from sagemaker.parameter import CategoricalParameter

INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING = {
    "xgboost": "XGBOOST",
    "sklearn": "SAGEMAKER-SCIKIT-LEARN",
    "pytorch": "PYTORCH",
    "tensorflow": "TENSORFLOW",
    "mxnet": "MXNET",
}

LOGGER = logging.getLogger("sagemaker")


class Phase:
    """Used to store phases of a traffic pattern to perform endpoint load testing.

    Required for an Advanced Inference Recommendations Job
    """

    def __init__(self, duration_in_seconds: int, initial_number_of_users: int, spawn_rate: int):
        """Initialize a `Phase`"""
        self.to_json = {
            "DurationInSeconds": duration_in_seconds,
            "InitialNumberOfUsers": initial_number_of_users,
            "SpawnRate": spawn_rate,
        }


class ModelLatencyThreshold:
    """Used to store inference request/response latency to perform endpoint load testing.

    Required for an Advanced Inference Recommendations Job
    """

    def __init__(self, percentile: str, value_in_milliseconds: int):
        """Initialize a `ModelLatencyThreshold`"""
        self.to_json = {"Percentile": percentile, "ValueInMilliseconds": value_in_milliseconds}


class InferenceRecommenderMixin:
    """A mixin class for SageMaker ``Inference Recommender`` that will be extended by ``Model``"""

    def right_size(
        self,
        sample_payload_url: str = None,
        supported_content_types: List[str] = None,
        supported_instance_types: List[str] = None,
        job_name: str = None,
        framework: str = None,
        job_duration_in_seconds: int = None,
        hyperparameter_ranges: List[Dict[str, CategoricalParameter]] = None,
        phases: List[Phase] = None,
        traffic_type: str = None,
        max_invocations: int = None,
        model_latency_thresholds: List[ModelLatencyThreshold] = None,
        max_tests: int = None,
        max_parallel_tests: int = None,
        log_level: Optional[str] = "Verbose",
    ):
        """Recommends an instance type for a SageMaker or BYOC model.

        Create a SageMaker ``Model`` or use a registered ``ModelPackage``,
        to start an Inference Recommender job.

        The name of the created model is accessible in the ``name`` field of
        this ``Model`` after right_size returns.

        Args:
            sample_payload_url (str): The S3 path where the sample payload is stored.
            supported_content_types: (list[str]): The supported MIME types for the input data.
            supported_instance_types (list[str]): A list of the instance types that this model
                is expected to work on. (default: None).
            job_name (str): The name of the Inference Recommendations Job. (default: None).
            framework (str): The machine learning framework of the Image URI.
                Only required to specify if you bring your own custom containers (default: None).
            job_duration_in_seconds (int): The maximum job duration that a job can run for.
                (default: None).
            hyperparameter_ranges (list[Dict[str, sagemaker.parameter.CategoricalParameter]]):
                Specifies the hyper parameters to be used during endpoint load tests.
                `instance_type` must be specified as a hyperparameter range.
                `env_vars` can be specified as an optional hyperparameter range. (default: None).
                Example::

                    hyperparameter_ranges = [{
                        'instance_types': CategoricalParameter(['ml.c5.xlarge', 'ml.c5.2xlarge']),
                        'OMP_NUM_THREADS': CategoricalParameter(['1', '2', '3', '4'])
                    }]

            phases (list[Phase]): Shape of the traffic pattern to use in the load test
                (default: None).
            traffic_type (str): Specifies the traffic pattern type. Currently only supports
                one type 'PHASES' (default: None).
            max_invocations (str): defines the minimum invocations per minute for the endpoint
                to support (default: None).
            model_latency_thresholds (list[ModelLatencyThreshold]): defines the maximum response
                latency for endpoints to support (default: None).
            max_tests (int): restricts how many endpoints in total are allowed to be
                spun up for this job (default: None).
            max_parallel_tests (int): restricts how many concurrent endpoints
                this job is allowed to spin up (default: None).
            log_level (str): specifies the inline output when waiting for right_size to complete
                (default: "Verbose").

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """

        if not framework and self._framework():
            framework = INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING.get(self._framework(), framework)

        framework_version = self._get_framework_version()

        endpoint_configurations = self._convert_to_endpoint_configurations_json(
            hyperparameter_ranges=hyperparameter_ranges
        )
        traffic_pattern = self._convert_to_traffic_pattern_json(
            traffic_type=traffic_type, phases=phases
        )
        stopping_conditions = self._convert_to_stopping_conditions_json(
            max_invocations=max_invocations, model_latency_thresholds=model_latency_thresholds
        )
        resource_limit = self._convert_to_resource_limit_json(
            max_tests=max_tests, max_parallel_tests=max_parallel_tests
        )

        if endpoint_configurations or traffic_pattern or stopping_conditions or resource_limit:
            LOGGER.info("Advanced Job parameters were specified. Running Advanced job...")
            job_type = "Advanced"
        else:
            LOGGER.info("Advanced Job parameters were not specified. Running Default job...")
            job_type = "Default"

        self._init_sagemaker_session_if_does_not_exist()

        if isinstance(self, sagemaker.model.Model) and not isinstance(
            self, sagemaker.model.ModelPackage
        ):
            primary_container_def = self.prepare_container_def()
            if not self.name:
                self._ensure_base_name_if_needed(
                    image_uri=primary_container_def["Image"],
                    script_uri=self.source_dir,
                    model_uri=self.model_data,
                )
                self._set_model_name_if_needed()

            create_model_args = dict(
                name=self.name,
                role=self.role,
                container_defs=None,
                primary_container=primary_container_def,
                vpc_config=self.vpc_config,
                enable_network_isolation=self.enable_network_isolation(),
            )
            LOGGER.warning("Attempting to create new model with name %s", self.name)
            self.sagemaker_session.create_model(**create_model_args)

        ret_name = self.sagemaker_session.create_inference_recommendations_job(
            role=self.role,
            job_name=job_name,
            job_type=job_type,
            job_duration_in_seconds=job_duration_in_seconds,
            model_name=self.name,
            model_package_version_arn=getattr(self, "model_package_arn", None),
            framework=framework,
            framework_version=framework_version,
            sample_payload_url=sample_payload_url,
            supported_content_types=supported_content_types,
            supported_instance_types=supported_instance_types,
            endpoint_configurations=endpoint_configurations,
            traffic_pattern=traffic_pattern,
            stopping_conditions=stopping_conditions,
            resource_limit=resource_limit,
        )

        self.inference_recommender_job_results = (
            self.sagemaker_session.wait_for_inference_recommendations_job(
                ret_name, log_level=log_level
            )
        )
        self.inference_recommendations = self.inference_recommender_job_results.get(
            "InferenceRecommendations"
        )

        return self

    def _update_params(
        self,
        **kwargs,
    ):
        """Check and update params based on inference recommendation id or right size case"""
        instance_type = kwargs["instance_type"]
        initial_instance_count = kwargs["initial_instance_count"]
        accelerator_type = kwargs["accelerator_type"]
        async_inference_config = kwargs["async_inference_config"]
        serverless_inference_config = kwargs["serverless_inference_config"]
        explainer_config = kwargs["explainer_config"]
        inference_recommendation_id = kwargs["inference_recommendation_id"]
        inference_recommender_job_results = kwargs["inference_recommender_job_results"]
        if inference_recommendation_id is not None:
            inference_recommendation = self._update_params_for_recommendation_id(
                instance_type=instance_type,
                initial_instance_count=initial_instance_count,
                accelerator_type=accelerator_type,
                async_inference_config=async_inference_config,
                serverless_inference_config=serverless_inference_config,
                inference_recommendation_id=inference_recommendation_id,
                explainer_config=explainer_config,
            )
        elif inference_recommender_job_results is not None:
            inference_recommendation = self._update_params_for_right_size(
                instance_type,
                initial_instance_count,
                accelerator_type,
                serverless_inference_config,
                async_inference_config,
                explainer_config,
            )

        return (
            inference_recommendation
            if inference_recommendation
            else (instance_type, initial_instance_count)
        )

    def _update_params_for_right_size(
        self,
        instance_type=None,
        initial_instance_count=None,
        accelerator_type=None,
        serverless_inference_config=None,
        async_inference_config=None,
        explainer_config=None,
    ):
        """Validates that Inference Recommendation parameters can be used in `model.deploy()`

        Args:
            instance_type (str): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. If not using
                serverless inference or the model has not called ``right_size()``,
                then it need to be a number larger or equals
                to 1 (default: None)
            initial_instance_count (int):The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference or the model has not called ``right_size()``,
                then it is required to deploy a model.
                (default: None)
            accelerator_type (str): whether accelerator_type has been passed into `model.deploy()`.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig)):
                whether serverless_inference_config has been passed into `model.deploy()`.
            async_inference_config (sagemaker.model_monitor.AsyncInferenceConfig):
                whether async_inference_config has been passed into `model.deploy()`.
            explainer_config (sagemaker.explainer.ExplainerConfig): whether explainer_config
                has been passed into `model.deploy()`.

        Returns:
            (string, int) or None: Top instance_type and associated initial_instance_count
            if self.inference_recommender_job_results has been generated. Otherwise, return None.
        """
        if accelerator_type:
            raise ValueError("accelerator_type is not compatible with right_size().")
        if instance_type or initial_instance_count:
            LOGGER.warning(
                "instance_type or initial_instance_count specified."
                "Overriding right_size() recommendations."
            )
            return None
        if async_inference_config:
            LOGGER.warning(
                "async_inference_config is specified. Overriding right_size() recommendations."
            )
            return None
        if serverless_inference_config:
            LOGGER.warning(
                "serverless_inference_config is specified. Overriding right_size() recommendations."
            )
            return None
        if explainer_config:
            LOGGER.warning(
                "explainer_config is specified. Overriding right_size() recommendations."
            )
            return None

        instance_type = self.inference_recommendations[0]["EndpointConfiguration"]["InstanceType"]
        initial_instance_count = self.inference_recommendations[0]["EndpointConfiguration"][
            "InitialInstanceCount"
        ]
        return (instance_type, initial_instance_count)

    def _update_params_for_recommendation_id(
        self,
        instance_type,
        initial_instance_count,
        accelerator_type,
        async_inference_config,
        serverless_inference_config,
        inference_recommendation_id,
        explainer_config,
    ):
        """Update parameters with inference recommendation results.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode. If not using
                serverless inference, then it is required to deploy a model.
            initial_instance_count (int): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``. If not using
                serverless inference, then it need to be a number larger or equals
                to 1.
            accelerator_type (str): Type of Elastic Inference accelerator to
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            async_inference_config (sagemaker.model_monitor.AsyncInferenceConfig): Specifies
                configuration related to async endpoint. Use this configuration when trying
                to create async endpoint and make async inference. If empty config object
                passed through, will use default config to deploy async endpoint. Deploy a
                real-time endpoint if it's None.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Use this configuration
                when trying to create serverless endpoint and make serverless inference. If
                empty object passed through, will use pre-defined values in
                ``ServerlessInferenceConfig`` class to deploy serverless endpoint. Deploy an
                instance based endpoint if it's None.
            inference_recommendation_id (str): The recommendation id which specifies
                the recommendation you picked from inference recommendation job
                results and would like to deploy the model and endpoint with
                recommended parameters.
            explainer_config (sagemaker.explainer.ExplainerConfig): Specifies online explainability
                configuration for use with Amazon SageMaker Clarify. Default: None.
        Raises:
            ValueError: If arguments combination check failed in these circumstances:
                - If only one of instance type or instance count specified or
                - If recommendation id does not follow the required format or
                - If recommendation id is not valid or
                - If inference recommendation id is specified along with incompatible parameters
        Returns:
            (string, int): instance type and associated instance count from selected
            inference recommendation id if arguments combination check passed.
        """

        if instance_type is not None and initial_instance_count is not None:
            LOGGER.warning(
                "Both instance_type and initial_instance_count are specified,"
                "overriding the recommendation result."
            )
            return (instance_type, initial_instance_count)

        # Validate non-compatible parameters with recommendation id
        if accelerator_type is not None:
            raise ValueError("accelerator_type is not compatible with inference_recommendation_id.")
        if async_inference_config is not None:
            raise ValueError(
                "async_inference_config is not compatible with inference_recommendation_id."
            )
        if serverless_inference_config is not None:
            raise ValueError(
                "serverless_inference_config is not compatible with inference_recommendation_id."
            )
        if explainer_config is not None:
            raise ValueError("explainer_config is not compatible with inference_recommendation_id.")

        # Validate recommendation id
        if not re.match(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,63}\/\w{8}$", inference_recommendation_id):
            raise ValueError("inference_recommendation_id is not valid")
        job_or_model_name = inference_recommendation_id.split("/")[0]

        sage_client = self.sagemaker_session.sagemaker_client
        # Get recommendation from right size job and model
        (
            right_size_recommendation,
            model_recommendation,
            right_size_job_res,
        ) = self._get_recommendation(
            sage_client=sage_client,
            job_or_model_name=job_or_model_name,
            inference_recommendation_id=inference_recommendation_id,
        )

        # Update params beased on model recommendation
        if model_recommendation:
            if initial_instance_count is None:
                raise ValueError("Must specify model recommendation id and instance count.")
            self.env.update(model_recommendation["Environment"])
            instance_type = model_recommendation["InstanceType"]
            return (instance_type, initial_instance_count)

        # Update params based on default inference recommendation
        if bool(instance_type) != bool(initial_instance_count):
            raise ValueError(
                "instance_type and initial_instance_count are mutually exclusive with"
                "recommendation id since they are in recommendation."
                "Please specify both of them if you want to override the recommendation."
            )
        input_config = right_size_job_res["InputConfig"]
        model_config = right_size_recommendation["ModelConfiguration"]
        envs = (
            model_config["EnvironmentParameters"]
            if "EnvironmentParameters" in model_config
            else None
        )
        # Update envs
        recommend_envs = {}
        if envs is not None:
            for env in envs:
                recommend_envs[env["Key"]] = env["Value"]
        self.env.update(recommend_envs)

        # Update params with non-compilation recommendation results
        if (
            "InferenceSpecificationName" not in model_config
            and "CompilationJobName" not in model_config
        ):

            if "ModelPackageVersionArn" in input_config:
                modelpkg_res = sage_client.describe_model_package(
                    ModelPackageName=input_config["ModelPackageVersionArn"]
                )
                self.model_data = modelpkg_res["InferenceSpecification"]["Containers"][0][
                    "ModelDataUrl"
                ]
                self.image_uri = modelpkg_res["InferenceSpecification"]["Containers"][0]["Image"]
            elif "ModelName" in input_config:
                model_res = sage_client.describe_model(ModelName=input_config["ModelName"])
                self.model_data = model_res["PrimaryContainer"]["ModelDataUrl"]
                self.image_uri = model_res["PrimaryContainer"]["Image"]
        else:
            if "InferenceSpecificationName" in model_config:
                modelpkg_res = sage_client.describe_model_package(
                    ModelPackageName=input_config["ModelPackageVersionArn"]
                )
                self.model_data = modelpkg_res["AdditionalInferenceSpecificationDefinition"][
                    "Containers"
                ][0]["ModelDataUrl"]
                self.image_uri = modelpkg_res["AdditionalInferenceSpecificationDefinition"][
                    "Containers"
                ][0]["Image"]
            elif "CompilationJobName" in model_config:
                compilation_res = sage_client.describe_compilation_job(
                    CompilationJobName=model_config["CompilationJobName"]
                )
                self.model_data = compilation_res["ModelArtifacts"]["S3ModelArtifacts"]
                self.image_uri = compilation_res["InferenceImage"]

        instance_type = right_size_recommendation["EndpointConfiguration"]["InstanceType"]
        initial_instance_count = right_size_recommendation["EndpointConfiguration"][
            "InitialInstanceCount"
        ]

        return (instance_type, initial_instance_count)

    def _convert_to_endpoint_configurations_json(
        self, hyperparameter_ranges: List[Dict[str, CategoricalParameter]]
    ):
        """Bundle right_size() parameters into an endpoint configuration for Advanced job"""
        if not hyperparameter_ranges:
            return None

        endpoint_configurations_to_json = []
        for parameter_range in hyperparameter_ranges:
            if not parameter_range.get("instance_types"):
                raise ValueError("instance_type must be defined as a hyperparameter_range")
            parameter_range = parameter_range.copy()
            instance_types = parameter_range.get("instance_types").values
            parameter_range.pop("instance_types")

            for instance_type in instance_types:
                parameter_ranges = [
                    {"Name": name, "Value": param.values} for name, param in parameter_range.items()
                ]
                endpoint_configurations_to_json.append(
                    {
                        "EnvironmentParameterRanges": {
                            "CategoricalParameterRanges": parameter_ranges
                        },
                        "InstanceType": instance_type,
                    }
                )

        return endpoint_configurations_to_json

    def _convert_to_traffic_pattern_json(self, traffic_type: str, phases: List[Phase]):
        """Bundle right_size() parameters into a traffic pattern for Advanced job"""
        if not phases:
            return None
        return {
            "Phases": [phase.to_json for phase in phases],
            "TrafficType": traffic_type if traffic_type else "PHASES",
        }

    def _convert_to_resource_limit_json(self, max_tests: int, max_parallel_tests: int):
        """Bundle right_size() parameters into a resource limit for Advanced job"""
        if not max_tests and not max_parallel_tests:
            return None
        resource_limit = {}
        if max_tests:
            resource_limit["MaxNumberOfTests"] = max_tests
        if max_parallel_tests:
            resource_limit["MaxParallelOfTests"] = max_parallel_tests
        return resource_limit

    def _convert_to_stopping_conditions_json(
        self, max_invocations: int, model_latency_thresholds: List[ModelLatencyThreshold]
    ):
        """Bundle right_size() parameters into stopping conditions for Advanced job"""
        if not max_invocations and not model_latency_thresholds:
            return None
        stopping_conditions = {}
        if max_invocations:
            stopping_conditions["MaxInvocations"] = max_invocations
        if model_latency_thresholds:
            stopping_conditions["ModelLatencyThresholds"] = [
                threshold.to_json for threshold in model_latency_thresholds
            ]
        return stopping_conditions

    def _get_recommendation(self, sage_client, job_or_model_name, inference_recommendation_id):
        """Get recommendation from right size job and model"""
        right_size_recommendation, model_recommendation, right_size_job_res = None, None, None
        right_size_recommendation, right_size_job_res = self._get_right_size_recommendation(
            sage_client=sage_client,
            job_or_model_name=job_or_model_name,
            inference_recommendation_id=inference_recommendation_id,
        )
        if right_size_recommendation is None:
            model_recommendation = self._get_model_recommendation(
                sage_client=sage_client,
                job_or_model_name=job_or_model_name,
                inference_recommendation_id=inference_recommendation_id,
            )
            if model_recommendation is None:
                raise ValueError("inference_recommendation_id is not valid")

        return right_size_recommendation, model_recommendation, right_size_job_res

    def _get_right_size_recommendation(
        self,
        sage_client,
        job_or_model_name,
        inference_recommendation_id,
    ):
        """Get recommendation from right size job"""
        right_size_recommendation, right_size_job_res = None, None
        try:
            right_size_job_res = sage_client.describe_inference_recommendations_job(
                JobName=job_or_model_name
            )
            if right_size_job_res:
                right_size_recommendation = self._search_recommendation(
                    recommendation_list=right_size_job_res["InferenceRecommendations"],
                    inference_recommendation_id=inference_recommendation_id,
                )
        except sage_client.exceptions.ResourceNotFound:
            pass

        return right_size_recommendation, right_size_job_res

    def _get_model_recommendation(
        self,
        sage_client,
        job_or_model_name,
        inference_recommendation_id,
    ):
        """Get recommendation from model"""
        model_recommendation = None
        try:
            model_res = sage_client.describe_model(ModelName=job_or_model_name)
            if model_res:
                model_recommendation = self._search_recommendation(
                    recommendation_list=model_res["DeploymentRecommendation"][
                        "RealTimeInferenceRecommendations"
                    ],
                    inference_recommendation_id=inference_recommendation_id,
                )
        except sage_client.exceptions.ResourceNotFound:
            pass

        return model_recommendation

    def _search_recommendation(self, recommendation_list, inference_recommendation_id):
        """Search recommendation based on recommendation id"""
        return next(
            (
                rec
                for rec in recommendation_list
                if rec["RecommendationId"] == inference_recommendation_id
            ),
            None,
        )
