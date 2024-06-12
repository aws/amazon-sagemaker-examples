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
"""This module contains code related to model card operations"""
from __future__ import absolute_import, print_function

import json
import logging
from datetime import datetime
from typing import Optional, Union, List, Any
from botocore.exceptions import ClientError
from boto3.session import Session as boto3_Session
from six.moves.urllib.parse import urlparse
from sagemaker.session import Session

from sagemaker.model_card.schema_constraints import (
    ModelCardStatusEnum,
    RiskRatingEnum,
    ObjectiveFunctionEnum,
    FacetEnum,
    MetricTypeEnum,
    ModelPackageStatusEnum,
    ModelApprovalStatusEnum,
    ENVIRONMENT_CONTAINER_IMAGES_MAX_SIZE,
    MODEL_ARTIFACT_MAX_SIZE,
    METRIC_VALUE_TYPE_MAP,
    TRAINING_DATASETS_MAX_SIZE,
    TRAINING_METRICS_MAX_SIZE,
    USER_PROVIDED_TRAINING_METRICS_MAX_SIZE,
    HYPER_PARAMETERS_MAX_SIZE,
    USER_PROVIDED_HYPER_PARAMETERS_MAX_SIZE,
    EVALUATION_DATASETS_MAX_SIZE,
    SOURCE_ALGORITHMS_MAX_SIZE,
)
from sagemaker.model_card.helpers import (
    _OneOf,
    _IsList,
    _IsModelCardObject,
    _JSONEncoder,
    _DefaultToRequestDict,
    _DefaultFromDict,
    _SkipEncodingDecoding,
    _hash_content_str,
    _read_s3_json,
)
from sagemaker.model_card.evaluation_metric_parsers import (
    EvaluationMetricTypeEnum,
    EVALUATION_METRIC_PARSERS,
)
from sagemaker.utils import _start_waiting, unique_name_from_base


logger = logging.getLogger(__name__)


class Environment(_DefaultToRequestDict, _DefaultFromDict):
    """Training/inference environment."""

    container_image = _IsList(str, ENVIRONMENT_CONTAINER_IMAGES_MAX_SIZE)

    def __init__(self, container_image: List[str]):
        """Initialize an Environment object.

        Args:
            container_image (list[str]): A list of SageMaker training/inference image URIs. The maximum list length is 15.
        """  # noqa E501 # pylint: disable=line-too-long
        self.container_image = container_image


class ModelOverview(_DefaultToRequestDict, _DefaultFromDict):
    """An overview of the model."""

    model_artifact = _IsList(str, MODEL_ARTIFACT_MAX_SIZE)
    inference_environment = _IsModelCardObject(Environment)

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_description: Optional[str] = None,
        model_version: Optional[Union[int, float]] = None,
        problem_type: Optional[str] = None,
        algorithm_type: Optional[str] = None,
        model_creator: Optional[str] = None,
        model_owner: Optional[str] = None,
        model_artifact: Optional[List[str]] = None,
        inference_environment: Optional[Environment] = None,
    ):
        """Initialize a Model Overview object.

        Args:
            model_id (str, optional): A SageMaker Model ARN or non-SageMaker Model ID (default: None).
            model_name (str, optional): A unique name for the model (default: None).
            model_description (str, optional): A description of the model (default: None).
            model_version (int or float, optional): The version of the model (default: None).
            problem_type (str, optional): The type of problem that the model solves. For example, "Binary Classification", "Multiclass Classification", "Linear Regression", "Computer Vision", or "Natural Language Processing" (default: None).
            algorithm_type (str, optional): The algorithm used to solve the problem type (default: None).
            model_creator (str, optional): The organization, research group, or authors that created the model (default: None).
            model_owner (str, optional): The individual or group that maintains the model in your organization (default: None).
            model_artifact (List[str], optional): A list of model artifact location URIs. The maximum list size is 15. (default: None).
            inference_environment (Environment, optional): An overview of the model's inference environment (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.model_id = model_id
        self.model_name = model_name
        self.model_description = model_description
        self.model_version = model_version
        self.problem_type = problem_type
        self.algorithm_type = algorithm_type
        self.model_creator = model_creator
        self.model_owner = model_owner
        self.model_artifact = model_artifact
        self.inference_environment = inference_environment

    @classmethod
    def from_model_name(cls, model_name: str, sagemaker_session: Session = None, **kwargs):
        """Initialize a model overview object from auto-discovered data.

        Args:
            model_name (str): The unique name of the model.
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
            **kwargs: Other arguments in ModelOverview, i.e. model_description,
                problem_type, algorithm_type, model_creator, model_owner, model_version
        Raises:
            ValueError: A model with this name does not exist.
            ValueError: A model card already exists for this model.
        """

        def call_describe_model():
            """Load existing model."""
            try:
                model_response = sagemaker_session.sagemaker_client.describe_model(
                    ModelName=model_name
                )
            except ClientError as e:
                if e.response["Error"]["Message"].startswith(  # pylint: disable=r1720
                    "Could not find model"
                ):
                    raise ValueError(
                        (
                            f"Model details for model {model_name} could not be found. "
                            "Make sure the model name is valid."
                        )
                    )
                else:
                    raise
            return model_response

        def search_model_associated_model_cards(model_id: str):
            """Check if a model card already exists for this model.

            Args:
                model_id (str): A SageMaker model ID.
            """
            response = sagemaker_session.sagemaker_client.search(
                Resource="ModelCard",
                SearchExpression={
                    "Filters": [
                        {
                            "Name": "ModelId",
                            "Operator": "Equals",
                            "Value": model_id,
                        }
                    ]
                },
            )
            return [c["ModelCard"]["ModelCardName"] for c in response["Results"]]

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        model_response = call_describe_model()

        associated_model_cards = search_model_associated_model_cards(model_response["ModelArn"])
        if associated_model_cards:
            raise ValueError(
                f"The model has been associated with {associated_model_cards} model cards."
            )

        if "Containers" in model_response:  # inference pipeline model
            artifacts = [c["ModelDataUrl"] for c in model_response["Containers"]]
        elif (
            "PrimaryContainer" in model_response
            and "ModelDataUrl" in model_response["PrimaryContainer"]
        ):
            artifacts = [model_response["PrimaryContainer"]["ModelDataUrl"]]
        else:
            artifacts = []

        kwargs.update(
            {
                "model_name": model_name,
                "model_id": model_response["ModelArn"],
                "inference_environment": Environment(
                    container_image=[model_response["PrimaryContainer"]["Image"]]
                ),
                "model_artifact": artifacts,
            }
        )

        return cls(**kwargs)


class ModelPackageCreator(_DefaultToRequestDict, _DefaultFromDict):
    """Information about the user who created the model package"""

    def __init__(self, user_profile_name: Optional[str] = None):
        """Initialize the ModelPackageCreator object.

        Args:
            user_profile_name (str, optional): The name of the user's profile. (default: None)
        """  # noqa E501 # pylint: disable=line-too-long
        self.user_profile_name = user_profile_name


class SourceAlgorithm(_DefaultToRequestDict, _DefaultFromDict):
    """Source algorithm object"""

    def __init__(self, algorithm_name: str, model_data_url: Optional[str] = None):
        """Initialize source algorithm object

        Args:
            algorithm_name (str): The ARN of an algorithm resource that was used to create the model package.
            model_data_url (str, optional): The Amazon S3 path where the model artifacts, which result from model training, are stored (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.algorithm_name = algorithm_name
        self.model_data_url = model_data_url


class Container(_DefaultToRequestDict, _DefaultFromDict):
    """Inference container to store the information like model data urls, images and nearest model names"""  # noqa E501 # pylint: disable=line-too-long

    def __init__(
        self,
        image: str,
        model_data_url: Optional[str] = None,
        nearest_model_name: Optional[str] = None,
    ):
        """Initialize inference container object.

        Args:
            image (str): The Amazon EC2 Container Registry (Amazon ECR) path where inference code is stored. Also known as inference environment.
            model_data_url (str, optional): The Amazon S3 path where the model artifacts, which result from model training, are stored. Also known as model artifact (default: None).
            nearest_model_name (str, optional): The name of a pre-trained machine learning benchmarked by Amazon SageMaker Inference Recommender model that matches your model (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.image = image
        self.model_data_url = model_data_url
        self.nearest_model_name = nearest_model_name


class InferenceSpecification(_DefaultToRequestDict, _DefaultFromDict):
    """Details about inference jobs that can be run with models based on this model package."""

    containers = _IsList(Container, ENVIRONMENT_CONTAINER_IMAGES_MAX_SIZE)

    def __init__(self, containers: List[Container]):
        """Initialize Inference specification object.

        Args:
            containers (list[Containers]): The list of inference containers.
        """
        self.containers = containers


class ModelPackage(_DefaultToRequestDict, _DefaultFromDict):
    """Model package version details"""

    model_package_status = _OneOf(ModelPackageStatusEnum)
    model_approval_status = _OneOf(ModelApprovalStatusEnum)
    created_by = _IsModelCardObject(ModelPackageCreator)
    source_algorithms = _IsList(SourceAlgorithm, SOURCE_ALGORITHMS_MAX_SIZE)
    inference_specification = _IsModelCardObject(InferenceSpecification)
    model_metrics = _SkipEncodingDecoding(dict)

    MODEL_PACKAGE_OPTIONAL_FIELDS_MAPPING = {
        "ModelPackageName": "model_package_name",
        "ModelPackageGroupName": "model_package_group_name",
        "ModelPackageVersion": "model_package_version",
        "ModelPackageStatus": "model_package_status",
        "ModelApprovalStatus": "model_approval_status",
        "ModelPackageDescription": "model_package_description",
        "ApprovalDescription": "approval_description",
        "Domain": "domain",
        "Task": "task",
    }

    INFERENCE_SPECIFICATION_FIELDS_MAPPING = {
        "NearestModelName": "nearest_model_name",
        "ModelDataUrl": "model_data_url",
    }

    MODEL_CARD_CONTENT_TO_BE_INHERITED = [
        "business_details",
        "intended_uses",
        "additional_information",
    ]

    def __init__(
        self,
        model_package_arn: Optional[str] = None,
        model_package_description: Optional[str] = None,
        model_package_status: Optional[Union[ModelPackageStatusEnum, str]] = None,
        approval_description: Optional[str] = None,
        model_approval_status: Optional[Union[ModelApprovalStatusEnum, str]] = None,
        model_package_group_name: Optional[str] = None,
        model_package_name: Optional[str] = None,
        model_package_version: Optional[int] = None,
        domain: Optional[str] = None,
        task: Optional[str] = None,
        created_by: Optional[ModelPackageCreator] = None,
        source_algorithms: Optional[List[SourceAlgorithm]] = None,
        inference_specification: Optional[InferenceSpecification] = None,
        model_metrics: Optional[dict] = None,
    ):
        """Initialize the ModelPackage object.

        Args:
            model_package_arn (str, optional): Model package ARN (default: None).
            model_package_description (str, optional): A brief summary of the model package (default: None).
            model_package_status (ModelPackageStatusEnum or str, optional): Current status of model package (default: None).
            approval_description (str, optional): A description provided for the model approval (default: None).
            model_approval_status (ModelApprovalStatusEnum or str, optional): Current approval status of model package (default: None).
            model_package_group_name (str, optional): If the model is a versioned model, the name of the model group that the versioned model belongs to (default: None).
            model_package_name (str, optional): Name of the model package (default: None).
            model_package_version (int, optional): Version of the model package (default: None).
            domain (str, optional): The machine learning domain of the model package you specified. Common machine learning domains include computer vision and natural language processing (default: None).
            task (str, optional): The machine learning task you specified that your model package accomplishes. Common machine learning tasks include object detection and image classification (default: None).
            created_by (ModelPackageCreator, optional): Information about the user who created model package (default: None).
            source_algorithms (SourceAlgorithms, optional):A list of algorithms that were used to create a model package (default: None).
            inference_specification (InferenceSpecification, optional): Details about inference jobs that can be run with models based on this model package (default: None).
            model_metrics (dict, optional): Temporary field to store model metrics that will be usd to auto discover evaluation details (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.model_package_arn = model_package_arn
        self.model_package_description = model_package_description
        self.model_package_status = model_package_status
        self.model_approval_status = model_approval_status
        self.approval_description = approval_description
        self.model_package_group_name = model_package_group_name
        self.model_package_name = model_package_name
        self.model_package_version = model_package_version
        self.domain = domain
        self.task = task
        self.created_by = created_by
        self.source_algorithms = source_algorithms
        self.inference_specification = inference_specification
        self.model_metrics = model_metrics

    @staticmethod
    def call_describe_model_package(model_package_name: str, sagemaker_session: Session = None):
        """Load existing model package.

        Args:
            model_package_name (str): Model package ARN.
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.

        Raises:
            ValueError: A model package with this name or ARN is not valid or does not exist.
        """

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        try:
            model_package_response = sagemaker_session.sagemaker_client.describe_model_package(
                ModelPackageName=model_package_name
            )
        except ClientError as e:
            if e.response["Error"]["Message"].startswith(  # pylint: disable=r1720
                "Could not find model package"
            ):
                raise ValueError(
                    (
                        f"Model package details for {model_package_name} could not be found. "
                        "Make sure the model package name or ARN is valid."
                    )
                )

            raise e

        return model_package_response

    @staticmethod
    def search_model_package_associated_model_cards(
        model_package_arn: str, sagemaker_session: Session = None
    ):
        """Check if a model card already exists for this model package version.

        Args:
            model_package_arn (str): Model package version ARN.
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
        Raises:
            ValueError: The identity does not have permission to perform sagemaker search operation.
        """

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        try:
            response = sagemaker_session.sagemaker_client.search(
                Resource="ModelCard",
                SearchExpression={
                    "Filters": [
                        {
                            "Name": "ModelId",
                            "Operator": "Equals",
                            "Value": model_package_arn,
                        }
                    ]
                },
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                raise ValueError(
                    (
                        "Received AccessDeniedException while calling SageMaker Search operation "
                        "on resource ModelCard. This could mean the IAM role does not "
                        "have the resource permissions, in which case please add resource access "
                        "and retry. For cases where the role has tag based resource policy, "
                        "continuing to wait for tag propagation.."
                    )
                )
            raise e

        return [c["ModelCard"]["ModelCardName"] for c in response["Results"]]

    @classmethod
    def from_model_package_arn(cls, model_package_arn: str, sagemaker_session: Session = None):
        """Create model package details using the response from `describe_model_package` API

        Args:
            model_package_arn (str): The ARN of the model package version.
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
        Raises:
            ValueError: A model package with this name or ARN does not exist.
            ValueError: A model card already exists for this model package.
            ValueError: A model card already has a model associated with it.
        """

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        model_package_response = cls.call_describe_model_package(
            model_package_name=model_package_arn, sagemaker_session=sagemaker_session
        )

        associated_model_cards = cls.search_model_package_associated_model_cards(
            model_package_response["ModelPackageArn"],
            sagemaker_session=sagemaker_session,
        )
        if associated_model_cards:
            raise ValueError(
                f"The model package has already been associated with {associated_model_cards} model cards."  # noqa E501  # pylint: disable=c0301
            )

        # To store all the necessary metadata information from model package response
        model_package_details = {"model_package_arn": model_package_response["ModelPackageArn"]}

        for key, field_name in cls.MODEL_PACKAGE_OPTIONAL_FIELDS_MAPPING.items():
            if key in model_package_response:
                model_package_details.update({field_name: model_package_response[key]})

        if "UserProfileName" in model_package_response["CreatedBy"]:
            model_package_creator = ModelPackageCreator(
                user_profile_name=model_package_response["CreatedBy"]["UserProfileName"]
            )
            model_package_details.update({"created_by": model_package_creator})

        if "InferenceSpecification" in model_package_response:
            containers_response = model_package_response["InferenceSpecification"]["Containers"]
            containers = []
            for container in containers_response:
                args = {"image": container["Image"]}
                for (
                    field,
                    arg_key,
                ) in cls.INFERENCE_SPECIFICATION_FIELDS_MAPPING.items():
                    if field in container:
                        args[arg_key] = container[field]
                containers.append(Container(**args))

            model_package_details.update(
                {"inference_specification": InferenceSpecification(containers)}
            )

        if "SourceAlgorithmSpecification" in model_package_response:
            source_algorithms_response = model_package_response["SourceAlgorithmSpecification"][
                "SourceAlgorithms"
            ]
            source_algorithms = [
                SourceAlgorithm(sa["AlgorithmName"], sa["ModelDataUrl"])
                if "ModelDataUrl" in sa
                else SourceAlgorithm(sa["AlgorithmName"])
                for sa in source_algorithms_response
            ]

            model_package_details.update({"source_algorithms": source_algorithms})

        if "ModelMetrics" in model_package_response:
            model_package_details.update({"model_metrics": model_package_response["ModelMetrics"]})

        return cls(**model_package_details)

    def discover_training_details(self, sagemaker_session: Session = None):
        """Auto discover training job details using model package response

        Args:
            sagemaker_session (Session, optional): A SageMaker Session object, used for SageMaker
                interactions (default: None). If not specified, a SageMaker Session
                is created using the default AWS configuration chain.
        """

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        training_details = None
        if self.inference_specification:
            containers = self.inference_specification.containers
            model_artifacts = [c.model_data_url for c in containers if c.model_data_url is not None]
            training_details = TrainingDetails.from_model_s3_artifacts(
                model_artifacts=model_artifacts, sagemaker_session=sagemaker_session
            )
        else:
            logger.info(
                (
                    "TrainingJobDetails auto-discovery was unsuccessful. "
                    "No inference specification found for the given model package."
                    "Please create one from scratch with TrainingJobDetails "
                    "or use from_training_job_name() instead."
                )
            )

        return training_details

    def discover_evaluation_details(self, sagemaker_session: Session = None):
        """Auto discover evaluation details from model package model metrics field

        Args:
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
        """

        def get_evaluation_job(
            name: str, s3_uri: str, metric_type: _OneOf(EvaluationMetricTypeEnum), session: Session
        ):
            evaluation_job = EvaluationJob(name=name)
            evaluation_job.add_metric_group_from_s3(
                s3_url=s3_uri, session=session, metric_type=metric_type
            )

            return evaluation_job

        if not sagemaker_session:
            sagemaker_session = Session()

        evaluation_details = []

        if self.model_metrics:
            model_metrics = self.model_metrics

            if "Bias" in model_metrics:
                bias_report = model_metrics["Bias"]
                for args_key in bias_report:
                    evaluation_details.append(
                        get_evaluation_job(
                            name="Bias " + args_key,
                            s3_uri=bias_report[args_key]["S3Uri"],
                            metric_type=EvaluationMetricTypeEnum.CLARIFY_BIAS,
                            session=sagemaker_session.boto_session,
                        )
                    )

            if "Explainability" in model_metrics:
                exp_report = model_metrics["Explainability"]

                if "Report" in exp_report:
                    evaluation_details.append(
                        get_evaluation_job(
                            name="Explainability report",
                            s3_uri=exp_report["Report"]["S3Uri"],
                            metric_type=EvaluationMetricTypeEnum.CLARIFY_EXPLAINABILITY,
                            session=sagemaker_session.boto_session,
                        )
                    )

            if "ModelQuality" in model_metrics:
                model_quality_report = model_metrics["ModelQuality"]

                if "Statistics" in model_quality_report:
                    evaluation_details.append(
                        get_evaluation_job(
                            name="Model quality report",
                            s3_uri=model_quality_report["Statistics"]["S3Uri"],
                            metric_type=EvaluationMetricTypeEnum.MODEL_MONITOR_MODEL_QUALITY,
                            session=sagemaker_session.boto_session,
                        )
                    )
        else:
            logger.info(
                (
                    "Evaluation details auto-discovery was unsuccessful. "
                    "ModelMetrics was not found in the given model package. "
                    "Please create one from scratch with EvaluationJob."
                )
            )

        return evaluation_details

    def get_latest_model_card_name_from_model_package_group(
        self, session: Optional[Session] = None
    ):
        """Get the name of latest model card associated with model package.

        Args:
                session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
        """
        latest_model_card_name = None

        if self.model_package_group_name is None:
            return latest_model_card_name

        if not session:
            session = Session()  # pylint: disable=W0106

        response = session.sagemaker_client.search(
            Resource="ModelCard",
            SearchExpression={
                "Filters": [
                    {
                        "Name": "ModelPackageGroupName",
                        "Operator": "Equals",
                        "Value": self.model_package_group_name,
                    }
                ]
            },
            SortBy="LastModifiedTime",
            SortOrder="Descending",
        )

        if response and response.get("Results"):
            latest_model_card = response["Results"][0]
            if latest_model_card.get("ModelCard"):
                latest_model_card_name = latest_model_card["ModelCard"]["ModelCardName"]

        return latest_model_card_name


class IntendedUses(_DefaultToRequestDict, _DefaultFromDict):
    """The intended uses of a model."""

    risk_rating = _OneOf(RiskRatingEnum)

    def __init__(
        self,
        purpose_of_model: Optional[str] = None,
        intended_uses: Optional[str] = None,
        factors_affecting_model_efficiency: Optional[str] = None,
        risk_rating: Optional[Union[RiskRatingEnum, str]] = RiskRatingEnum.UNKNOWN,
        explanations_for_risk_rating: Optional[str] = None,
    ):
        """Initialize an Intended Uses object.

        Args:
            purpose_of_model (str, optional): The general purpose of this model (default: None).
            intended_uses (str, optional): The intended use cases for this model (default: None).
            factors_affecting_model_efficiency (str, optional): Factors affecting model efficacy (default: None).
            risk_rating (RiskRatingEnum or str, optional): Your organization's risk rating for this model. It is highly recommended to use sagemaker.model_card.RiskRatingEnum. Possible values include: ``RiskRatingEnum.HIGH`` ("High"), ``RiskRatingEnum.LOW`` ("Low"), ``RiskRatingEnum.MEDIUM`` ("Medium"), or ``RiskRatingEnum.UNKNOWN`` ("Unknown"). Defaults to ``RiskRatingEnum.UNKNOWN``.
            explanations_for_risk_rating (str, optional): An explanation of why your organization categorizes this model with this risk rating (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.purpose_of_model = purpose_of_model
        self.intended_uses = intended_uses
        self.factors_affecting_model_efficiency = factors_affecting_model_efficiency
        self.risk_rating = risk_rating
        self.explanations_for_risk_rating = explanations_for_risk_rating


class BusinessDetails(_DefaultToRequestDict, _DefaultFromDict):
    """The business details of a model."""

    def __init__(
        self,
        business_problem: Optional[str] = None,
        business_stakeholders: Optional[str] = None,
        line_of_business: Optional[str] = None,
    ):
        """Initialize an Business Details object.

        Args:
            business_problem (str, optional): The business problem of this model (default: None).
            business_stakeholders (str, optional): The business stakeholders for this model (default: None).
            line_of_business (str, optional): The line of business for this model (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.business_problem = business_problem
        self.business_stakeholders = business_stakeholders
        self.line_of_business = line_of_business


class Function(_DefaultToRequestDict, _DefaultFromDict):
    """Function details."""

    function = _OneOf(ObjectiveFunctionEnum)
    facet = _OneOf(FacetEnum)

    def __init__(
        self,
        function: Optional[Union[ObjectiveFunctionEnum, str]] = None,
        facet: Optional[Union[FacetEnum, str]] = None,
        condition: Optional[str] = None,
    ):
        """Initialize a Function object.

        Args:
            function (ObjectiveFunctionEnum or str, optional): The optimization direction of the model's objective function. It is highly recommended to use sagemaker.model_card.ObjectiveFunctionEnum. Possible values include: ``ObjectiveFunctionEnum.MAXIMIZE`` ("Maximize") or ``ObjectiveFunctionEnum.MINIMIZE`` ("Minimize") (default: None).
            facet (FacetEnum or str, optional): The metric of the model's objective function. For example, `loss` or `rmse`. It is highly recommended to use sagemaker.model_card.FacetEnum. Possible values include:, ``FacetEnum.ACCURACY`` ("Accuracy"), ``FacetEnum.AUC`` ("AUC"), ``FacetEnum.LOSS`` ("Loss"), ``FacetEnum.MAE`` ("MAE"), or ``FacetEnum.RMSE`` ("RMSE") (default: None).
            condition (str, optional): An optional description of any conditions of your objective function metric (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.function = function
        self.facet = facet
        self.condition = condition


class ObjectiveFunction(_DefaultToRequestDict, _DefaultFromDict):
    """The objective function that is optimized during model training."""

    function = _IsModelCardObject(Function)

    def __init__(
        self,
        function: Function,
        notes: Optional[str] = None,
    ):
        """Initialize an Objective Function object.

        Args:
            function (Function): A Function object that details optimization direction, metric, and additional descriptions.
            notes (str, optional): Notes about the objective function, including other considerations for possible objective functions (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.function = function
        self.notes = notes


class Metric(_DefaultToRequestDict, _DefaultFromDict):
    """Metric data."""

    type = _OneOf(MetricTypeEnum)

    def __init__(
        self,
        name: str,
        type: Union[MetricTypeEnum, str],  # pylint: disable=W0622
        value: Union[int, float, str, bool, List],
        notes: Optional[str] = None,
        x_axis_name: Optional[Union[str, list]] = None,
        y_axis_name: Optional[Union[str, list]] = None,
    ):
        """Initialize a Metric object.

        Args:
            name (str): The name of the metric.
            type (str or MetricTypeEnum): It is highly recommended to use sagemaker.model_card.MetricTypeEnum. Possible values include: ``MetricTypeEnum.BAR_CHART`` ("bar_char"), ``MetricTypeEnum.BOOLEAN`` ("boolean"), ``MetricTypeEnum.LINEAR_GRAPH`` ("linear_graph"), ``MetricTypeEnum.MATRIX`` ("matrix"), ``MetricTypeEnum.NUMBER`` ("number"), or ``MetricTypeEnum.STRING`` ("string").
            value (int or float or str or bool or List): The datatype of the metric. The metric's `value` must be compatible with the metric's `type`.
            notes (str, optional): Any notes to add to the metric (default: None).
            x_axis_name (str, optional): The name of the x axis (default: None).
            y_axis_name (str, optional): The name of the y axis (default: None).
        """  # noqa E501  # pylint: disable=line-too-long
        self.name = name
        self.type = type
        self.value = value
        self.notes = notes
        self.x_axis_name = x_axis_name
        self.y_axis_name = y_axis_name

    def __str__(self):
        """Return str(self)."""
        return f"Metric(name={self.name}, type={self.type}, value={self.value})"

    def __repr__(self):
        """Return repr(self)."""
        return self.__str__()

    @property
    def value(self):
        """Getter for value."""
        return self._value

    @value.setter
    def value(self, val: Union[int, float, str, bool, List]):
        """Setter for value.

        Args:
            val (int or float or str or bool): The metric value.
        Raises:
            ValueError: If the metric `type` doesn't match the metric `value`.
        """
        if type(val) not in METRIC_VALUE_TYPE_MAP[self.type]:
            raise ValueError(
                f"One of type {METRIC_VALUE_TYPE_MAP[self.type]} is expected for metric type {self.type}"
            )
        self._value = val


class TrainingMetric(_DefaultToRequestDict, _DefaultFromDict):
    """Training metric data.

    Should only be used during auto-population of training details.
    """

    def __init__(
        self,
        name: str,
        value: Union[int, float],
        notes: Optional[str] = None,
    ):
        """Initialize a TrainingMetric object.

        Args:
            name (str): The metric name.
            value (int or float): The metric value.
            notes (str, optional): Notes on the metric (default: None).
        """
        self.name = name
        self.value = value
        self.notes = notes


class HyperParameter(_DefaultToRequestDict, _DefaultFromDict):
    """Hyper-Parameters data."""

    def __init__(
        self,
        name: str,
        value: str,
    ):
        """Initialize a HyperParameter object.

        Args:
            name (str): The hyper parameter name.
            value (str): The hyper parameter value.
        """
        self.name = name
        self.value = value


class TrainingJobDetails(_DefaultToRequestDict, _DefaultFromDict):
    """The overview of a training job."""

    training_datasets = _IsList(str, TRAINING_DATASETS_MAX_SIZE)
    training_metrics = _IsList(TrainingMetric, TRAINING_METRICS_MAX_SIZE)
    user_provided_training_metrics = _IsList(
        TrainingMetric, USER_PROVIDED_TRAINING_METRICS_MAX_SIZE
    )
    hyper_parameters = _IsList(HyperParameter, HYPER_PARAMETERS_MAX_SIZE)
    user_provided_hyper_parameters = _IsList(
        HyperParameter, USER_PROVIDED_HYPER_PARAMETERS_MAX_SIZE
    )
    training_environment = _IsModelCardObject(Environment)

    def __init__(
        self,
        training_arn: Optional[str] = None,
        training_datasets: Optional[List[str]] = None,
        training_environment: Optional[Environment] = None,
        training_metrics: Optional[List[TrainingMetric]] = None,
        user_provided_training_metrics: Optional[List[TrainingMetric]] = None,
        hyper_parameters: Optional[List[HyperParameter]] = None,
        user_provided_hyper_parameters: Optional[List[HyperParameter]] = None,
    ):
        """Initialize a Training Job Details object.

        Args:
            training_arn (str, optional): The SageMaker training job Amazon Resource Name (ARN) (default: None).
            training_datasets (List[str], optional): The location of the datasets used to train the model. The maximum list size is 15. (default: None).
            training_environment (Environment, optional): The SageMaker training image URI. (default: None).
            training_metrics (list[TrainingMetric], optional): SageMaker training job results. The maximum `training_metrics` list length is 50 (default: None).
            user_provided_training_metrics (list[TrainingMetric], optional): Custom training job results. The maximum `user_provided_training_metrics` list length is 50 (default: None).
            hyper_parameters (list[HyperParameter], optional): SageMaker hyper parameter results. The maximum `hyper_parameters` list length is 100 (default: None).
            user_provided_hyper_parameters (list[HyperParameter], optional): Custom hyper parameter results. The maximum `user_provided_hyper_parameters` list length is 100 (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.training_arn = training_arn
        self.training_datasets = training_datasets
        self.training_environment = training_environment
        self.training_metrics = training_metrics
        self.user_provided_training_metrics = user_provided_training_metrics
        self.hyper_parameters = hyper_parameters
        self.user_provided_hyper_parameters = user_provided_hyper_parameters


class TrainingDetails(_DefaultToRequestDict, _DefaultFromDict):
    """The overview of model training."""

    objective_function = _IsModelCardObject(ObjectiveFunction)
    training_job_details = _IsModelCardObject(TrainingJobDetails)

    def __init__(
        self,
        objective_function: Optional[ObjectiveFunction] = None,
        training_observations: Optional[str] = None,
        training_job_details: Optional[TrainingJobDetails] = None,
    ):
        """Initialize a TrainingDetails object.

        Args:
            objective_function (ObjectiveFunction, optional): The objective function that is optimized during training (default: None).
            training_observations (str, optional): Any observations about training (default: None).
            training_job_details (TrainingJobDetails, optional): Details about any associated training jobs (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.objective_function = objective_function
        self.training_observations = training_observations
        self.training_job_details = training_job_details

    @staticmethod
    def _create_training_details(training_job_data: dict, cls: "TrainingDetails", **kwargs):
        """Create a Training Details object from the data queried from APIs.

        Args:
            training_job_data (dict): Training job data queried from either
                search or `describe_training_job`.
            cls (TrainingDetails): TrainingDetails class.
        """

        if training_job_data:
            job = {
                "training_arn": training_job_data["TrainingJobArn"],
                "training_environment": Environment(
                    container_image=[training_job_data["AlgorithmSpecification"]["TrainingImage"]]
                ),
                "training_metrics": [
                    TrainingMetric(i["MetricName"], i["Value"])
                    for i in training_job_data["FinalMetricDataList"]
                ]
                if "FinalMetricDataList" in training_job_data
                else [],
                "hyper_parameters": [
                    HyperParameter(key, value)
                    for key, value in training_job_data["HyperParameters"].items()
                ],
            }
            kwargs.update({"training_job_details": TrainingJobDetails(**job)})
            instance = cls(**kwargs)
        else:
            instance = None

        return instance

    @staticmethod
    def call_search_training_job(model_data_url: str, sagemaker_session: Session = None):
        """Search training job.

        Args:
            model_data_url (str): The Amazon S3 path where the model artifacts, which result from model training, are stored. It will be used to search training job.
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
        Raises:
            ValueError: The identity does not have permission to perform sagemaker search operation.
        """  # noqa E501 # pylint: disable=line-too-long

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        try:
            res = sagemaker_session.sagemaker_client.search(
                Resource="TrainingJob",
                SearchExpression={
                    "Filters": [
                        {
                            "Name": "ModelArtifacts.S3ModelArtifacts",
                            "Operator": "Equals",
                            "Value": model_data_url,
                        }
                    ]
                },
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                raise ValueError(
                    (
                        "Received AccessDeniedException while calling SageMaker Search operation "
                        "on resource TrainingJob. This could mean the IAM role does not "
                        "have the resource permissions, in which case please add resource access "
                        "and retry. For cases where the role has tag based resource policy, "
                        "continuing to wait for tag propagation.."
                    )
                )
            raise e

        training_job_data = None
        if len(res["Results"]) == 1:
            training_job_data = res["Results"][0]["TrainingJob"]

        return training_job_data

    @staticmethod
    def discover_training_job_details(
        model_artifacts: List[str], sagemaker_session: Session = None
    ):
        """Find training job details using auto discovered model artifacts

        Args:
            model_artifacts (List[str]): List of model artifacts with S3 uri.
            sagemaker_session (Session, optional): A SageMaker Session object, used for SageMaker
                interactions (default: None). If not specified, a SageMaker Session
                is created using the default AWS configuration chain.
        """  # noqa E501 # pylint: disable=line-too-long

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        training_job_details = None
        if len(model_artifacts) == 1:
            training_job_details = TrainingDetails.call_search_training_job(
                model_artifacts[0], sagemaker_session
            )
        elif len(model_artifacts) == 0:
            logger.warning(
                (
                    "TrainingJobDetails auto-discovery failed. "
                    "No associated training job. "
                    "Please create one from scratch with TrainingJobDetails "
                    "or use from_training_job_name() instead."
                )
            )
        else:
            logger.warning(
                (
                    "TrainingJobDetails auto-discovery failed. "
                    "There are %s associated training jobs. "
                    "Further clarification is required. "
                    "You could use TrainingDetails.training_job_name after "
                    "which training job to use is decided."
                ),
                len(model_artifacts),
            )

        return training_job_details

    @classmethod
    def from_model_overview(
        cls, model_overview: ModelOverview, sagemaker_session: Session = None, **kwargs
    ):
        """Initialize a Training Details object from an auto-discovered model overview.

        Args:
            model_overview (ModelOverview): A ModelOverview object.
            sagemaker_session (Session, optional): A SageMaker Session object, used for SageMaker interactions (default: None). If not specified, a SageMaker Session is created using the default AWS configuration chain.
            **kwargs: Other arguments in TrainingDetails, i.e. objective_function, training_observations
        """  # noqa E501 # pylint: disable=line-too-long

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        training_job_data = cls.discover_training_job_details(
            model_artifacts=model_overview.model_artifact, sagemaker_session=sagemaker_session
        )

        return cls._create_training_details(training_job_data=training_job_data, cls=cls, **kwargs)

    @classmethod
    def from_model_s3_artifacts(
        cls, model_artifacts: List[str], sagemaker_session: Session = None, **kwargs
    ):
        """Initialize a Training Details object from auto-discovered model artifacts.

        Args:
            model_artifacts (List[str]): List of model artifacts with S3 uri.
            sagemaker_session (Session, optional): A SageMaker Session object, used for SageMaker interactions (default: None). If not specified,
                a SageMaker Session is created using the default AWS configuration chain.
            **kwargs: Other arguments in TrainingDetails, i.e. objective_function, training_observations
        """  # noqa E501 # pylint: disable=line-too-long

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        training_job_data = cls.discover_training_job_details(
            model_artifacts=model_artifacts, sagemaker_session=sagemaker_session
        )

        return cls._create_training_details(training_job_data=training_job_data, cls=cls, **kwargs)

    @classmethod
    def from_training_job_name(
        cls, training_job_name: str, sagemaker_session: Session = None, **kwargs
    ):
        """Initialize a Training Details object from a training job name.

        Args:
            training_job_name (str): Training job name.
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
            **kwargs: Other arguments in TrainingDetails, i.e. objective_function,
                training_observations
        Raises:
            ValueError: No traininig job is found.
            ValueError: multiple training jobs are matched.
        """

        def call_describe_training_job():
            """Load existing training job."""
            try:
                training_job_data = sagemaker_session.sagemaker_client.describe_training_job(
                    TrainingJobName=training_job_name
                )
            except ClientError as e:
                if (  # pylint: disable=r1720
                    e.response["Error"]["Message"] == "Requested resource not found."
                ):
                    raise ValueError(
                        (
                            "Training job details could not be found. "
                            "Make sure the training job name is valid."
                        )
                    )
                else:
                    raise

            return training_job_data

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        training_job_data = call_describe_training_job()

        return cls._create_training_details(training_job_data=training_job_data, cls=cls, **kwargs)

    def add_metric(self, metric: TrainingMetric):
        """Add custom training metrics.

        Args:
            metric (TrainingMetric): The custom metric to add.
        """
        if not self.training_job_details:
            self.training_job_details = TrainingJobDetails()
        self.training_job_details.user_provided_training_metrics.append(metric)

    def add_parameter(self, parameter: HyperParameter):
        """Add custom hyper-parameter.

        Args:
            parameter (HyperParameter): The custom parameter to add.
        """
        if not self.training_job_details:
            self.training_job_details = TrainingJobDetails()
        self.training_job_details.user_provided_hyper_parameters.append(parameter)


class MetricGroup(_DefaultToRequestDict, _DefaultFromDict):
    """Group of metric data"""

    metric_data = _IsList(Metric)

    def __init__(self, name: str, metric_data: Optional[List[Metric]] = None):
        """Initialize a Metric Group object.

        Args:
            name (str): The metric group name.
            metric_data (List[Metric]): A list of Metric objects.
        """
        self.name = name
        self.metric_data = metric_data

    def __str__(self):
        """Return str(self)."""
        metric_size = 0 if self.metric_data is None else len(self.metric_data)
        return f"MetricGroup(name={self.name}, size={metric_size})"

    def __repr__(self):
        """Return repr(self)."""
        return self.__str__()

    def add_metric(self, metric: Metric):
        """Add metric to the metric group.

        Args:
            metric (Metric): The Metric object to add.
        """
        self.metric_data.append(metric)


class EvaluationJob(_DefaultToRequestDict, _DefaultFromDict):
    """Overview of an evaluation job."""

    datasets = _IsList(str, EVALUATION_DATASETS_MAX_SIZE)
    metric_groups = _IsList(MetricGroup)

    def __init__(
        self,
        name: str,
        evaluation_observation: Optional[str] = None,
        evaluation_job_arn: Optional[str] = None,
        datasets: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        metric_groups: Optional[List[MetricGroup]] = None,
    ):
        """Initialize an Evaluation Job object.

        Args:
            name (str): The evaluation job name.
            evaluation_observation (str, optional): Any observations made during model evaluation (default: None).
            evaluation_job_arn (str, optional): The Amazon Resource Name (ARN) of the evaluation job (default: None).
            datasets (List[str], optional): Evaluation dataset locations. Maximum list length is 10 (default: None).
            metadata (Optional[dict], optional): Additional attributes associated with the evaluation results (default: None).
            metric_groups (List[MetricGroup], optional): An evaluation Metric Group object (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.name = name
        self.evaluation_observation = evaluation_observation
        self.evaluation_job_arn = evaluation_job_arn
        self.datasets = datasets
        self.metadata = metadata
        self.metric_groups = metric_groups

    def get_metric_group(self, group_name):
        """Get a metric group.

        Args:
            group_name (str): The metric group name.
        """
        return self.metric_groups.to_map(key_attribute="name").get(group_name)

    def add_metric_group(self, group_name: str):
        """Add a metric group

        Args:
            group_name (str): The metric group name.
        """
        metric_group = MetricGroup(name=group_name)
        self.metric_groups.append(metric_group)
        return metric_group

    def _add_metric_groups_data(self, metric_groups: List):
        """Encode the model card evaluation metrics to Metric and MetricGroup.

        Args:
            metric_groups (List): List of metric groups raw data.
        """
        for group_data in metric_groups:
            group = self.add_metric_group(group_data["name"])
            for item in group_data["metric_data"]:
                group.add_metric(Metric(**item))

    def _parse_evaluation_metric_json(self, json_data: dict, metric_type: EvaluationMetricTypeEnum):
        """Parse the evaluation metric data to the model card metric schema.

        Args:
            json_data (dict): Metric JSON data.
            metric_type EvaluationMetricTypeEnum: The evaluation metric type
                for the data in the evaluation metrics JSON file.
                Possible values include: ``EvaluationMetricTypeEnum.MODEL_CARD_METRIC_SCHEMA``, ``EvaluationMetricTypeEnum.CLARIFY_BIAS``,
                ``EvaluationMetricTypeEnum.CLARIFY_EXPLAINABILITY``, ``EvaluationMetricTypeEnum.MODEL_MONITOR_MODEL_QUALITY``, ``EvaluationMetricTypeEnum.REGRESSION``,
                ``EvaluationMetricTypeEnum.BINARY_CLASSIFICATION``, or ``EvaluationMetricTypeEnum.MULTICLASS_CLASSIFICATION``.
        """  # noqa E501 # pylint: disable=line-too-long
        if not isinstance(metric_type, EvaluationMetricTypeEnum):
            raise ValueError("Please use sagemaker.model_card.EvaluationMetricTypeEnum")

        parser = EVALUATION_METRIC_PARSERS[metric_type]
        return parser.run(json_data)

    def add_metric_group_from_json(
        self,
        json_path: str,
        metric_type: EvaluationMetricTypeEnum = EvaluationMetricTypeEnum.MODEL_CARD_METRIC_SCHEMA,
    ):
        """Add evaluation metric files from S3 bucket.

        Args:
            json_path (str): The path for the evaluation metrics JSON file.
            metric_type (EvaluationMetricTypeEnum, optional): The evaluation metric type
                for the data in the evaluation metrics JSON file.
                Possible values include:
                ``EvaluationMetricTypeEnum.MODEL_CARD_METRIC_SCHEMA``,
                ``EvaluationMetricTypeEnum.CLARIFY_BIAS``,
                ``EvaluationMetricTypeEnum.CLARIFY_EXPLAINABILITY``,
                ``EvaluationMetricTypeEnum.MODEL_MONITOR_MODEL_QUALITY``,
                ``EvaluationMetricTypeEnum.REGRESSION``,
                ``EvaluationMetricTypeEnum.BINARY_CLASSIFICATION``,
                or `EvaluationMetricTypeEnum.MULTICLASS_CLASSIFICATION``
                (default: ``EvaluationMetricTypeEnum.MODEL_CARD_METRIC_SCHEMA``).
        """
        with open(json_path, "r", encoding="utf-8") as istr:
            json_data = json.load(istr)

        result = self._parse_evaluation_metric_json(json_data, metric_type)
        self._add_metric_groups_data(result["metric_groups"])

    def add_metric_group_from_s3(
        self,
        session: boto3_Session,
        s3_url: str,
        metric_type: EvaluationMetricTypeEnum = EvaluationMetricTypeEnum.MODEL_CARD_METRIC_SCHEMA,
    ):
        """Add evaluation metric files from an S3 bucket.

        Args:
            session (Session): A Boto3 session.
            s3_url (str): The S3 URL for the evaluation metrics JSON file.
            metric_type (EvaluationMetricTypeEnum, optional): The evaluation metric type
                for the data in the evaluation metrics JSON file.
                Possible values include:
                ``EvaluationMetricTypeEnum.MODEL_CARD_METRIC_SCHEMA``,
                ``EvaluationMetricTypeEnum.CLARIFY_BIAS``,
                ``EvaluationMetricTypeEnum.CLARIFY_EXPLAINABILITY``,
                ``EvaluationMetricTypeEnum.MODEL_MONITOR_MODEL_QUALITY``,
                ``EvaluationMetricTypeEnum.REGRESSION``,
                ``EvaluationMetricTypeEnum.BINARY_CLASSIFICATION``,
                or ``EvaluationMetricTypeEnum.MULTICLASS_CLASSIFICATION``
                (default: ``EvaluationMetricTypeEnum.MODEL_CARD_METRIC_SCHEMA``).
        """
        parsed_url = urlparse(s3_url)
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip("/")
        json_data = _read_s3_json(session, bucket, key)

        result = self._parse_evaluation_metric_json(json_data, metric_type)
        self._add_metric_groups_data(result["metric_groups"])


class AdditionalInformation(_DefaultToRequestDict, _DefaultFromDict):
    """Additional information for a model card."""

    def __init__(
        self,
        ethical_considerations: Optional[str] = None,
        caveats_and_recommendations: Optional[str] = None,
        custom_details: Optional[dict] = None,
    ):
        """Initialize an Additional Information object.

        Args:
            ethical_considerations (str, optional): Any ethical considerations to document about the model (default: None).
            caveats_and_recommendations (str, optional): Caveats and recommendations for those who might use this model in their applications (default: None).
            custom_details (dict, optional): Any additional custom information to document about the model (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.ethical_considerations = ethical_considerations
        self.caveats_and_recommendations = caveats_and_recommendations
        self.custom_details = custom_details


class ModelCard(object):
    """Use an Amazon SageMaker Model Card to document qualitative and quantitative information about a model."""  # noqa E501  # pylint: disable=c0301

    DECODER_ATTRIBUTE_MAP = {
        "ModelCardName": "name",
        "ModelCardArn": "arn",
        "ModelCardStatus": "status",
        "ModelCardVersion": "version",
        "CreationTime": "created_time",
        "CreatedBy": "created_by",
        "LastModifiedTime": "last_modified_time",
        "LastModifiedBy": "last_modified_by",
    }
    CREATE_MODEL_CARD_REQUIRED = ["ModelCardName", "ModelCardStatus"]
    INITIAL_VERSION = 1

    status = _OneOf(ModelCardStatusEnum)
    model_overview = _IsModelCardObject(ModelOverview)
    intended_uses = _IsModelCardObject(IntendedUses)
    business_details = _IsModelCardObject(BusinessDetails)
    training_details = _IsModelCardObject(TrainingDetails)
    evaluation_details = _IsList(EvaluationJob)
    additional_information = _IsModelCardObject(AdditionalInformation)
    _model_package_details = _IsModelCardObject(ModelPackage)

    def __init__(
        self,
        name: str,
        status: Optional[Union[ModelCardStatusEnum, str]] = ModelCardStatusEnum.DRAFT,
        arn: Optional[str] = None,
        version: Optional[int] = None,
        created_time: Optional[datetime] = None,
        created_by: Optional[dict] = None,
        last_modified_time: Optional[datetime] = None,
        last_modified_by: Optional[dict] = None,
        model_overview: Optional[ModelOverview] = None,
        intended_uses: Optional[IntendedUses] = None,
        business_details: Optional[BusinessDetails] = None,
        training_details: Optional[TrainingDetails] = None,
        evaluation_details: Optional[List[EvaluationJob]] = None,
        additional_information: Optional[AdditionalInformation] = None,
        sagemaker_session: Optional[Session] = None,
        model_package_details: Optional[ModelPackage] = None,
    ):
        """Initialize an Amazon SageMaker Model Card.

        Args:
            name (str): The unique name of the model card.
            status (ModelCardStatusEnum or str, optional): Your organization's approval status of the model card. It is highly recommended to use sagemaker.model_card.ModelCardStatusEnum. Possible values include: ``ModelCardStatusEnum.APPROVED`` ("Approved"), ``ModelCardStatusEnum.ARCHIVED`` ("Archived"), ``ModelCardStatusEnum.DRAFT`` ("Draft"), or ``ModelCardStatusEnum.PENDING_REVIEW`` ("PendingReview"). Defaults to ``ModelCardStatusEnum.DRAFT``.
            arn (str, optional): The Amazon Resource Name (ARN) of the model card (default: None).
            version (int, optional): The model card version (default: None).
            created_time (datetime, optional): The date/time that you created the model card (default: None).
            created_by (dict, optional): The group or individual that created the model card (default: None).
            last_modified_time (datetime, optional): The last time that the model card was modified (default: None).
            last_modified_by (dict, optional): The group or individual that last modified the model card (default: None).
            model_overview (ModelOverview, optional): An overview of the model (default: None).
            intended_uses (IntendedUses, optional): The intended uses of the model (default: None).
            business_details (BusinessDetails, optional): The business details of the model (default: None).
            training_details (TrainingDetails, optional): The training details of the model (default: None).
            evaluation_details (List[EvaluationJob], optional): The evaluation details of the model (default: None).
            additional_information (AdditionalInformation, optional): Additional information about the model (default: None).
            sagemaker_session (Session, optional): A SageMaker Session object, used for SageMaker interactions (default: None). If not specified, a SageMaker Session is created using the default AWS configuration chain.
            model_package_details (ModelPackage, optional): Model package version metadata information (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.sagemaker_session = sagemaker_session or Session()
        self.name = name
        self.arn = arn
        self.status = status
        self.version = version
        self.created_time = created_time
        self.created_by = created_by
        self.last_modified_time = last_modified_time
        self.last_modified_by = last_modified_by
        self.model_overview = model_overview
        self.intended_uses = intended_uses
        self.business_details = business_details
        self.training_details = training_details
        self.evaluation_details = evaluation_details
        self.additional_information = additional_information
        self.model_package_details = model_package_details

    @property
    def model_package_details(self):
        """Model package auto discovered information"""
        return self._model_package_details

    @model_package_details.setter
    def model_package_details(self, value):
        """Setter method for model_package_details.

        When this is set, it should call _from_model_package private method for auto discovery of training, evaluation details and additional information.
        Raises:
            ValueError: The model card has already been associated with a different model entity.
        """  # noqa E501  # pylint: disable=c0301
        if value is not None:
            # Check if model card already has a model id associated with it
            if self.model_overview is not None and self.model_overview.model_id:
                raise ValueError(
                    f"The model card has already been associated with a model with model Id {self.model_overview.model_id}"  # noqa E501  # pylint: disable=c0301
                )

        self._model_package_details = value
        self._from_model_package()

    def _from_model_package(self):
        """Auto discover training details, evaluation details and carry over information"""
        if self.model_package_details:
            # auto discover training details using information from model package
            if self.training_details is None:
                self.training_details = self.model_package_details.discover_training_details(
                    self.sagemaker_session
                )
            elif (
                self.training_details is not None
                and self.training_details.training_job_details is None
            ):
                training_details = self.model_package_details.discover_training_details(
                    self.sagemaker_session
                )
                self.training_details.training_job_details = training_details.training_job_details
            else:
                logger.info(
                    (
                        "Skipping training details auto discovery. "
                        "Training details already exists for this model card."
                    )
                )

            # auto discover evaluation details using information from model package
            if not self.evaluation_details:
                self.evaluation_details = self.model_package_details.discover_evaluation_details(
                    self.sagemaker_session
                )

            # get the name of latest model card from model package group
            latest_model_card_name = (
                self.model_package_details.get_latest_model_card_name_from_model_package_group(
                    self.sagemaker_session
                )
            )
            # if found latest model card, carry over additional content to current model card
            if latest_model_card_name is not None:
                response = self.sagemaker_session.sagemaker_client.describe_model_card(
                    ModelCardName=latest_model_card_name
                )
                self._inherit_from_other_model_card(
                    json.loads(response["Content"]), ModelPackage.MODEL_CARD_CONTENT_TO_BE_INHERITED
                )

    def create(self):
        """Create the model card"""

        if self.arn:  # model card has been created
            raise FileExistsError(f"Model card {self.name} already exists with arn {self.arn}")

        request_args = self._create_request_args()
        logger.info("Creating model card with name: %s", self.name)
        logger.debug("CreateModelCard request: %s", json.dumps(request_args, indent=4))
        self.sagemaker_session.sagemaker_client.create_model_card(**request_args)

        # udpate model card with the latest data from server
        response = self.sagemaker_session.sagemaker_client.describe_model_card(
            ModelCardName=self.name
        )
        self.created_time = response["CreationTime"]
        self.last_modified_time = response["LastModifiedTime"]
        self.version = response["ModelCardVersion"]
        self.arn = response["ModelCardArn"]

        return self.arn

    def _inherit_from_other_model_card(  # pylint: disable=redefined-builtin
        self, other_content: dict, keys: List[str]
    ):
        """Update additonal content based on latest model card when the field is empty.

        Args:
            other_content (dict): The content of the other model card.
            keys (list[str]): The list of keys.
        """  # noqa E501 # pylint: disable=line-too-long

        for key in keys:
            if getattr(self, key) is None and key in other_content:
                setattr(self, key, other_content[key])

    def _create_request_args(self):
        """Generate the request body for create model card call."""

        def get_value_type(attr_val: Any, attr_name: str):
            """Get the attribute value type"""
            if isinstance(attr_val, property):
                return type(self.__class__.__dict__[f"_{attr_name}"])

            return type(attr_val)

        request_args = {}
        for arg in ModelCard.CREATE_MODEL_CARD_REQUIRED:
            request_args[arg] = getattr(self, ModelCard.DECODER_ATTRIBUTE_MAP[arg])

        content = {}
        for attr_name, attr_val in ModelCard.__dict__.items():
            if get_value_type(attr_val, attr_name) in [
                _IsModelCardObject,
                _IsList,
            ] and not attr_name.startswith("_"):
                object_val = getattr(self, attr_name)
                if object_val:
                    content[attr_name] = object_val

        request_args["Content"] = json.dumps(content, cls=_JSONEncoder)

        return request_args

    @classmethod
    def load(cls, name: str, version: Optional[int] = None, sagemaker_session: Session = None):
        """Load a model card.

        Args:
            name (str): The unique name of the model card.
            version (int, optional): The model card version (default: None).
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
        """

        def decode_attributes(response: dict):
            """Decode attributes from API response to class attributes.

            Args:
                response (dict): describe_model_card response.
            """
            decoded = {}
            for var, attr in cls.DECODER_ATTRIBUTE_MAP.items():
                if var in response:
                    decoded[attr] = response[var]

            content = json.loads(response["Content"])
            decoded = {**decoded, **content}

            return decoded

        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        logger.info("Load model card: %s", name)
        request_args = {"ModelCardName": name}
        if version:
            request_args["ModelCardVersion"] = version
        response = sagemaker_session.sagemaker_client.describe_model_card(**request_args)

        model_card_args = {
            "sagemaker_session": sagemaker_session,
            **decode_attributes(response),
        }

        return cls(**model_card_args)

    def update(self, **kwargs):
        """Update the model card.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        # use created_time to infer if the model card
        if self.created_time is None:
            raise ValueError(
                "Please create a model card or load an existing model card before update."
            )

        logger.info("Updating model card: %s", self.name)

        # update the current attributes if any
        for attribute, value in kwargs.items():
            if hasattr(self, attribute):
                setattr(self, attribute, value)
            else:
                logger.warning("%s doesn't exist in model card.", attribute)

        # Get the latest model card on the server
        previous = self.sagemaker_session.sagemaker_client.describe_model_card(
            ModelCardName=self.name
        )
        previous_content_hash = _hash_content_str(previous["Content"])

        current = self._create_request_args()
        logger.debug("UpdateModelCard request: %s", json.dumps(current, indent=4))

        result = {}
        if previous["ModelCardStatus"] != current["ModelCardStatus"]:
            logger.info("Update model card status")
            update_status_response = self.sagemaker_session.sagemaker_client.update_model_card(
                ModelCardName=current["ModelCardName"],
                ModelCardStatus=current["ModelCardStatus"],
            )
            result["status"] = update_status_response

        current_content_hash = _hash_content_str(current["Content"])
        if previous_content_hash != current_content_hash:
            logger.info("Update model card content")
            update_content_response = self.sagemaker_session.sagemaker_client.update_model_card(
                ModelCardName=current["ModelCardName"], Content=current["Content"]
            )
            result["content"] = update_content_response

        return result

    def delete(self):
        """Delete the model card"""
        logger.info("Deleting model card with name: %s", self.name)
        self.sagemaker_session.sagemaker_client.delete_model_card(ModelCardName=self.name)
        self.status = ModelCardStatusEnum.ARCHIVED

        return True

    def export_pdf(
        self,
        s3_output_path: str,
        export_job_name: Optional[str] = None,
        model_card_version: Optional[int] = None,
    ):
        """Export the model card as a PDF.

        Args:
            s3_output_path (str): The S3 output path for your model card PDF.
            export_job_name (str, optional): The model card export job name.
                (default: None).
            model_card_version (int, optional): The version of the model card you want to export.
                (default: None).
        """
        if not model_card_version:
            model_card_version = self.version

        if not export_job_name:
            export_job_name = unique_name_from_base(self.name)

        job = ModelCardExportJob(
            export_job_name=export_job_name,
            model_card_name=self.name,
            model_card_version=model_card_version,
            sagemaker_session=self.sagemaker_session,
            s3_output_path=s3_output_path,
        )
        output_path = job.create()
        return output_path

    def list_export_jobs(self, **kwargs):
        """List all the model card export jobs for a specific model card.

        Args:
            kwargs: additional parameters from
                sagemaker_session.sagemaker_client.list_model_card_export_jobs.
        """
        return ModelCardExportJob.list_export_jobs(
            model_card_name=self.name,
            sagemaker_session=self.sagemaker_session,
            **kwargs,
        )

    def get_version_history(self, **kwargs):
        """Get the version history of the model card.

        Args:
            kwargs: additional parameters from
                sagemaker_session.sagemaker_client.list_model_card_versions.
        """
        response = self.sagemaker_session.sagemaker_client.list_model_card_versions(
            ModelCardName=self.name, **kwargs
        )

        return response["ModelCardVersionSummaryList"]


class ModelCardExportJob(object):
    """Model card export job class."""

    EXPORT_JOB_POLLING_FREQUENCY = 4  # seconds

    def __init__(
        self,
        export_job_name: str,
        model_card_name: str,
        model_card_version: int,
        s3_output_path: str,
        s3_export_artifacts: Optional[str] = None,
        export_job_arn: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        status: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ):
        """Initialize a model card export job instance.

        Args:
            export_job_name (str): The model card export job name.
            model_card_name (str): The unique name of the model card you want to export.
            model_card_version (int): The version of the model card you want to export.
            s3_output_path (str): The S3 output path for your model card PDF.
            s3_export_artifacts (str, optional): The full S3 URI for your model card PDF
                (default: None).
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
            status (str, optional): The status of the model card export job.
                (default: None).
            export_job_arn: (str, optional): The Amazon Resource Name (ARN) of the model card export job.
                (default: None).
            failure_reason: (str, optional): The failure reason if your model card fails to export.
                (default: None).
        """  # noqa E501 # pylint: disable=line-too-long
        self.export_job_name = export_job_name
        self.model_card_name = model_card_name
        self.model_card_version = model_card_version
        self.s3_output_path = s3_output_path
        self.s3_export_artifacts = s3_export_artifacts
        self.sagemaker_session = sagemaker_session or Session()
        self.export_job_arn = export_job_arn
        self.status = status
        self.failure_reason = failure_reason

    def create(self):
        """Create a model card export job."""
        output_config = {"S3OutputPath": self.s3_output_path}
        response = self.sagemaker_session.sagemaker_client.create_model_card_export_job(
            ModelCardName=self.model_card_name,
            ModelCardVersion=self.model_card_version,
            ModelCardExportJobName=self.export_job_name,
            OutputConfig=output_config,
        )
        self.export_job_arn = response["ModelCardExportJobArn"]

        # Wait for the job to finish
        job = ModelCardExportJob.load(self.export_job_arn, self.sagemaker_session)
        while job.status == "InProgress":
            job = ModelCardExportJob.load(self.export_job_arn, self.sagemaker_session)
            _start_waiting(self.EXPORT_JOB_POLLING_FREQUENCY)

        if job.status == "Failed":
            logger.warning(
                "Failed to export model card to %s. %s",
                job.s3_export_artifacts,
                job.failure_reason,
            )
            output = None
        elif job.status == "Completed":
            logger.info(
                "Model card %s is successfully exported to %s.",
                job.model_card_name,
                job.s3_export_artifacts,
            )
            output = job.s3_export_artifacts

        return output

    @classmethod
    def load(cls, export_job_arn: str, sagemaker_session: Session = None):
        """Load the model card export job.

        Args:
            export_job_arn (str): The Amazon Resource Name (ARN) of the export job.
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
        """
        if not sagemaker_session:
            sagemaker_session == Session()  # pylint: disable=W0106

        response = sagemaker_session.sagemaker_client.describe_model_card_export_job(
            ModelCardExportJobArn=export_job_arn
        )

        key_args = {
            "export_job_name": response["ModelCardExportJobName"],
            "model_card_name": response["ModelCardName"],
            "model_card_version": response["ModelCardVersion"],
            "sagemaker_session": sagemaker_session,
            "status": response["Status"],
            "export_job_arn": response["ModelCardExportJobArn"],
            "s3_output_path": response["OutputConfig"]["S3OutputPath"],
        }

        if "ExportArtifacts" in response:
            key_args["s3_export_artifacts"] = response["ExportArtifacts"]["S3ExportArtifacts"]
        if response["Status"] == "Failed" and "FailureReason" in response:
            key_args["failure_reason"] = response["FailureReason"]

        return cls(**key_args)

    @staticmethod
    def list_export_jobs(
        model_card_name: str, sagemaker_session: Optional[Session] = None, **kwargs
    ):
        """List all model card export jobs for a specific model card.

        Args:
            model_card_name (str): The unique name of the model card.
            sagemaker_session (Session, optional): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, a SageMaker Session is created using the default AWS configuration
                chain.
            kwargs: additional parameters from
                sagemaker_session.sagemaker_client.list_model_card_export_jobs.
        """
        if not sagemaker_session:
            sagemaker_session = Session()  # pylint: disable=W0106

        return sagemaker_session.sagemaker_client.list_model_card_export_jobs(
            ModelCardName=model_card_name, **kwargs
        )
