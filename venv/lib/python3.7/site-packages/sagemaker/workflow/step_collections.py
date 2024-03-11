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

import warnings
from typing import List, Union, Optional

import attr

from sagemaker.estimator import EstimatorBase
from sagemaker.model import Model
from sagemaker import PipelineModel
from sagemaker.predictor import Predictor
from sagemaker.transformer import Transformer
from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.steps import Step, CreateModelStep, TransformStep
from sagemaker.workflow._utils import _RegisterModelStep, _RepackModelStep
from sagemaker.workflow.retry import RetryPolicy
from sagemaker.utils import update_container_with_inference_params


@attr.s
class StepCollection:
    """A wrapper of pipeline steps for workflow.

    Attributes:
        name (str): The name of the `StepCollection`.
        steps (List[Step]): A list of steps.
    """

    name: str = attr.ib()
    steps: List[Step] = attr.ib(factory=list)

    def request_dicts(self) -> List[RequestType]:
        """Get the request structure for workflow service calls."""
        return [step.to_request() for step in self.steps]

    @property
    def properties(self):
        """The properties of the particular `StepCollection`."""
        if not self.steps:
            return None
        return self.steps[-1].properties


class RegisterModel(StepCollection):  # pragma: no cover
    """Register Model step collection for workflow."""

    _REGISTER_MODEL_NAME_BASE = "RegisterModel"
    _REPACK_MODEL_NAME_BASE = "RepackModel"

    def __init__(
        self,
        name: str,
        content_types,
        response_types,
        inference_instances=None,
        transform_instances=None,
        estimator: EstimatorBase = None,
        model_data=None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        repack_model_step_retry_policies: List[RetryPolicy] = None,
        register_model_step_retry_policies: List[RetryPolicy] = None,
        model_package_group_name=None,
        model_metrics=None,
        approval_status=None,
        image_uri=None,
        compile_model_family=None,
        display_name=None,
        description=None,
        tags=None,
        model: Union[Model, PipelineModel] = None,
        drift_check_baselines=None,
        customer_metadata_properties=None,
        domain=None,
        sample_payload_url=None,
        task=None,
        framework=None,
        framework_version=None,
        nearest_model_name=None,
        data_input_configuration=None,
        **kwargs,
    ):
        """Construct steps `_RepackModelStep` and `_RegisterModelStep` based on the estimator.

        Args:
            name (str): The name of the training step.
            estimator: The estimator instance.
            model_data: The S3 uri to the model data from training.
            content_types (list): The supported MIME types for the input data (default: None).
            response_types (list): The supported MIME types for the output data (default: None).
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time (default: None).
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed (default: None).
            depends_on (List[Union[str, Step, StepCollection]]): The list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that the first step
                in the collection depends on (default: None).
            repack_model_step_retry_policies (List[RetryPolicy]): The list of retry policies
                for the repack model step
            register_model_step_retry_policies (List[RetryPolicy]): The list of retry policies
                for register model step
            model_package_group_name (str): The Model Package Group name or Arn, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
                or "PendingManualApproval" (default: "PendingManualApproval").
            image_uri (str): The container image uri for Model Package, if not specified,
                Estimator's training container image is used (default: None).
            compile_model_family (str): The instance family for the compiled model. If
                specified, a compiled model is used (default: None).
            description (str): Model Package description (default: None).
            tags (List[dict[str, str]]): The list of tags to attach to the model package group. Note
                that tags will only be applied to newly created model package groups; if the
                name of an existing group is passed to "model_package_group_name",
                tags will not be applied.
            model (object or Model): A PipelineModel object that comprises a list of models
                which gets executed as a serial inference pipeline or a Model object.
            drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
            customer_metadata_properties (dict[str, str]): A dictionary of key-value paired
                metadata properties (default: None).
            domain (str): Domain values can be "COMPUTER_VISION", "NATURAL_LANGUAGE_PROCESSING",
                "MACHINE_LEARNING" (default: None).
            sample_payload_url (str): The S3 path where the sample payload is stored
                (default: None).
            task (str): Task values which are supported by Inference Recommender are "FILL_MASK",
                "IMAGE_CLASSIFICATION", "OBJECT_DETECTION", "TEXT_GENERATION", "IMAGE_SEGMENTATION",
                "CLASSIFICATION", "REGRESSION", "OTHER" (default: None).
            framework (str): Machine learning framework of the model package container image
                (default: None).
            framework_version (str): Framework version of the Model Package Container Image
                (default: None).
            nearest_model_name (str): Name of a pre-trained machine learning benchmarked by
                Amazon SageMaker Inference Recommender (default: None).
            data_input_configuration (str): Input object for the model (default: None).

            **kwargs: additional arguments to `create_model`.
        """
        self.name = name
        steps: List[Step] = []
        repack_model = False
        self.model_list = None
        self.container_def_list = None
        subnets = None
        security_group_ids = None

        if estimator is not None:
            subnets = estimator.subnets
            security_group_ids = estimator.security_group_ids
        elif model is not None and model.vpc_config is not None:
            subnets = model.vpc_config["Subnets"]
            security_group_ids = model.vpc_config["SecurityGroupIds"]

        if "entry_point" in kwargs:
            repack_model = True
            entry_point = kwargs.pop("entry_point", None)
            source_dir = kwargs.pop("source_dir", None)
            dependencies = kwargs.pop("dependencies", None)
            kwargs = dict(**kwargs, output_kms_key=kwargs.pop("model_kms_key", None))

            repack_model_step = _RepackModelStep(
                name="{}-{}".format(self.name, self._REPACK_MODEL_NAME_BASE),
                depends_on=depends_on,
                retry_policies=repack_model_step_retry_policies,
                sagemaker_session=estimator.sagemaker_session,
                role=estimator.role,
                model_data=model_data,
                entry_point=entry_point,
                source_dir=source_dir,
                dependencies=dependencies,
                tags=tags,
                subnets=subnets,
                security_group_ids=security_group_ids,
                description=description,
                display_name=display_name,
                **kwargs,
            )
            steps.append(repack_model_step)
            model_data = repack_model_step.properties.ModelArtifacts.S3ModelArtifacts

            # remove kwargs consumed by model repacking step
            kwargs.pop("output_kms_key", None)

        elif model is not None:
            if isinstance(model, PipelineModel):
                self.model_list = model.models
            elif isinstance(model, Model):
                self.model_list = [model]

            for model_entity in self.model_list:
                if estimator is not None:
                    sagemaker_session = estimator.sagemaker_session
                    role = estimator.role
                else:
                    sagemaker_session = model_entity.sagemaker_session
                    role = model_entity.role
                if hasattr(model_entity, "entry_point") and model_entity.entry_point is not None:
                    repack_model = True
                    entry_point = model_entity.entry_point
                    source_dir = model_entity.source_dir
                    dependencies = model_entity.dependencies
                    kwargs = dict(**kwargs, output_kms_key=model_entity.model_kms_key)
                    model_name = model_entity.name or model_entity._framework_name

                    repack_model_step = _RepackModelStep(
                        name="{}-{}".format(model_name, self._REPACK_MODEL_NAME_BASE),
                        depends_on=depends_on,
                        retry_policies=repack_model_step_retry_policies,
                        sagemaker_session=sagemaker_session,
                        role=role,
                        model_data=model_entity.model_data,
                        entry_point=entry_point,
                        source_dir=source_dir,
                        dependencies=dependencies,
                        tags=tags,
                        subnets=subnets,
                        security_group_ids=security_group_ids,
                        description=description,
                        display_name=display_name,
                        **kwargs,
                    )
                    steps.append(repack_model_step)
                    model_entity.model_data = (
                        repack_model_step.properties.ModelArtifacts.S3ModelArtifacts
                    )

                    # remove kwargs consumed by model repacking step
                    kwargs.pop("output_kms_key", None)

            if isinstance(model, PipelineModel):
                self.container_def_list = model.pipeline_container_def(
                    inference_instances[0] if inference_instances else None
                )
            elif isinstance(model, Model):
                self.container_def_list = [
                    model.prepare_container_def(
                        inference_instances[0] if inference_instances else None
                    )
                ]

            self.container_def_list = update_container_with_inference_params(
                framework=framework,
                framework_version=framework_version,
                nearest_model_name=nearest_model_name,
                data_input_configuration=data_input_configuration,
                container_list=self.container_def_list,
            )

        register_model_step = _RegisterModelStep(
            name="{}-{}".format(self.name, self._REGISTER_MODEL_NAME_BASE),
            estimator=estimator,
            model_data=model_data,
            content_types=content_types,
            response_types=response_types,
            inference_instances=inference_instances,
            transform_instances=transform_instances,
            model_package_group_name=model_package_group_name,
            model_metrics=model_metrics,
            drift_check_baselines=drift_check_baselines,
            approval_status=approval_status,
            image_uri=image_uri,
            compile_model_family=compile_model_family,
            description=description,
            display_name=display_name,
            tags=tags,
            container_def_list=self.container_def_list,
            retry_policies=register_model_step_retry_policies,
            customer_metadata_properties=customer_metadata_properties,
            domain=domain,
            sample_payload_url=sample_payload_url,
            task=task,
            **kwargs,
        )
        if not repack_model:
            register_model_step.add_depends_on(depends_on)

        steps.append(register_model_step)
        self.steps = steps

        warnings.warn(
            (
                "We are deprecating the use of RegisterModel. "
                "Please use the ModelStep instead. For more, see: "
                "https://sagemaker.readthedocs.io/en/stable/"
                "amazon_sagemaker_model_building_pipeline.html#model-step"
            ),
            DeprecationWarning,
        )


class EstimatorTransformer(StepCollection):
    """Creates a Transformer step collection for workflow."""

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        model_data,
        model_inputs,
        instance_count,
        instance_type,
        transform_inputs,
        description: str = None,
        display_name: str = None,
        # model arguments
        image_uri=None,
        predictor_cls=None,
        env=None,
        # transformer arguments
        strategy=None,
        assemble_with=None,
        output_path=None,
        output_kms_key=None,
        accept=None,
        max_concurrent_transforms=None,
        max_payload=None,
        tags=None,
        volume_kms_key=None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        # step retry policies
        repack_model_step_retry_policies: List[RetryPolicy] = None,
        model_step_retry_policies: List[RetryPolicy] = None,
        transform_step_retry_policies: List[RetryPolicy] = None,
        **kwargs,
    ):
        """Construct steps required for a Transformer step collection:

        An estimator-centric step collection. It models what happens in workflows
        when invoking the `transform()` method on an estimator instance:
        First, if custom
        model artifacts are required, a `_RepackModelStep` is included.
        Second, a
        `CreateModelStep` with the model data passed in from a training step or other
        training job output.
        Finally, a `TransformerStep`.

        If repacking
        the model artifacts is not necessary, only the CreateModelStep and TransformerStep
        are in the step collection.

        Args:
            name (str): The name of the Transform Step.
            estimator: The estimator instance.
            instance_count (int): The number of EC2 instances to use.
            instance_type (str): The type of EC2 instance to use.
            strategy (str): The strategy used to decide how to batch records in
                a single request (default: None). Valid values: 'MultiRecord'
                and 'SingleRecord'.
            assemble_with (str): How the output is assembled (default: None).
                Valid values: 'Line' or 'None'.
            output_path (str): The S3 location for saving the transform result. If
                not specified, results are stored to a default bucket.
            output_kms_key (str): Optional. A KMS key ID for encrypting the
                transform output (default: None).
            accept (str): The accept header passed by the client to
                the inference endpoint. If it is supported by the endpoint,
                it will be the format of the batch transform output.
            env (dict): The Environment variables to be set for use during the
                transform job (default: None).
            depends_on (List[Union[str, Step, StepCollection]]): The list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that the first step
                in the collection depends on (default: None).
            repack_model_step_retry_policies (List[RetryPolicy]): The list of retry policies
                for the repack model step
            model_step_retry_policies (List[RetryPolicy]): The list of retry policies for
                model step
            transform_step_retry_policies (List[RetryPolicy]): The list of retry policies for
                transform step
        """
        self.name = name
        steps = []
        if "entry_point" in kwargs:
            entry_point = kwargs.get("entry_point", None)
            source_dir = kwargs.get("source_dir", None)
            dependencies = kwargs.get("dependencies", None)
            repack_model_step = _RepackModelStep(
                name=f"{name}RepackModel",
                depends_on=depends_on,
                retry_policies=repack_model_step_retry_policies,
                sagemaker_session=estimator.sagemaker_session,
                role=estimator.role,
                model_data=model_data,
                entry_point=entry_point,
                source_dir=source_dir,
                dependencies=dependencies,
                tags=tags,
                subnets=estimator.subnets,
                security_group_ids=estimator.security_group_ids,
                description=description,
                display_name=display_name,
                output_kms_key=estimator.output_kms_key,
            )
            steps.append(repack_model_step)
            model_data = repack_model_step.properties.ModelArtifacts.S3ModelArtifacts

        def predict_wrapper(endpoint, session):
            return Predictor(endpoint, session)

        predictor_cls = predictor_cls or predict_wrapper

        model = Model(
            image_uri=image_uri or estimator.training_image_uri(),
            model_data=model_data,
            predictor_cls=predictor_cls,
            vpc_config=None,
            sagemaker_session=estimator.sagemaker_session,
            role=estimator.role,
            env=kwargs.get("env", None),
            name=kwargs.get("name", None),
            enable_network_isolation=kwargs.get("enable_network_isolation", None),
            model_kms_key=kwargs.get("model_kms_key", None),
            image_config=kwargs.get("image_config", None),
        )
        model_step = CreateModelStep(
            name=f"{name}CreateModelStep",
            model=model,
            inputs=model_inputs,
            description=description,
            display_name=display_name,
            retry_policies=model_step_retry_policies,
        )
        if "entry_point" not in kwargs and depends_on:
            # if the CreateModelStep is the first step in the collection
            model_step.add_depends_on(depends_on)
        steps.append(model_step)

        transformer = Transformer(
            model_name=model_step.properties.ModelName,
            instance_count=instance_count,
            instance_type=instance_type,
            strategy=strategy,
            assemble_with=assemble_with,
            output_path=output_path,
            output_kms_key=output_kms_key,
            accept=accept,
            max_concurrent_transforms=max_concurrent_transforms,
            max_payload=max_payload,
            env=env,
            tags=tags,
            base_transform_job_name=name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=estimator.sagemaker_session,
        )
        transform_step = TransformStep(
            name=f"{name}TransformStep",
            transformer=transformer,
            inputs=transform_inputs,
            description=description,
            display_name=display_name,
            retry_policies=transform_step_retry_policies,
        )
        steps.append(transform_step)

        self.steps = steps
