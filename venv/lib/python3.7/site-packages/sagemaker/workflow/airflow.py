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
from __future__ import print_function, absolute_import

import os
import re

import sagemaker
from sagemaker import fw_utils, job, utils, s3, session, vpc_utils
from sagemaker.amazon import amazon_estimator
from sagemaker.tensorflow import TensorFlow
from sagemaker.estimator import EstimatorBase
from sagemaker.processing import Processor


def prepare_framework(estimator, s3_operations):
    """Prepare S3 operations and environment variables related to framework.

    S3 operations specify where to upload `source_dir`.

    Args:
        estimator (sagemaker.estimator.Estimator): The framework estimator to
            get information from and update.
        s3_operations (dict): The dict to specify s3 operations (upload
            `source_dir` ).
    """
    if estimator.code_location is not None:
        bucket, key = s3.parse_s3_url(estimator.code_location)
        key = s3.s3_path_join(key, estimator._current_job_name, "source", "sourcedir.tar.gz")
    elif estimator.uploaded_code is not None:
        bucket, key = s3.parse_s3_url(estimator.uploaded_code.s3_prefix)
    else:
        bucket = estimator.sagemaker_session.default_bucket
        key = s3.s3_path_join(
            estimator.sagemaker_session.default_bucket_prefix,
            estimator._current_job_name,
            "source",
            "sourcedir.tar.gz",
        )

    script = os.path.basename(estimator.entry_point)

    if estimator.source_dir and estimator.source_dir.lower().startswith("s3://"):
        code_dir = estimator.source_dir
        estimator.uploaded_code = fw_utils.UploadedCode(s3_prefix=code_dir, script_name=script)
    else:
        code_dir = "s3://{}/{}".format(bucket, key)
        estimator.uploaded_code = fw_utils.UploadedCode(s3_prefix=code_dir, script_name=script)
        s3_operations["S3Upload"] = [
            {
                "Path": estimator.source_dir or estimator.entry_point,
                "Bucket": bucket,
                "Key": key,
                "Tar": True,
            }
        ]
    estimator._hyperparameters[sagemaker.model.DIR_PARAM_NAME] = code_dir
    estimator._hyperparameters[sagemaker.model.SCRIPT_PARAM_NAME] = script
    estimator._hyperparameters[
        sagemaker.model.CONTAINER_LOG_LEVEL_PARAM_NAME
    ] = estimator.container_log_level
    estimator._hyperparameters[sagemaker.model.JOB_NAME_PARAM_NAME] = estimator._current_job_name
    estimator._hyperparameters[
        sagemaker.model.SAGEMAKER_REGION_PARAM_NAME
    ] = estimator.sagemaker_session.boto_region_name


def prepare_amazon_algorithm_estimator(estimator, inputs, mini_batch_size=None):
    """Sets up amazon algorithm estimator.

    This is done by adding the required `feature_dim` hyperparameter from training data.

    Args:
        estimator (sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase): An estimator
            for a built-in Amazon algorithm to get information from and update.
        inputs: The training data.
           * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of

                Amazon :class:~`Record` objects serialized and stored in S3. For
                use with an estimator for an Amazon algorithm.

            * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                  :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
                  where each instance is a different channel of training data.
        mini_batch_size:
    """
    if isinstance(inputs, list):
        for record in inputs:
            if isinstance(record, amazon_estimator.RecordSet) and record.channel == "train":
                estimator.feature_dim = record.feature_dim
                break
    elif isinstance(inputs, amazon_estimator.RecordSet):
        estimator.feature_dim = inputs.feature_dim
    else:
        raise TypeError("Training data must be represented in RecordSet or list of RecordSets")
    estimator.mini_batch_size = mini_batch_size


def training_base_config(estimator, inputs=None, job_name=None, mini_batch_size=None):  # noqa: C901
    """Export Airflow base training config from an estimator

    Args:
        estimator (sagemaker.estimator.EstimatorBase): The estimator to export
            training config from. Can be a BYO estimator, Framework estimator or
            Amazon algorithm estimator.
        inputs: Information about the training data. Please refer to the ``fit()``
            method of
                the associated estimator, as this can take any of the following
                forms:

            * (str) - The S3 location where training data is saved.

            * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) - If using multiple
                  channels for training data, you can specify a dict mapping channel names to
                  strings or :func:`~sagemaker.inputs.TrainingInput` objects.

            * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources that can
                  provide additional information about the training dataset. See
                  :func:`sagemaker.inputs.TrainingInput` for full details.

            * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                  Amazon :class:~`Record` objects serialized and stored in S3.
                  For use with an estimator for an Amazon algorithm.

            * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                  :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
                  where each instance is a different channel of training data.
        job_name (str): Specify a training job name if needed.
        mini_batch_size (int): Specify this argument only when estimator is a
            built-in estimator of an Amazon algorithm. For other estimators,
            batch size should be specified in the estimator.

    Returns:
        dict: Training config that can be directly used by
        SageMakerTrainingOperator in Airflow.
    """
    if isinstance(estimator, sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase):
        estimator.prepare_workflow_for_training(
            records=inputs, mini_batch_size=mini_batch_size, job_name=job_name
        )
    else:
        estimator.prepare_workflow_for_training(job_name=job_name)

    s3_operations = {}

    if job_name is not None:
        estimator._current_job_name = job_name
    else:
        base_name = estimator.base_job_name or utils.base_name_from_image(
            estimator.training_image_uri(),
            default_base_name=EstimatorBase.JOB_CLASS_NAME,
        )
        estimator._current_job_name = utils.name_from_base(base_name)

    if estimator.output_path is None:
        estimator.output_path = s3.s3_path_join(
            "s3://",
            estimator.sagemaker_session.default_bucket(),
            estimator.sagemaker_session.default_bucket_prefix,
            with_end_slash=True,
        )

    if isinstance(estimator, sagemaker.estimator.Framework):
        prepare_framework(estimator, s3_operations)

    elif isinstance(estimator, amazon_estimator.AmazonAlgorithmEstimatorBase):
        prepare_amazon_algorithm_estimator(estimator, inputs, mini_batch_size)
    job_config = job._Job._load_config(inputs, estimator, expand_role=False, validate_uri=False)

    train_config = {
        "AlgorithmSpecification": {
            "TrainingImage": estimator.training_image_uri(),
            "TrainingInputMode": estimator.input_mode,
        },
        "OutputDataConfig": job_config["output_config"],
        "StoppingCondition": job_config["stop_condition"],
        "ResourceConfig": job_config["resource_config"],
        "RoleArn": job_config["role"],
    }

    if job_config["input_config"] is not None:
        train_config["InputDataConfig"] = job_config["input_config"]

    if job_config["vpc_config"] is not None:
        train_config["VpcConfig"] = job_config["vpc_config"]

    if estimator.use_spot_instances:
        # estimator.use_spot_instances may be a Pipeline ParameterBoolean object
        # which is parsed during the Pipeline execution runtime
        train_config["EnableManagedSpotTraining"] = estimator.use_spot_instances

    if estimator.hyperparameters() is not None:
        hyperparameters = {str(k): str(v) for (k, v) in estimator.hyperparameters().items()}

    if hyperparameters and len(hyperparameters) > 0:
        train_config["HyperParameters"] = hyperparameters

    if s3_operations:
        train_config["S3Operations"] = s3_operations

    if (estimator.checkpoint_local_path is not None) & (estimator.checkpoint_s3_uri is not None):
        train_config["CheckpointConfig"] = {
            "LocalPath": estimator.checkpoint_local_path,
            "S3Uri": estimator.checkpoint_s3_uri,
        }
    return train_config


def training_config(estimator, inputs=None, job_name=None, mini_batch_size=None):
    """Export Airflow training config from an estimator

    Args:
        estimator (sagemaker.estimator.EstimatorBase): The estimator to export
            training config from. Can be a BYO estimator, Framework estimator or
            Amazon algorithm estimator.
        inputs: Information about the training data. Please refer to the ``fit()``
            method of the associated estimator, as this can take any of the following forms:
            * (str) - The S3 location where training data is saved.

            * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) - If using multiple
                  channels for training data, you can specify a dict mapping channel names to
                  strings or :func:`~sagemaker.inputs.TrainingInput` objects.

            * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources that can
                  provide additional information about the training dataset. See
                  :func:`sagemaker.inputs.TrainingInput` for full details.

            * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                  Amazon :class:~`Record` objects serialized and stored in S3.
                  For use with an estimator for an Amazon algorithm.

            * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                  :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
                  where each instance is a different channel of training data.
        job_name (str): Specify a training job name if needed.
        mini_batch_size (int): Specify this argument only when estimator is a
            built-in estimator of an Amazon algorithm. For other estimators,
            batch size should be specified in the estimator.

    Returns:
        dict: Training config that can be directly used by
        SageMakerTrainingOperator in Airflow.
    """

    train_config = training_base_config(estimator, inputs, job_name, mini_batch_size)

    train_config["TrainingJobName"] = estimator._current_job_name

    if estimator.tags is not None:
        train_config["Tags"] = estimator.tags

    if estimator.metric_definitions is not None:
        train_config["AlgorithmSpecification"]["MetricDefinitions"] = estimator.metric_definitions

    return train_config


def tuning_config(tuner, inputs, job_name=None, include_cls_metadata=False, mini_batch_size=None):
    """Export Airflow tuning config from a HyperparameterTuner

    Args:
        tuner (sagemaker.tuner.HyperparameterTuner): The tuner to export tuning
            config from.
        inputs: Information about the training data. Please refer to the ``fit()``
            method of the associated estimator in the tuner, as this can take any of the
            following forms:

            * (str) - The S3 location where training data is saved.

            * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) - If using multiple
                  channels for training data, you can specify a dict mapping channel names to
                  strings or :func:`~sagemaker.inputs.TrainingInput` objects.

            * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources that can
                  provide additional information about the training dataset. See
                  :func:`sagemaker.inputs.TrainingInput` for full details.

            * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                  Amazon :class:~`Record` objects serialized and stored in S3.
                  For use with an estimator for an Amazon algorithm.

            * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                  :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
                  where each instance is a different channel of training data.

            * (dict[str, one the forms above]): Required by only tuners created via
                  the factory method ``HyperparameterTuner.create()``. The keys should be the
                  same estimator names as keys for the ``estimator_dict`` argument of the
                  ``HyperparameterTuner.create()`` method.
        job_name (str): Specify a tuning job name if needed.
        include_cls_metadata: It can take one of the following two forms.

            * (bool) - Whether or not the hyperparameter tuning job should include information
                about the estimator class (default: False). This information is passed as a
                hyperparameter, so if the algorithm you are using cannot handle unknown
                hyperparameters (e.g. an Amazon SageMaker built-in algorithm that does not
                have a custom estimator in the Python SDK), then set ``include_cls_metadata``
                to ``False``.
            * (dict[str, bool]) - This version should be used for tuners created via the factory
                method ``HyperparameterTuner.create()``, to specify the flag for individual
                estimators provided in the ``estimator_dict`` argument of the method. The keys
                would be the same estimator names as in ``estimator_dict``. If one estimator
                doesn't need the flag set, then no need to include it in the dictionary. If none
                of the estimators need the flag set, then an empty dictionary ``{}`` must be used.

        mini_batch_size: It can take one of the following two forms.

            * (int) - Specify this argument only when estimator is a built-in estimator of an
                Amazon algorithm. For other estimators, batch size should be specified in the
                estimator.
            * (dict[str, int]) - This version should be used for tuners created via the factory
                method ``HyperparameterTuner.create()``, to specify the value for individual
                estimators provided in the ``estimator_dict`` argument of the method. The keys
                would be the same estimator names as in ``estimator_dict``. If one estimator
                doesn't need the value set, then no need to include it in the dictionary. If
                none of the estimators need the value set, then an empty dictionary ``{}``
                must be used.

    Returns:
        dict: Tuning config that can be directly used by SageMakerTuningOperator in Airflow.
    """

    tuner._prepare_job_name_for_tuning(job_name=job_name)

    tune_config = {
        "HyperParameterTuningJobName": tuner._current_job_name,
        "HyperParameterTuningJobConfig": _extract_tuning_job_config(tuner),
    }

    if tuner.estimator:
        (
            tune_config["TrainingJobDefinition"],
            s3_operations,
        ) = _extract_training_config_from_estimator(
            tuner, inputs, include_cls_metadata, mini_batch_size
        )
    else:
        (
            tune_config["TrainingJobDefinitions"],
            s3_operations,
        ) = _extract_training_config_list_from_estimator_dict(
            tuner, inputs, include_cls_metadata, mini_batch_size
        )

    if s3_operations:
        tune_config["S3Operations"] = s3_operations

    if tuner.tags:
        tune_config["Tags"] = tuner.tags

    if tuner.warm_start_config:
        tune_config["WarmStartConfig"] = tuner.warm_start_config.to_input_req()

    return tune_config


def _extract_tuning_job_config(tuner):
    """Extract tuning job config from a HyperparameterTuner"""
    tuning_job_config = {
        "Strategy": tuner.strategy,
        "ResourceLimits": {
            "MaxNumberOfTrainingJobs": tuner.max_jobs,
            "MaxParallelTrainingJobs": tuner.max_parallel_jobs,
        },
        "TrainingJobEarlyStoppingType": tuner.early_stopping_type,
    }

    if tuner.objective_metric_name:
        tuning_job_config["HyperParameterTuningJobObjective"] = {
            "Type": tuner.objective_type,
            "MetricName": tuner.objective_metric_name,
        }

    parameter_ranges = tuner.hyperparameter_ranges()
    if parameter_ranges:
        tuning_job_config["ParameterRanges"] = parameter_ranges

    return tuning_job_config


def _extract_training_config_from_estimator(tuner, inputs, include_cls_metadata, mini_batch_size):
    """Extract training job config from a HyperparameterTuner that uses the ``estimator`` field"""
    train_config = training_base_config(tuner.estimator, inputs, mini_batch_size)
    train_config.pop("HyperParameters", None)

    tuner._prepare_static_hyperparameters_for_tuning(include_cls_metadata=include_cls_metadata)
    train_config["StaticHyperParameters"] = tuner.static_hyperparameters

    if tuner.metric_definitions:
        train_config["AlgorithmSpecification"]["MetricDefinitions"] = tuner.metric_definitions

    s3_operations = train_config.pop("S3Operations", None)
    return train_config, s3_operations


def _extract_training_config_list_from_estimator_dict(
    tuner, inputs, include_cls_metadata, mini_batch_size
):
    """Extracts a list of training job configs from a Hyperparameter Tuner.

    It uses the ``estimator_dict`` field.
    """
    estimator_names = sorted(tuner.estimator_dict.keys())
    tuner._validate_dict_argument(name="inputs", value=inputs, allowed_keys=estimator_names)
    tuner._validate_dict_argument(
        name="include_cls_metadata", value=include_cls_metadata, allowed_keys=estimator_names
    )
    tuner._validate_dict_argument(
        name="mini_batch_size", value=mini_batch_size, allowed_keys=estimator_names
    )

    train_config_dict = {}
    for (estimator_name, estimator) in tuner.estimator_dict.items():
        train_config_dict[estimator_name] = training_base_config(
            estimator=estimator,
            inputs=inputs.get(estimator_name) if inputs else None,
            mini_batch_size=mini_batch_size.get(estimator_name) if mini_batch_size else None,
        )

    tuner._prepare_static_hyperparameters_for_tuning(include_cls_metadata=include_cls_metadata)

    train_config_list = []
    s3_operations_list = []

    for estimator_name in sorted(train_config_dict.keys()):
        train_config = train_config_dict[estimator_name]
        train_config.pop("HyperParameters", None)
        train_config["StaticHyperParameters"] = tuner.static_hyperparameters_dict[estimator_name]

        train_config["AlgorithmSpecification"][
            "MetricDefinitions"
        ] = tuner.metric_definitions_dict.get(estimator_name)

        train_config["DefinitionName"] = estimator_name
        train_config["TuningObjective"] = {
            "Type": tuner.objective_type,
            "MetricName": tuner.objective_metric_name_dict[estimator_name],
        }
        train_config["HyperParameterRanges"] = tuner.hyperparameter_ranges_dict()[estimator_name]

        s3_operations_list.append(train_config.pop("S3Operations", {}))

        train_config_list.append(train_config)

    return train_config_list, _merge_s3_operations(s3_operations_list)


def _merge_s3_operations(s3_operations_list):
    """Merge a list of S3 operation dictionaries into one"""
    s3_operations_merged = {}
    for s3_operations in s3_operations_list:
        for (key, operations) in s3_operations.items():
            if key not in s3_operations_merged:
                s3_operations_merged[key] = []
            for operation in operations:
                if operation not in s3_operations_merged[key]:
                    s3_operations_merged[key].append(operation)
    return s3_operations_merged


def update_submit_s3_uri(estimator, job_name):
    """Updated the S3 URI of the framework source directory in given estimator.

    Args:
        estimator (sagemaker.estimator.Framework): The Framework estimator to
            update.
        job_name (str): The new job name included in the submit S3 URI

    Returns:
        str: The updated S3 URI of framework source directory
    """
    if estimator.uploaded_code is None:
        return

    pattern = r"(?<=/)[^/]+?(?=/source/sourcedir.tar.gz)"

    # update the S3 URI with the latest training job.
    # s3://path/old_job/source/sourcedir.tar.gz will become s3://path/new_job/source/sourcedir.tar.gz
    submit_uri = estimator.uploaded_code.s3_prefix
    submit_uri = re.sub(pattern, job_name, submit_uri)
    script_name = estimator.uploaded_code.script_name
    estimator.uploaded_code = fw_utils.UploadedCode(submit_uri, script_name)


def update_estimator_from_task(estimator, task_id, task_type):
    """Update training job of the estimator from a task in the DAG

    Args:
        estimator (sagemaker.estimator.EstimatorBase): The estimator to update
        task_id (str): The task id of any
            airflow.contrib.operators.SageMakerTrainingOperator or
            airflow.contrib.operators.SageMakerTuningOperator that generates
            training jobs in the DAG.
        task_type (str): Whether the task is from SageMakerTrainingOperator or
            SageMakerTuningOperator. Values can be 'training', 'tuning' or None
            (which means training job is not from any task).
    """
    if task_type is None:
        return
    if task_type.lower() == "training":
        training_job = "{{ ti.xcom_pull(task_ids='%s')['Training']['TrainingJobName'] }}" % task_id
        job_name = training_job
    elif task_type.lower() == "tuning":
        training_job = (
            "{{ ti.xcom_pull(task_ids='%s')['Tuning']['BestTrainingJob']['TrainingJobName'] }}"
            % task_id
        )
        # need to strip the double quotes in json to get the string
        job_name = (
            "{{ ti.xcom_pull(task_ids='%s')['Tuning']['TrainingJobDefinition']"
            "['StaticHyperParameters']['sagemaker_job_name'].strip('%s') }}" % (task_id, '"')
        )
    else:
        raise ValueError("task_type must be either 'training', 'tuning' or None.")
    estimator._current_job_name = training_job
    if isinstance(estimator, sagemaker.estimator.Framework):
        update_submit_s3_uri(estimator, job_name)


def prepare_framework_container_def(model, instance_type, s3_operations):
    """This prepares the framework model container information and specifies related S3 operations.

    Prepare the framework model container information. Specify related S3
    operations for Airflow to perform. (Upload `source_dir` )

    Args:
        model (sagemaker.model.FrameworkModel): The framework model
        instance_type (str): The EC2 instance type to deploy this Model to. For
            example, 'ml.p2.xlarge'.
        s3_operations (dict): The dict to specify S3 operations (upload
            `source_dir` ).

    Returns:
        dict: The container information of this framework model.
    """
    deploy_image = model.image_uri
    if not deploy_image:
        region_name = model.sagemaker_session.boto_session.region_name
        deploy_image = model.serving_image_uri(region_name, instance_type)

    base_name = utils.base_name_from_image(deploy_image)
    model.name = model.name or utils.name_from_base(base_name)

    bucket, key_prefix = s3.determine_bucket_and_prefix(
        bucket=model.bucket,
        key_prefix=None,
        sagemaker_session=model.sagemaker_session,
    )

    if model.entry_point is not None:
        script = os.path.basename(model.entry_point)
        key = "{}/source/sourcedir.tar.gz".format(model.name)

        if model.source_dir and model.source_dir.lower().startswith("s3://"):
            code_dir = model.source_dir
            model.uploaded_code = fw_utils.UploadedCode(s3_prefix=code_dir, script_name=script)
        else:
            code_dir = s3.s3_path_join("s3://", bucket, key_prefix, key)
            model.uploaded_code = fw_utils.UploadedCode(s3_prefix=code_dir, script_name=script)
            s3_operations["S3Upload"] = [
                {"Path": model.source_dir or script, "Bucket": bucket, "Key": key, "Tar": True}
            ]

    deploy_env = dict(model.env)
    deploy_env.update(model._script_mode_env_vars())

    try:
        if model.model_server_workers:
            deploy_env[sagemaker.model.MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(
                model.model_server_workers
            )
    except AttributeError:
        # This applies to a FrameworkModel which is not SageMaker Deep Learning Framework Model
        pass

    return sagemaker.container_def(deploy_image, model.model_data, deploy_env)


def model_config(model, instance_type=None, role=None, image_uri=None):
    """Export Airflow model config from a SageMaker model

    Args:
        model (sagemaker.model.Model): The Model object from which to export the Airflow config
        instance_type (str): The EC2 instance type to deploy this Model to. For
            example, 'ml.p2.xlarge'
        role (str): The ``ExecutionRoleArn`` IAM Role ARN for the model
        image_uri (str): An Docker image URI to use for deploying the model

    Returns:
        dict: Model config that can be directly used by SageMakerModelOperator
            in Airflow. It can also be part of the config used by
            SageMakerEndpointOperator and SageMakerTransformOperator in Airflow.
    """
    s3_operations = {}
    model.image_uri = image_uri or model.image_uri

    if isinstance(model, sagemaker.model.FrameworkModel):
        container_def = prepare_framework_container_def(model, instance_type, s3_operations)
    else:
        container_def = model.prepare_container_def()
        base_name = utils.base_name_from_image(container_def["Image"])
        model.name = model.name or utils.name_from_base(base_name)

    primary_container = session._expand_container_def(container_def)

    config = {
        "ModelName": model.name,
        "PrimaryContainer": primary_container,
        "ExecutionRoleArn": role or model.role,
    }

    if model.vpc_config:
        config["VpcConfig"] = model.vpc_config

    if s3_operations:
        config["S3Operations"] = s3_operations

    return config


def model_config_from_estimator(
    estimator,
    task_id,
    task_type,
    instance_type=None,
    role=None,
    image_uri=None,
    name=None,
    model_server_workers=None,
    vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
):
    """Export Airflow model config from a SageMaker estimator

    Args:
        estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to
            export Airflow config from. It has to be an estimator associated
            with a training job.
        task_id (str): The task id of any
            airflow.contrib.operators.SageMakerTrainingOperator or
            airflow.contrib.operators.SageMakerTuningOperator that generates
            training jobs in the DAG. The model config is built based on the
            training job generated in this operator.
        task_type (str): Whether the task is from SageMakerTrainingOperator or
            SageMakerTuningOperator. Values can be 'training', 'tuning' or None
            (which means training job is not from any task).
        instance_type (str): The EC2 instance type to deploy this Model to. For
            example, 'ml.p2.xlarge'
        role (str): The ``ExecutionRoleArn`` IAM Role ARN for the model
        image_uri (str): A Docker image URI to use for deploying the model
        name (str): Name of the model
        model_server_workers (int): The number of worker processes used by the
            inference server. If None, server will use one worker per vCPU. Only
            effective when estimator is a SageMaker framework.
        vpc_config_override (dict[str, list[str]]): Override for VpcConfig set on
            the model. Default: use subnets and security groups from this Estimator.
            * 'Subnets' (list[str]): List of subnet ids.
            * 'SecurityGroupIds' (list[str]): List of security group ids.

    Returns:
        dict: Model config that can be directly used by SageMakerModelOperator in Airflow. It can
            also be part of the config used by SageMakerEndpointOperator in Airflow.
    """
    update_estimator_from_task(estimator, task_id, task_type)
    if isinstance(estimator, sagemaker.estimator.Estimator):
        model = estimator.create_model(
            role=role, image_uri=image_uri, vpc_config_override=vpc_config_override
        )
    elif isinstance(estimator, sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase):
        model = estimator.create_model(vpc_config_override=vpc_config_override)
    elif isinstance(estimator, TensorFlow):
        model = estimator.create_model(
            role=role, vpc_config_override=vpc_config_override, entry_point=estimator.entry_point
        )
    elif isinstance(estimator, sagemaker.estimator.Framework):
        model = estimator.create_model(
            model_server_workers=model_server_workers,
            role=role,
            vpc_config_override=vpc_config_override,
            entry_point=estimator.entry_point,
        )
    else:
        raise TypeError(
            "Estimator must be one of sagemaker.estimator.Estimator, sagemaker.estimator.Framework"
            " or sagemaker.amazon.amazon_estimator.AmazonAlgorithmEstimatorBase."
        )
    model.name = name

    return model_config(model, instance_type, role, image_uri)


def transform_config(
    transformer,
    data,
    data_type="S3Prefix",
    content_type=None,
    compression_type=None,
    split_type=None,
    job_name=None,
    input_filter=None,
    output_filter=None,
    join_source=None,
):
    """Export Airflow transform config from a SageMaker transformer

    Args:
        transformer (sagemaker.transformer.Transformer): The SageMaker
            transformer to export Airflow config from.
        data (str): Input data location in S3.
        data_type (str): What the S3 location defines (default: 'S3Prefix').
            Valid values:

            * 'S3Prefix' - the S3 URI defines a key name prefix. All objects with this prefix will
                  be used as inputs for the transform job.

            * 'ManifestFile' - the S3 URI points to a single manifest file listing each S3 object
                  to use as an input for the transform job.

        content_type (str): MIME type of the input data (default: None).
        compression_type (str): Compression type of the input data, if
            compressed (default: None). Valid values: 'Gzip', None.
        split_type (str): The record delimiter for the input object (default:
            'None'). Valid values: 'None', 'Line', 'RecordIO', and 'TFRecord'.
        job_name (str): job name (default: None). If not specified, one will be
            generated.
        input_filter (str): A JSONPath to select a portion of the input to
            pass to the algorithm container for inference. If you omit the
            field, it gets the value '$', representing the entire input.
            For CSV data, each row is taken as a JSON array,
            so only index-based JSONPaths can be applied, e.g. $[0], $[1:].
            CSV data should follow the `RFC format <https://tools.ietf.org/html/rfc4180>`_.
            See `Supported JSONPath Operators
            <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html#data-processing-operators>`_
            for a table of supported JSONPath operators.
            For more information, see the SageMaker API documentation for
            `CreateTransformJob
            <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
            Some examples: "$[1:]", "$.features" (default: None).
        output_filter (str): A JSONPath to select a portion of the
            joined/original output to return as the output.
            For more information, see the SageMaker API documentation for
            `CreateTransformJob
            <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
            Some examples: "$[1:]", "$.prediction" (default: None).
        join_source (str): The source of data to be joined to the transform
            output. It can be set to 'Input' meaning the entire input record
            will be joined to the inference result. You can use OutputFilter
            to select the useful portion before uploading to S3. (default:
            None). Valid values: Input, None.

    Returns:
        dict: Transform config that can be directly used by
        SageMakerTransformOperator in Airflow.
    """
    if job_name is not None:
        transformer._current_job_name = job_name
    else:
        base_name = transformer.base_transform_job_name
        transformer._current_job_name = (
            utils.name_from_base(base_name) if base_name is not None else transformer.model_name
        )

    if transformer.output_path is None:
        transformer.output_path = s3.s3_path_join(
            "s3://",
            transformer.sagemaker_session.default_bucket(),
            transformer.sagemaker_session.default_bucket_prefix,
            transformer._current_job_name,
        )

    job_config = sagemaker.transformer._TransformJob._load_config(
        data, data_type, content_type, compression_type, split_type, transformer
    )

    config = {
        "TransformJobName": transformer._current_job_name,
        "ModelName": transformer.model_name,
        "TransformInput": job_config["input_config"],
        "TransformOutput": job_config["output_config"],
        "TransformResources": job_config["resource_config"],
    }

    data_processing = sagemaker.transformer._TransformJob._prepare_data_processing(
        input_filter, output_filter, join_source
    )
    if data_processing is not None:
        config["DataProcessing"] = data_processing

    if transformer.strategy is not None:
        config["BatchStrategy"] = transformer.strategy

    if transformer.max_concurrent_transforms is not None:
        config["MaxConcurrentTransforms"] = transformer.max_concurrent_transforms

    if transformer.max_payload is not None:
        config["MaxPayloadInMB"] = transformer.max_payload

    if transformer.env is not None:
        config["Environment"] = transformer.env

    if transformer.tags is not None:
        config["Tags"] = transformer.tags

    return config


def transform_config_from_estimator(
    estimator,
    task_id,
    task_type,
    instance_count,
    instance_type,
    data,
    data_type="S3Prefix",
    content_type=None,
    compression_type=None,
    split_type=None,
    job_name=None,
    model_name=None,
    strategy=None,
    assemble_with=None,
    output_path=None,
    output_kms_key=None,
    accept=None,
    env=None,
    max_concurrent_transforms=None,
    max_payload=None,
    tags=None,
    role=None,
    volume_kms_key=None,
    model_server_workers=None,
    image_uri=None,
    vpc_config_override=None,
    input_filter=None,
    output_filter=None,
    join_source=None,
):
    """Export Airflow transform config from a SageMaker estimator

    Args:
        estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to
            export Airflow config from. It has to be an estimator associated
            with a training job.
        task_id (str): The task id of any
            airflow.contrib.operators.SageMakerTrainingOperator or
            airflow.contrib.operators.SageMakerTuningOperator that generates
            training jobs in the DAG. The transform config is built based on the
            training job generated in this operator.
        task_type (str): Whether the task is from SageMakerTrainingOperator or
            SageMakerTuningOperator. Values can be 'training', 'tuning' or None
            (which means training job is not from any task).
        instance_count (int): Number of EC2 instances to use.
        instance_type (str): Type of EC2 instance to use, for example,
            'ml.c4.xlarge'.
        data (str): Input data location in S3.
        data_type (str): What the S3 location defines (default: 'S3Prefix').
            Valid values:

            * 'S3Prefix' - the S3 URI defines a key name prefix. All objects with this prefix will
                  be used as inputs for the transform job.

            * 'ManifestFile' - the S3 URI points to a single manifest file listing each S3 object
                  to use as an input for the transform job.
        content_type (str): MIME type of the input data (default: None).
        compression_type (str): Compression type of the input data, if
            compressed (default: None). Valid values: 'Gzip', None.
        split_type (str): The record delimiter for the input object (default:
            'None'). Valid values: 'None', 'Line', 'RecordIO', and 'TFRecord'.
        job_name (str): transform job name (default: None). If not specified,
            one will be generated.
        model_name (str): model name (default: None). If not specified, one will
            be generated.
        strategy (str): The strategy used to decide how to batch records in a
            single request (default: None). Valid values: 'MultiRecord' and
            'SingleRecord'.
        assemble_with (str): How the output is assembled (default: None). Valid
            values: 'Line' or 'None'.
        output_path (str): S3 location for saving the transform result. If not
            specified, results are stored to a default bucket.
        output_kms_key (str): Optional. KMS key ID for encrypting the transform
            output (default: None).
        accept (str): The accept header passed by the client to
            the inference endpoint. If it is supported by the endpoint,
            it will be the format of the batch transform output.
        env (dict): Environment variables to be set for use during the transform
            job (default: None).
        max_concurrent_transforms (int): The maximum number of HTTP requests to
            be made to each individual transform container at one time.
        max_payload (int): Maximum size of the payload in a single HTTP request
            to the container in MB.
        tags (list[dict]): List of tags for labeling a transform job. If none
            specified, then the tags used for the training job are used for the
            transform job.
        role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
            which is also used during transform jobs. If not specified, the role
            from the Estimator will be used.
        volume_kms_key (str): Optional. KMS key ID for encrypting the volume
            attached to the ML compute instance (default: None).
        model_server_workers (int): Optional. The number of worker processes
            used by the inference server. If None, server will use one worker
            per vCPU.
        image_uri (str): A Docker image URI to use for deploying the model
        vpc_config_override (dict[str, list[str]]): Override for VpcConfig set on
            the model. Default: use subnets and security groups from this Estimator.

            * 'Subnets' (list[str]): List of subnet ids.
            * 'SecurityGroupIds' (list[str]): List of security group ids.

        input_filter (str): A JSONPath to select a portion of the input to
            pass to the algorithm container for inference. If you omit the
            field, it gets the value '$', representing the entire input.
            For CSV data, each row is taken as a JSON array,
            so only index-based JSONPaths can be applied, e.g. $[0], $[1:].
            CSV data should follow the `RFC format <https://tools.ietf.org/html/rfc4180>`_.
            See `Supported JSONPath Operators
            <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html#data-processing-operators>`_
            for a table of supported JSONPath operators.
            For more information, see the SageMaker API documentation for
            `CreateTransformJob
            <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
            Some examples: "$[1:]", "$.features" (default: None).
        output_filter (str): A JSONPath to select a portion of the
            joined/original output to return as the output.
            For more information, see the SageMaker API documentation for
            `CreateTransformJob
            <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
            Some examples: "$[1:]", "$.prediction" (default: None).
        join_source (str): The source of data to be joined to the transform
            output. It can be set to 'Input' meaning the entire input record
            will be joined to the inference result. You can use OutputFilter
            to select the useful portion before uploading to S3. (default:
            None). Valid values: Input, None.

    Returns:
        dict: Transform config that can be directly used by
        SageMakerTransformOperator in Airflow.
    """
    model_base_config = model_config_from_estimator(
        estimator=estimator,
        task_id=task_id,
        task_type=task_type,
        instance_type=instance_type,
        role=role,
        image_uri=image_uri,
        name=model_name,
        model_server_workers=model_server_workers,
        vpc_config_override=vpc_config_override,
    )

    if isinstance(estimator, sagemaker.estimator.Framework):
        transformer = estimator.transformer(
            instance_count,
            instance_type,
            strategy,
            assemble_with,
            output_path,
            output_kms_key,
            accept,
            env,
            max_concurrent_transforms,
            max_payload,
            tags,
            role,
            model_server_workers,
            volume_kms_key,
        )
    else:
        transformer = estimator.transformer(
            instance_count,
            instance_type,
            strategy,
            assemble_with,
            output_path,
            output_kms_key,
            accept,
            env,
            max_concurrent_transforms,
            max_payload,
            tags,
            role,
            volume_kms_key,
        )
    transformer.model_name = model_base_config["ModelName"]

    transform_base_config = transform_config(
        transformer,
        data,
        data_type,
        content_type,
        compression_type,
        split_type,
        job_name,
        input_filter,
        output_filter,
        join_source,
    )

    config = {"Model": model_base_config, "Transform": transform_base_config}

    return config


def deploy_config(model, initial_instance_count, instance_type, endpoint_name=None, tags=None):
    """Export Airflow deploy config from a SageMaker model

    Args:
        model (sagemaker.model.Model): The SageMaker model to export the Airflow
            config from.
        initial_instance_count (int): The initial number of instances to run in
            the ``Endpoint`` created from this ``Model``.
        instance_type (str): The EC2 instance type to deploy this Model to. For
            example, 'ml.p2.xlarge'.
        endpoint_name (str): The name of the endpoint to create (default: None).
            If not specified, a unique endpoint name will be created.
        tags (list[dict]): List of tags for labeling a training job. For more,
            see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

    Returns:
        dict: Deploy config that can be directly used by
        SageMakerEndpointOperator in Airflow.
    """
    model_base_config = model_config(model, instance_type)

    production_variant = sagemaker.production_variant(
        model.name, instance_type, initial_instance_count
    )
    name = model.name
    config_options = {"EndpointConfigName": name, "ProductionVariants": [production_variant]}
    if tags is not None:
        config_options["Tags"] = tags

    endpoint_name = endpoint_name or name
    endpoint_base_config = {"EndpointName": endpoint_name, "EndpointConfigName": name}

    config = {
        "Model": model_base_config,
        "EndpointConfig": config_options,
        "Endpoint": endpoint_base_config,
    }

    # if there is s3 operations needed for model, move it to root level of config
    s3_operations = model_base_config.pop("S3Operations", None)
    if s3_operations is not None:
        config["S3Operations"] = s3_operations

    return config


def deploy_config_from_estimator(
    estimator,
    task_id,
    task_type,
    initial_instance_count,
    instance_type,
    model_name=None,
    endpoint_name=None,
    tags=None,
    **kwargs
):
    """Export Airflow deploy config from a SageMaker estimator

    Args:
        estimator (sagemaker.model.EstimatorBase): The SageMaker estimator to
            export Airflow config from. It has to be an estimator associated
            with a training job.
        task_id (str): The task id of any
            airflow.contrib.operators.SageMakerTrainingOperator or
            airflow.contrib.operators.SageMakerTuningOperator that generates
            training jobs in the DAG. The endpoint config is built based on the
            training job generated in this operator.
        task_type (str): Whether the task is from SageMakerTrainingOperator or
            SageMakerTuningOperator. Values can be 'training', 'tuning' or None
            (which means training job is not from any task).
        initial_instance_count (int): Minimum number of EC2 instances to deploy
            to an endpoint for prediction.
        instance_type (str): Type of EC2 instance to deploy to an endpoint for
            prediction, for example, 'ml.c4.xlarge'.
        model_name (str): Name to use for creating an Amazon SageMaker model. If
            not specified, one will be generated.
        endpoint_name (str): Name to use for creating an Amazon SageMaker
            endpoint. If not specified, the name of the SageMaker model is used.
        tags (list[dict]): List of tags for labeling a training job. For more,
            see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
        **kwargs: Passed to invocation of ``create_model()``. Implementations
            may customize ``create_model()`` to accept ``**kwargs`` to customize
            model creation during deploy. For more, see the implementation docs.

    Returns:
        dict: Deploy config that can be directly used by
        SageMakerEndpointOperator in Airflow.
    """
    update_estimator_from_task(estimator, task_id, task_type)
    model = estimator.create_model(**kwargs)
    model.name = model_name
    config = deploy_config(model, initial_instance_count, instance_type, endpoint_name, tags)
    return config


def processing_config(
    processor,
    inputs=None,
    outputs=None,
    job_name=None,
    experiment_config=None,
    container_arguments=None,
    container_entrypoint=None,
    kms_key_id=None,
):
    """Export Airflow processing config from a SageMaker processor

    Args:
        processor (sagemaker.processor.Processor): The SageMaker
            processor to export Airflow config from.
        inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job. These must be provided as
                :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
        outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
            the processing job. These can be specified as either path strings or
            :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
        job_name (str): Processing job name. If not specified, the processor generates
            a default job name, based on the base job name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
        container_arguments ([str]): The arguments for a container used to run a processing job.
        container_entrypoint ([str]): The entrypoint for a container used to run a processing job.
        kms_key_id (str): The AWS Key Management Service (AWS KMS) key that Amazon SageMaker
            uses to encrypt the processing job output. KmsKeyId can be an ID of a KMS key,
            ARN of a KMS key, alias of a KMS key, or alias of a KMS key.
            The KmsKeyId is applied to all outputs.

    Returns:
        dict: Processing config that can be directly used by
            SageMakerProcessingOperator in Airflow.
    """
    if job_name is not None:
        processor._current_job_name = job_name
    else:
        base_name = processor.base_job_name
        processor._current_job_name = (
            utils.name_from_base(base_name)
            if base_name is not None
            else utils.base_name_from_image(processor.image_uri, Processor.JOB_CLASS_NAME)
        )

    config = {
        "ProcessingJobName": processor._current_job_name,
        "ProcessingInputs": input_output_list_converter(inputs),
    }

    processing_output_config = sagemaker.processing.ProcessingJob.prepare_output_config(
        kms_key_id, input_output_list_converter(outputs)
    )

    config["ProcessingOutputConfig"] = processing_output_config

    if experiment_config is not None:
        config["ExperimentConfig"] = experiment_config

    app_specification = sagemaker.processing.ProcessingJob.prepare_app_specification(
        container_arguments, container_entrypoint, processor.image_uri
    )
    config["AppSpecification"] = app_specification

    config["RoleArn"] = processor.role

    if processor.env is not None:
        config["Environment"] = processor.env

    if processor.network_config is not None:
        config["NetworkConfig"] = processor.network_config._to_request_dict()

    processing_resources = sagemaker.processing.ProcessingJob.prepare_processing_resources(
        instance_count=processor.instance_count,
        instance_type=processor.instance_type,
        volume_kms_key_id=processor.volume_kms_key,
        volume_size_in_gb=processor.volume_size_in_gb,
    )
    config["ProcessingResources"] = processing_resources

    if processor.max_runtime_in_seconds is not None:
        stopping_condition = sagemaker.processing.ProcessingJob.prepare_stopping_condition(
            processor.max_runtime_in_seconds
        )
        config["StoppingCondition"] = stopping_condition

    if processor.tags is not None:
        config["Tags"] = processor.tags

    return config


def input_output_list_converter(object_list):
    """Converts a list of ProcessingInput or ProcessingOutput objects to a list of dicts

    Args:
        object_list (list[ProcessingInput or ProcessingOutput]

    Returns:
        List of dicts
    """
    if object_list:
        return [obj._to_request_dict() for obj in object_list]
    return object_list
