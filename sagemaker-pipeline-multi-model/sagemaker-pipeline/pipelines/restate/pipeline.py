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
"""Example workflow pipeline script for RESVM pipeline.
                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)
Implements a get_pipeline(**kwargs) method.
"""

import os

import boto3
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig, TuningStep

###
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.dataset_definition.inputs import (
    AthenaDatasetDefinition,
    DatasetDefinition,
    RedshiftDatasetDefinition,
)


import time
import uuid
import sagemaker

import os
import json
import boto3

from sagemaker.processing import Processor
from sagemaker.network import NetworkConfig

from sagemaker.workflow.steps import ProcessingStep

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig

from sagemaker.tuner import (
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes,
)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="",
    pipeline_name="",
    base_job_prefix="",
):
    """Gets a SageMaker ML Pipeline instance working with on RE data.
    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.2xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"",  # Change this to point to the s3 location of your raw input data.
    )

    # Sagemaker session
    sess = sagemaker_session

    # You can configure this with your own bucket name, e.g.
    # bucket = "my-bucket"
    bucket = sess.default_bucket()

    print(f"Data Wrangler export storage bucket: {bucket}")

    # unique flow export ID
    flow_export_id = f"{time.strftime('%d-%H-%M-%S', time.gmtime())}-{str(uuid.uuid4())[:8]}"
    flow_export_name = f"flow-{flow_export_id}"

    # Output name is auto-generated from the select node's ID + output name from the flow file.
    output_name = "99ae1ec3-dd5f-453c-bfae-721dac423cd7.default"

    s3_output_prefix = f"export-{flow_export_name}/output"
    s3_output_path = f"s3://{bucket}/{s3_output_prefix}"
    print(f"Flow S3 export result path: {s3_output_path}")

    processing_job_output = ProcessingOutput(
        output_name=output_name,
        source="/opt/ml/processing/output",
        destination=s3_output_path,
        s3_upload_mode="EndOfJob",
    )

    # name of the flow file which should exist in the current notebook working directory
    flow_file_name = "sagemaker-pipeline/restate-athena-california.flow"

    # Load .flow file from current notebook working directory
    #!echo "Loading flow file from current notebook working directory: $PWD"

    with open(flow_file_name) as f:
        flow = json.load(f)

    # Upload flow to S3
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        flow_file_name,
        bucket,
        f"data_wrangler_flows/{flow_export_name}.flow",
        ExtraArgs={"ServerSideEncryption": "aws:kms"},
    )

    flow_s3_uri = f"s3://{bucket}/data_wrangler_flows/{flow_export_name}.flow"

    print(f"Data Wrangler flow {flow_file_name} uploaded to {flow_s3_uri}")

    flow_input = ProcessingInput(
        source=flow_s3_uri,
        destination="/opt/ml/processing/flow",
        input_name="flow",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
    )

    # IAM role for executing the processing job.
    iam_role = role

    # Unique processing job name. Give a unique name every time you re-execute processing jobs
    processing_job_name = f"data-wrangler-flow-processing-{flow_export_id}"

    # Data Wrangler Container URL.
    container_uri = sagemaker.image_uris.retrieve(
        framework="data-wrangler",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
    )

    # Processing Job Instance count and instance type.
    instance_count = 2
    instance_type = "ml.m5.4xlarge"

    # Size in GB of the EBS volume to use for storing data during processing
    volume_size_in_gb = 30

    # Content type for each output. Data Wrangler supports CSV as default and Parquet.
    output_content_type = "CSV"

    # Network Isolation mode; default is off
    enable_network_isolation = False

    # List of tags to be passed to the processing job
    user_tags = []

    # Output configuration used as processing job container arguments
    output_config = {output_name: {"content_type": output_content_type}}

    # KMS key for per object encryption; default is None
    kms_key = None

    processor = Processor(
        role=iam_role,
        image_uri=container_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size_in_gb=volume_size_in_gb,
        network_config=NetworkConfig(enable_network_isolation=enable_network_isolation),
        sagemaker_session=sess,
        output_kms_key=kms_key,
        tags=user_tags,
    )

    data_wrangler_step = ProcessingStep(
        name="DataWranglerProcess",
        processor=processor,
        inputs=[flow_input],
        outputs=[processing_job_output],
        job_arguments=[f"--output-config '{json.dumps(output_config)}'"],
    )

    # Processing step for feature engineering
    # this processor does not have awswrangler installed
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-restate-preprocess",  # choose any name
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_process = ProcessingStep(
        name="Preprocess",  # choose any name
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=data_wrangler_step.properties.ProcessingOutputConfig.Outputs[
                    output_name
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/data/raw-data-dir",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=[
            "--input-data",
            data_wrangler_step.properties.ProcessingOutputConfig.Outputs[
                output_name
            ].S3Output.S3Uri,
        ],
    )

    # Training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/restateTrain"
    model_bucket_key = f"{sagemaker_session.default_bucket()}/{base_job_prefix}/restateTrain"
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    xgb_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",  # we are using the Sagemaker built in xgboost algorithm
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    xgb_train = Estimator(
        image_uri=xgb_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/restate-xgb-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    xgb_train.set_hyperparameters(
        num_round=50,
    )

    xgb_train.set_hyperparameters(grow_policy="lossguide")

    xgb_objective_metric_name = "validation:mse"
    xgb_hyperparameter_ranges = {
        "max_depth": IntegerParameter(2, 10, scaling_type="Linear"),
    }

    xgb_tuner_log = HyperparameterTuner(
        xgb_train,
        xgb_objective_metric_name,
        xgb_hyperparameter_ranges,
        max_jobs=3,
        max_parallel_jobs=3,
        strategy="Random",
        objective_type="Minimize",
    )

    xgb_step_tuning = TuningStep(
        name="XGBHPTune",
        tuner=xgb_tuner_log,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    dtree_image_uri = sagemaker_session.sagemaker_client.describe_image_version(
        ImageName="restate-dtree"
    )["ContainerImage"]

    dtree_train = Estimator(
        image_uri=dtree_image_uri,
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        base_job_name=f"{base_job_prefix}/restate-dtree-train",
        output_path=model_path,
        sagemaker_session=sagemaker_session,
    )

    dtree_objective_metric_name = "validation:mse"
    dtree_metric_definitions = [{"Name": "validation:mse", "Regex": "mse:(\S+)"}]

    dtree_hyperparameter_ranges = {
        "max_depth": IntegerParameter(10, 50, scaling_type="Linear"),
        "max_leaf_nodes": IntegerParameter(2, 12, scaling_type="Linear"),
    }

    dtree_tuner_log = HyperparameterTuner(
        dtree_train,
        dtree_objective_metric_name,
        dtree_hyperparameter_ranges,
        dtree_metric_definitions,
        max_jobs=3,
        max_parallel_jobs=3,
        strategy="Random",
        objective_type="Minimize",
    )

    dtree_step_tuning = TuningStep(
        name="DTreeHPTune",
        tuner=dtree_tuner_log,
        inputs={
            "training": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    dtree_script_eval = ScriptProcessor(
        image_uri=dtree_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-dtree-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    dtree_evaluation_report = PropertyFile(
        name="EvaluationReportDTree",
        output_name="dtree_evaluation",
        path="dtree_evaluation.json",
    )

    dtree_step_eval = ProcessingStep(
        name="DTreeEval",
        processor=dtree_script_eval,
        inputs=[
            ProcessingInput(
                # source=dtree_step_train.properties.ModelArtifacts.S3ModelArtifacts,
                source=dtree_step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="dtree_evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
        code=os.path.join(BASE_DIR, "dtree_evaluate.py"),
        property_files=[dtree_evaluation_report],
    )

    xgb_script_eval = ScriptProcessor(
        image_uri=xgb_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-xgb-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    xgb_evaluation_report = PropertyFile(
        name="EvaluationReportXGBoost",
        output_name="xgb_evaluation",
        path="xgb_evaluation.json",
    )

    xgb_step_eval = ProcessingStep(
        name="XGBEval",
        processor=xgb_script_eval,
        inputs=[
            ProcessingInput(
                source=xgb_step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="xgb_evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "xgb_evaluate.py"),
        property_files=[xgb_evaluation_report],
    )

    xgb_model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/xgb_evaluation.json".format(
                xgb_step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    dtree_model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/dtree_evaluation.json".format(
                dtree_step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            content_type="application/json",
        )
    )

    xgb_eval_metrics = JsonGet(
        step=xgb_step_eval,
        property_file=xgb_evaluation_report,
        json_path="regression_metrics.r2s.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
    )

    dtree_eval_metrics = JsonGet(
        step=dtree_step_eval,
        property_file=dtree_evaluation_report,
        json_path="regression_metrics.r2s.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
    )

    # Register model step that will be conditionally executed
    dtree_step_register = RegisterModel(
        name="DTreeReg",
        estimator=dtree_train,
        model_data=dtree_step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=dtree_model_metrics,
    )

    # Register model step that will be conditionally executed
    xgb_step_register = RegisterModel(
        name="XGBReg",
        estimator=xgb_train,
        model_data=xgb_step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=xgb_model_metrics,
    )

    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step=dtree_step_eval,
            property_file=dtree_evaluation_report,
            json_path="regression_metrics.r2s.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=JsonGet(
            step=xgb_step_eval,
            property_file=xgb_evaluation_report,
            json_path="regression_metrics.r2s.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),  # You can change the threshold here
    )

    step_cond = ConditionStep(
        name="AccuracyCond",
        conditions=[cond_lte],
        if_steps=[dtree_step_register],
        else_steps=[xgb_step_register],
    )
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data
        ],
        pipeline_experiment_config=PipelineExperimentConfig(
            pipeline_name + "-" + create_date, "restate-{}".format(create_date)
        ),
        steps=[
            data_wrangler_step,
            step_process,
            dtree_step_tuning,
            xgb_step_tuning,
            dtree_step_eval,
            xgb_step_eval,
            step_cond,
        ],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
