#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import random
import traceback
import uuid

import boto3


def recommend_instance_type(code_choice, dataset_directory):
    """
    Based on the code and [airline] dataset-size choices we recommend
    instance types that we've tested and are known to work.
    Feel free to ignore/make a different choice.
    """
    recommended_instance_type = None

    if "CPU" in code_choice and dataset_directory in ["1_year", "3_year", "NYC_taxi"]:  # noqa
        detail_str = "16 cpu cores, 64GB memory"
        recommended_instance_type = "ml.m5.4xlarge"

    elif "CPU" in code_choice and dataset_directory in ["10_year"]:
        detail_str = "96 cpu cores, 384GB memory"
        recommended_instance_type = "ml.m5.24xlarge"

    if code_choice == "singleGPU":
        detail_str = "1x GPU [ V100 ], 16GB GPU memory, 61GB CPU memory"
        recommended_instance_type = "ml.p3.2xlarge"
        assert dataset_directory not in ["10_year"]  # ! switch to multi-GPU

    elif code_choice == "multiGPU":
        detail_str = "4x GPUs [ V100 ], 64GB GPU memory,  244GB CPU memory"
        recommended_instance_type = "ml.p3.8xlarge"

    print(
        f"recommended instance type : {recommended_instance_type} \n"
        f"instance details          : {detail_str}"
    )

    return recommended_instance_type


def validate_dockerfile(rapids_base_container, dockerfile_name="Dockerfile"):
    """Validate that our desired rapids base image matches the Dockerfile"""
    with open(dockerfile_name, "r") as dockerfile_handle:
        if rapids_base_container not in dockerfile_handle.read():
            raise Exception(
                "Dockerfile base layer [i.e. FROM statment] does"
                " not match the variable rapids_base_container"
            )


def summarize_choices(
    s3_data_input,
    s3_model_output,
    code_choice,
    algorithm_choice,
    cv_folds,
    instance_type,
    use_spot_instances_flag,
    search_strategy,
    max_jobs,
    max_parallel_jobs,
    max_duration_of_experiment_seconds,
):
    """
    Print the configuration choices,
    often useful before submitting large jobs
    """
    print(f"s3 data input    =\t{s3_data_input}")
    print(f"s3 model output  =\t{s3_model_output}")
    print(f"compute          =\t{code_choice}")
    print(f"algorithm        =\t{algorithm_choice}, {cv_folds} cv-fold")
    print(f"instance         =\t{instance_type}")
    print(f"spot instances   =\t{use_spot_instances_flag}")
    print(f"hpo strategy     =\t{search_strategy}")
    print(f"max_experiments  =\t{max_jobs}")
    print(f"max_parallel     =\t{max_parallel_jobs}")
    print(f"max runtime      =\t{max_duration_of_experiment_seconds} sec")


def summarize_hpo_results(tuning_job_name):
    """
    Query tuning results and display the best score,
    parameters, and job-name
    """
    hpo_results = (
        boto3.Session()
        .client("sagemaker")
        .describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)
    )

    best_job = hpo_results["BestTrainingJob"]["TrainingJobName"]
    best_score = hpo_results["BestTrainingJob"]["FinalHyperParameterTuningJobObjectiveMetric"][
        "Value"
    ]  # noqa
    best_params = hpo_results["BestTrainingJob"]["TunedHyperParameters"]
    print(f"best score: {best_score}")
    print(f"best params: {best_params}")
    print(f"best job-name: {best_job}")
    return hpo_results


def download_best_model(bucket, s3_model_output, hpo_results, local_directory):
    """Download best model from S3"""
    try:
        target_bucket = boto3.resource("s3").Bucket(bucket)
        path_prefix = os.path.join(
            s3_model_output.split("/")[-1],
            hpo_results["BestTrainingJob"]["TrainingJobName"],
            "output",
        )
        objects = target_bucket.objects.filter(Prefix=path_prefix)

        for obj in objects:
            path, filename = os.path.split(obj.key)

            local_filename = os.path.join(local_directory, "best_" + filename)
            s3_path_to_model = os.path.join("s3://", bucket, path_prefix, filename)
            target_bucket.download_file(obj.key, local_filename)
            print(
                f"Successfully downloaded best model\n"
                f"> filename: {local_filename}\n"
                f"> local directory : {local_directory}\n\n"
                f"full S3 path : {s3_path_to_model}"
            )

        return local_filename, s3_path_to_model

    except Exception as download_error:
        print(f"! Unable to download best model: {download_error}")
        return None


def new_job_name_from_config(
    dataset_directory, region, code_choice, algorithm_choice, cv_folds, instance_type, trim_limit=32
):
    """
    Build a jobname string that captures the HPO configuration options.
    This is helpful for intepreting logs and for general book-keeping
    """
    job_name = None
    try:
        if dataset_directory in ["1_year", "3_year", "10_year"]:
            data_choice_str = "air"
            validate_region(region)
        elif dataset_directory in ["NYC_taxi"]:
            data_choice_str = "nyc"
            validate_region(region)
        else:
            data_choice_str = "byo"

        code_choice_str = code_choice[0] + code_choice[-3:]

        if "randomforest" in algorithm_choice.lower():
            algorithm_choice_str = "RF"
        if "xgboost" in algorithm_choice.lower():
            algorithm_choice_str = "XGB"

        # instance_type_str = '-'.join(instance_type.split('.')[1:])

        random_str = "".join(random.choices(uuid.uuid4().hex, k=trim_limit))

        job_name = (
            f"{data_choice_str}-{code_choice_str}"
            f"-{algorithm_choice_str}-{cv_folds}cv"
            f"-{random_str}"
        )

        job_name = job_name[:trim_limit]

        print(f"generated job name : {job_name}\n")

    except Exception:
        traceback.print_exc()

    return job_name


def validate_region(region):
    """
    Check that the current [compute] region is one of the
    two regions where the demo data is hosted
    """
    if isinstance(region, list):
        region = region[0]

    if region not in ["us-east-1", "us-west-2"]:
        raise Exception(
            "Unsupported region based on demo data location,"
            " please switch to us-east-1 or us-west-2"
        )
