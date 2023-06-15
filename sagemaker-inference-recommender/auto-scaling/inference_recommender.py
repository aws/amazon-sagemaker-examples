import boto3
import time

region = boto3.Session().region_name
sm_client = boto3.client("sagemaker", region_name=region)


def trigger_inference_recommender_job(model_url, payload_url, container_url, instance_type, execution_role, framework,
                                      framework_version, domain="MACHINE_LEARNING", task="OTHER", model_name="classifier",
                                      mime_type="text/csv"):
    """
    This function creates model package, and starts an inference recommender default job.
    Then it waits for the job to be completed.
    """
    model_package_arn = create_model_package(model_url, payload_url, container_url, instance_type,
                                             framework, framework_version, domain, task, model_name, mime_type)
    job_name = create_inference_recommender_job(model_package_arn, execution_role)
    wait_for_job_completion(job_name)
    return job_name, model_package_arn


def create_model_package(model_url, payload_url, container_url, instance_type, framework, framework_version,
                         domain, task, model_name, mime_type):
    """
    This function create a model package with provided input.
    """
    model_package_group_name = "{}-model-".format(framework) + str(round(time.time()))
    model_package_group_description = "{} models".format(task.lower())

    model_package_group_input_dict = {
        "ModelPackageGroupName": model_package_group_name,
        "ModelPackageGroupDescription": model_package_group_description,
    }

    sm_client.create_model_package_group(**model_package_group_input_dict)

    model_package_description = "{} {} inference recommender".format(framework, model_name)
    model_approval_status = "PendingManualApproval"
    create_model_package_input_dict = {
        "ModelPackageGroupName": model_package_group_name,
        "Domain": domain.upper(),
        "Task": task.upper(),
        "SamplePayloadUrl": payload_url,
        "ModelPackageDescription": model_package_description,
        "ModelApprovalStatus": model_approval_status,
    }
    supported_realtime_inference_types = [
        instance_type
    ]
    model_package_inference_specification = {
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": container_url,
                    "Framework": framework.upper(),
                    "FrameworkVersion": framework_version,
                    "NearestModelName": model_name,
                }
            ],
            "SupportedContentTypes": [mime_type],  # required, must be non-null
            "SupportedResponseMIMETypes": [],
            "SupportedRealtimeInferenceInstanceTypes": supported_realtime_inference_types,  # optional
        }
    }

    # Specify the model data
    model_package_inference_specification["InferenceSpecification"]["Containers"][0][
        "ModelDataUrl"
    ] = model_url
    create_model_package_input_dict.update(model_package_inference_specification)
    create_mode_package_response = sm_client.create_model_package(**create_model_package_input_dict)
    model_package_arn = create_mode_package_response["ModelPackageArn"]
    return model_package_arn


def create_inference_recommender_job(model_package_arn, execution_role):
    """
    This function creates inference recommender default job.
    """
    job_name = "recommender-instance-" + str(round(time.time()))
    job_description = "job to find scaling limit"
    job_type = "Default"
    recommender_id = sm_client.create_inference_recommendations_job(JobName=job_name, JobDescription=job_description,
                                                                    JobType=job_type, RoleArn=execution_role,
                                                                    InputConfig={"ModelPackageVersionArn":
                                                                                     model_package_arn})

    return job_name


def trigger_inference_recommender_evaluation_job(model_package_arn, execution_role, endpoint_name, instance_type,
                                                 max_invocations, max_model_latency, spawn_rate):
    """
    This function create inference recommender advanced job and waits for its completion.
    """
    job_name = "scaling-evaluation-" + str(round(time.time()))
    job_description = "evaluate scaling policy"
    job_type = "Advanced"
    advanced_response = sm_client.create_inference_recommendations_job(
        JobName=job_name,
        JobDescription=job_description,
        JobType=job_type,
        RoleArn=execution_role,
        InputConfig={
            "ModelPackageVersionArn": model_package_arn,
            "Endpoints": [{"EndpointName": endpoint_name}],
            "JobDurationInSeconds": 7200,
            "EndpointConfigurations": [
                {
                    "InstanceType": instance_type,
                }
            ],
            "ResourceLimit": {"MaxNumberOfTests": 2, "MaxParallelOfTests": 2},
            "TrafficPattern": {
                "TrafficType": "PHASES",
                "Phases": [{"InitialNumberOfUsers": 1, "SpawnRate": spawn_rate, "DurationInSeconds": 3600}],
            },
        },
        StoppingConditions={
            "MaxInvocations": max_invocations,
            "ModelLatencyThresholds": [{"Percentile": "P95", "ValueInMilliseconds": max_model_latency}],
        },
    )
    wait_for_job_completion(job_name)
    return job_name


def wait_for_job_completion(job_name):
    """
    This function waits for the Inference job to the in terminal state.
    """
    finished = False
    while not finished:
        inference_recommender_job = sm_client.describe_inference_recommendations_job(JobName=job_name)
        if inference_recommender_job["Status"] in ["COMPLETED", "STOPPED", "FAILED"]:
            finished = True
        else:
            print("Inference Recommender Job {} is in progress".format(job_name))
            time.sleep(300)

    if inference_recommender_job["Status"] == "FAILED":
        print("Inference recommender job failed ")
        print("Failed Reason: {}".format(inference_recommender_job))
    else:
        print("Inference recommender job completed")

