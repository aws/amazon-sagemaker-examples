from dataclasses import dataclass
import boto3

# Construct Batch client
batch_client = boto3.client("batch")


class ServiceJobType:
    SAGEMAKER_TRAINING = "SAGEMAKER_TRAINING"


class ServiceEnvs:
    C5_SE_NAME = "ml-c5-xlarge-fifo"
    C5_SE_MAX_CAPACITY = 1
    C5_SE_CREATE_REQUEST = {
        "serviceEnvironmentName": C5_SE_NAME,
        "serviceEnvironmentType": ServiceJobType.SAGEMAKER_TRAINING,
        "state": "ENABLED",
        "capacityLimits": [
            {"maxCapacity": C5_SE_MAX_CAPACITY, "capacityUnit": "NUM_INSTANCES"}
        ],
    }
    G6_SE_NAME = "ml-g6-12xlarge-fss"
    G6_SE_MAX_CAPACITY = 1
    G6_SE_CREATE_REQUEST = {
        "serviceEnvironmentName": G6_SE_NAME,
        "serviceEnvironmentType": ServiceJobType.SAGEMAKER_TRAINING,
        "state": "ENABLED",
        "capacityLimits": [
            {"maxCapacity": G6_SE_MAX_CAPACITY, "capacityUnit": "NUM_INSTANCES"}
        ],
    }


class SchedulingPolicies:
    G6_SP_NAME = "ml-g6-12xlarge-sp"
    G6_SP_CREATE_REQUEST = {
        "name": G6_SP_NAME,
        "fairsharePolicy": {
            "shareDecaySeconds": 3600,
            "shareDistribution": [
                {"shareIdentifier": "HIGHPRI", "weightFactor": 1},
                {"shareIdentifier": "MIDPRI", "weightFactor": 3},
                {"shareIdentifier": "LOWPRI", "weightFactor": 5},
            ],
        },
    }


class TrainingJobQueues:
    C5_TJ_QUEUE_NAME = "ml-c5-xlarge-queue"
    C5_TJ_QUEUE_PRIORITY = 2
    C5_TJ_QUEUE_CREATE_REQUEST = {
        "jobQueueName": C5_TJ_QUEUE_NAME,
        "jobQueueType": ServiceJobType.SAGEMAKER_TRAINING,
        "state": "ENABLED",
        "priority": C5_TJ_QUEUE_PRIORITY,
        "serviceEnvironmentOrder": [
            {"order": 1, "serviceEnvironment": ServiceEnvs.C5_SE_NAME},
        ],
    }

    G6_TJ_QUEUE_NAME = "ml-g6-12xlarge-queue"
    G6_TJ_QUEUE_PRIORITY = 1
    G6_TJ_QUEUE_CREATE_REQUEST = {
        "jobQueueName": G6_TJ_QUEUE_NAME,
        "jobQueueType": ServiceJobType.SAGEMAKER_TRAINING,
        "state": "ENABLED",
        "priority": G6_TJ_QUEUE_PRIORITY,
        "serviceEnvironmentOrder": [
            {"order": 1, "serviceEnvironment": ServiceEnvs.G6_SE_NAME},
        ],
        "schedulingPolicyArn": SchedulingPolicies.G6_SP_NAME,
    }


@dataclass
class Resource:
    name: str
    arn: str


@dataclass
class Resources:
    service_environments: list = None
    scheduling_policies: list = None
    job_queues: list = None

    def __post_init__(self):
        if self.service_environments is None:
            self.service_environments = []
        if self.scheduling_policies is None:
            self.scheduling_policies = []
        if self.job_queues is None:
            self.job_queues = []
