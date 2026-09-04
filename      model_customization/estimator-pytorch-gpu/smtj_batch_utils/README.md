### Create AWS Batch resources for SageMaker Training job

#### Define AWS Batch Queue for SageMaker Training Job

To create your batch queues, edit the file [config.py](./utils/config.py). The configuration system uses a dynamic naming convention that automatically discovers and creates resources based on attribute names.

##### Configuration Naming Convention

The system automatically detects resources using these naming patterns:

**Service Environments:**

- `*_SE_NAME` - The name of the service environment
- `*_SE_CREATE_REQUEST` - The creation request dictionary

**Scheduling Policies:**

- `*_SP_NAME` - The name of the scheduling policy
- `*_SP_CREATE_REQUEST` - The creation request dictionary

**Job Queues:**

- `*_QUEUE_NAME` - The name of the job queue
- `*_QUEUE_CREATE_REQUEST` - The creation request dictionary

**Example Configuration:**

```python
class ServiceEnvs:
    C5_SE_NAME = "ml-c5-xlarge-fifo"
    C5_SE_CREATE_REQUEST = {
        "serviceEnvironmentName": C5_SE_NAME,
        "serviceEnvironmentType": "SAGEMAKER_TRAINING",
        "state": "ENABLED",
        "capacityLimits": [{"maxCapacity": 3, "capacityUnit": "NUM_INSTANCES"}]
    }

class SchedulingPolicies:
    G6_SP_NAME = "ml-g6-scheduling-policy"
    G6_SP_CREATE_REQUEST = {
        "name": G6_SP_NAME,
        "fairsharePolicy": {
            "shareDecaySeconds": 3600,
            "shareDistribution": [
                {"shareIdentifier": "HIGHPRI", "weightFactor": 1}
            ]
        }
    }

class TrainingJobQueues:
    G6_TJ_QUEUE_NAME = "ml-g6-queue"
    G6_TJ_QUEUE_CREATE_REQUEST = {
        "jobQueueName": G6_TJ_QUEUE_NAME,
        "jobQueueType": "SAGEMAKER_TRAINING",
        "state": "ENABLED",
        "priority": 1,
        "serviceEnvironmentOrder": [{"order": 1, "serviceEnvironment": "ml-g6-service-env"}],
        "schedulingPolicyArn": "ml-g6-scheduling-policy"  # References SP_NAME
    }
```

##### Queue Types for SageMaker Training

1. **FIFO Queue** - No scheduling policy required
2. **Fair-Share Queue** - Requires scheduling policy ARN (reference the SP_NAME)

The system automatically links scheduling policies to queues by matching the policy name.

#### Create AWS Batch resources

To create your Batch job queues and service environments using boto3, run the [create_resources.py](create_resources.py) script from a terminal as shown below.

```
python3 create_resources.py
```

You can view your new resources from the CLI with the following commands.
