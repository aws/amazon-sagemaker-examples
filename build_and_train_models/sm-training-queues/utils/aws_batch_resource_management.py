import json
import logging
import time
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Resource:
    """
    Represents an AWS resource with a name and ARN.

    Attributes:
        name (str): The name of the AWS resource.
        arn (str): The Amazon Resource Name (ARN) of the resource.
    """

    name: str
    arn: str


@dataclass
class Resources:
    """
    Container for AWS Batch resources used in the application.

    Attributes:
        job_queue (Resource): The AWS Batch job queue resource.
        service_environment (Resource): The AWS Batch service environment resource.
    """

    job_queue: Resource = None
    service_environment: Resource = None
    batch_role: Resource = None
    sagemaker_exeuction_role: Resource = None


class AwsBatchResourceManager:
    """
    Manager for AWS Batch resources including service environments and job queues.

    This class provides methods to create, update, delete, and monitor AWS Batch resources.

    Attributes:
        TERMINAL_JOB_STATUSES (set): Set of job statuses considered terminal.
    """

    TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED"}

    def __init__(self, batch_client):
        """
        Initialize the AWS Batch Resource Manager.

        Args:
            batch_client: The boto3 Batch client to use for AWS operations.
        """
        self._batch_client = batch_client

    def create_service_environment(self, create_se_request: dict):
        """
        Create a new AWS Batch service environment.

        If the service environment already exists, returns the existing environment details.

        Args:
            create_se_request (dict): Request parameters for creating a service environment.
                Must contain 'serviceEnvironmentName' key.

        Returns:
            dict: Response containing the service environment name and ARN.

        Raises:
            ClientError: If there's an error creating the service environment.
        """
        try:
            return self._batch_client.create_service_environment(**create_se_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":
                logger.info("ServiceEnvironment already exists, skipping creation.")
                desc_resp = self._batch_client.describe_service_environments(
                    serviceEnvironments=[create_se_request["serviceEnvironmentName"]]
                )
                return {
                    "serviceEnvironmentName": desc_resp["serviceEnvironments"][0][
                        "serviceEnvironmentName"
                    ],
                    "serviceEnvironmentArn": desc_resp["serviceEnvironments"][0][
                        "serviceEnvironmentArn"
                    ],
                }

            logger.error(f"Error: {json.dumps(error.response, indent=4)}")
            raise error

    def update_service_environment(self, update_se_request):
        """
        Update an existing AWS Batch service environment.

        Args:
            update_se_request (dict): Request parameters for updating a service environment.

        Returns:
            dict: Response from the update operation.

        Raises:
            ClientError: If there's an error updating the service environment.
        """
        try:
            return self._batch_client.update_service_environment(**update_se_request)
        except ClientError as error:
            logger.error(f"Error: {json.dumps(error.response, indent=4)}")
            raise error

    def await_service_environment_update(
        self, service_environment_name: str, expected_status: str, expected_state: str
    ):
        """
        Wait for a service environment to reach the expected status and state.

        This method polls the service environment status until it reaches the expected state
        or is deleted if that's the expected status.

        Args:
            se_name (str): Name of the service environment to monitor.
            expected_status (str): The expected status to wait for (e.g., "VALID", "DELETED").
            expected_state (str, optional): The expected state to wait for (e.g., "ENABLED", "DISABLED").

        Returns:
            dict: The describe service environments response when the expected state is reached.
        """
        while True:
            describe_response = self._batch_client.describe_service_environments(
                serviceEnvironments=[service_environment_name]
            )
            if describe_response["serviceEnvironments"]:
                se = describe_response["serviceEnvironments"][0]

                state = se["state"]
                status = se["status"]

                if status == expected_status and state == expected_state:
                    break
                if status == "INVALID":
                    raise ValueError(f"Something went wrong!  {json.dumps(jq, indent=4)}")
            elif expected_status == "DELETED":
                logger.info(f"ServiceEnvironment {service_environment_name} has been deleted")
                break

            time.sleep(5)

    def delete_service_environment(self, service_environment_name: str):
        """
        Delete an AWS Batch service environment.

        This method follows the proper deletion workflow:
        1. Disable the service environment
        2. Wait for the disable operation to complete
        3. Delete the service environment
        4. Wait for the deletion to complete

        Args:
            se_name (str): Name of the service environment to delete.
        """
        logger.info(f"Setting ServiceEnvironment {service_environment_name} to DISABLED")
        self._batch_client.update_service_environment(
            serviceEnvironment=service_environment_name, state="DISABLED"
        )

        logger.info("Waiting for ServiceEnvironment update to finish...")
        self.await_service_environment_update(service_environment_name, "VALID", "DISABLED")

        logger.info(f"Deleting ServiceEnvironment {service_environment_name}")
        self._batch_client.delete_service_environment(serviceEnvironment=service_environment_name)

        logger.info("Waiting for ServiceEnvironment update to finish...")
        self.await_service_environment_update(service_environment_name, "DELETED", "DISABLED")

    def create_job_queue(self, create_jq_request: dict):
        """
        Create a new AWS Batch job queue.

        If the job queue already exists, returns the existing job queue details.

        Args:
            create_jq_request (dict): Request parameters for creating a job queue.
                Must contain 'jobQueueName' key.

        Returns:
            dict: Response containing the job queue name and ARN.

        Raises:
            ClientError: If there's an error creating the job queue.
        """
        try:
            return self._batch_client.create_job_queue(**create_jq_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":
                logger.info("JobQueue already exists, skipping creation")
                desc_resp = self._batch_client.describe_job_queues(
                    jobQueues=[create_jq_request["jobQueueName"]]
                )
                return {
                    "jobQueueName": desc_resp["jobQueues"][0]["jobQueueName"],
                    "jobQueueArn": desc_resp["jobQueues"][0]["jobQueueArn"],
                }

            logger.error(f"Error: {json.dumps(error.response, indent=4)}")
            raise error

    def delete_job_queue(self, job_queue_name: str):
        """
        Delete an AWS Batch job queue.

        This method follows the proper deletion workflow:
        1. Disable the job queue
        2. Wait for the disable operation to complete
        3. Delete the job queue
        4. Wait for the deletion to complete

        Args:
            jq_name (str): Name of the job queue to delete.
        """
        logger.info(f"Disabling JobQueue {job_queue_name}")
        self._batch_client.update_job_queue(jobQueue=job_queue_name, state="DISABLED")

        logger.info("Waiting for JobQueue update to finish...")
        self.await_job_queue_update(job_queue_name, "VALID", "DISABLED")

        logger.info(f"Deleting JobQueue {job_queue_name}")
        self._batch_client.delete_job_queue(jobQueue=job_queue_name)

        logger.info("Waiting for JobQueue update to finish...")
        self.await_job_queue_update(job_queue_name, "DELETED", "DISABLED")

    def await_job_queue_update(
        self, job_queue_name: str, expected_status: str, expected_state: str
    ):
        """
        Wait for a job queue to reach the expected status and state.

        This method polls the job queue status until it reaches the expected state and status
        or is deleted if that's the expected status.

        Args:
            jq_name (str): Name of the job queue to monitor.
            expected_status (str): The expected status to wait for (e.g., "VALID", "DELETED").
            expected_state (str, optional): The expected state to wait for (e.g., "ENABLED", "DISABLED").

        Raises:
            ValueError: If the job queue enters an INVALID status.
        """
        while True:
            describe_jq_response = self._batch_client.describe_job_queues(
                jobQueues=[job_queue_name]
            )
            if describe_jq_response["jobQueues"]:
                jq = describe_jq_response["jobQueues"][0]

                state = jq["state"]
                status = jq["status"]

                if status == expected_status and state == expected_state:
                    break
                if status == "INVALID":
                    raise ValueError(f"Something went wrong!  {json.dumps(jq, indent=4)}")
            elif expected_status == "DELETED":
                logger.info(f"JobQueue {job_queue_name} has been deleted")
                break

            time.sleep(5)


class RoleManager:
    """
    Manager for creating and managing IAM roles required for SageMaker training jobs with AWS Batch.

    This class provides methods to create the necessary IAM roles for AWS Batch to interact with
    SageMaker training jobs, including the batch role and SageMaker execution role.

    Attributes:
        iam_client: The boto3 IAM client to use for creating roles
        sts_client: The boto3 STS client to use for getting account information
    """

    def __init__(self, iam_client, sts_client):
        """
        Initialize the RoleManager with IAM and STS clients.

        Args:
            iam_client: The boto3 IAM client to use.
            sts_client: The boto3 STS client to use.
        """
        self.iam_client = iam_client
        self.sts_client = sts_client

    def create_batch_role(self, batch_role_name: str):
        """
        Create an IAM role for AWS Batch to interact with SageMaker training jobs.

        This method creates a role with permissions for AWS Batch to manage SageMaker
        training jobs, including the ability to create service-linked roles and pass roles
        to SageMaker.

        Args:
            batch_role_name (str): The name to use for the IAM role.
        Returns:
            Resource: A Resource object containing the name and ARN of the created role.

        Raises:
            ClientError: If there's an error creating the role (except when the role already exists).
        """
        get_caller_id_resp = self.sts_client.get_caller_identity()
        account_id = get_caller_id_resp["Account"]

        try:
            create_role_resp = self.iam_client.create_role(
                RoleName=batch_role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    }
                ),
                Description="Role for AWS Batch for SageMaker Training jobs.",
                MaxSessionDuration=3600,
            )
            self.iam_client.put_role_policy(
                RoleName=batch_role_name,
                PolicyName="AWSBatchForSageMakerTrainingJobsPolicy",
                PolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {"Effect": "Allow", "Action": ["batch:*"], "Resource": "*"},
                            {
                                "Effect": "Allow",
                                "Action": ["iam:CreateServiceLinkedRole"],
                                "Resource": "arn:aws:iam::*:role/*AWSServiceRoleForAWSBatchWithSagemaker",
                                "Condition": {
                                    "StringEquals": {
                                        "iam:AWSServiceName": "sagemaker-queuing.batch.amazonaws.com"
                                    }
                                },
                            },
                            {
                                "Effect": "Allow",
                                "Action": "iam:PassRole",
                                "Resource": "*",
                                "Condition": {
                                    "StringEquals": {
                                        "iam:PassedToService": ["sagemaker.amazonaws.com"]
                                    }
                                },
                            },
                        ],
                    }
                ),
            )

            return Resource(
                name=create_role_resp["Role"]["RoleName"], arn=create_role_resp["Role"]["Arn"]
            )
        except ClientError as error:
            if error.response["Error"]["Code"] == "EntityAlreadyExists":
                print(error.response["Error"]["Message"])
                get_resp = self.iam_client.get_role(RoleName=batch_role_name)

                return Resource(name=get_resp["Role"]["RoleName"], arn=get_resp["Role"]["Arn"])

            logger.error(
                f"Error creating {batch_role_name}: {json.dumps(error.__dict__, indent=4)}"
            )
            raise error

    def create_sagemaker_execution_role(self, sagemaker_execution_role_name: str):
        """
        Create an IAM role for SageMaker to execute training jobs.

        This method creates a role with the AmazonSageMakerFullAccess policy attached,
        allowing SageMaker to access necessary resources for training jobs.

        Args:
            sagemaker_execution_role_name (str): The name to use for the IAM role.
        Returns:
            Resource: A Resource object containing the name and ARN of the created role.

        Raises:
            ClientError: If there's an error creating the role (except when the role already exists).
        """
        try:
            create_role_resp = self.iam_client.create_role(
                RoleName=sagemaker_execution_role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": ["sagemaker.amazonaws.com"]},
                                "Action": "sts:AssumeRole",
                            }
                        ],
                    }
                ),
                Description="SageMaker training execution role.",
                MaxSessionDuration=3600,
            )
            self.iam_client.attach_role_policy(
                RoleName=sagemaker_execution_role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            )

            return Resource(
                name=create_role_resp["Role"]["RoleName"], arn=create_role_resp["Role"]["Arn"]
            )
        except ClientError as error:
            if error.response["Error"]["Code"] == "EntityAlreadyExists":
                print(error.response["Error"]["Message"])
                get_resp = self.iam_client.get_role(RoleName=sagemaker_execution_role_name)

                return Resource(name=get_resp["Role"]["RoleName"], arn=get_resp["Role"]["Arn"])

            logger.error(
                f"Error creating {sagemaker_execution_role_name}: {json.dumps(error.__dict__, indent=4)}"
            )
            raise error


def create_roles(
    role_manager: RoleManager, batch_role_name: str, sagemaker_execution_role_name: str
):
    """
    Create all required IAM roles for SageMaker training jobs with AWS Batch.

    This function creates both the AWS Batch role and the SageMaker execution role
    using the current AWS account ID.

    Returns:
        Resources: A Resources class containing the created roles
    """
    logger.info("Creating batch role")
    batch_role = role_manager.create_batch_role(batch_role_name)

    logger.info("Creating sagemaker execution role")
    sagemaker_execution_role = role_manager.create_sagemaker_execution_role(
        sagemaker_execution_role_name
    )

    resources = Resources(batch_role=batch_role, sagemaker_exeuction_role=sagemaker_execution_role)

    logger.info(f"Role creation complete: {resources}")
    return resources


def assume_role_and_get_session(role: Resource, sts_client):
    """
    Assumes the specified IAM role and returns a boto3 session with the assumed credentials.

    Args:
        role: The IAM role resource to assume
        sts_client: The boto3 STS client

    Returns:
        A boto3 session configured with the assumed role credentials
    """
    response = sts_client.assume_role(RoleArn=role.arn, RoleSessionName="AssumeRoleSession")

    credentials = response["Credentials"]

    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def create_resources(
    resource_manager: AwsBatchResourceManager,
    job_queue_name: str,
    service_environment_name: str,
    max_capacity: int = 1,
):
    """
    Create AWS Batch resources including a service environment and job queue.

    This function creates a SageMaker training service environment and a corresponding
    job queue, waiting for each resource to reach a VALID state before proceeding.

    Args:
        resource_manager (AwsBatchResourceManager): The resource manager to use for creating resources.

    Returns:
        Resources: A Resources object containing the created service environment and job queue.
    """
    # Create ServiceEnvironment
    logger.info(f"Creating ServiceEnvironment: {service_environment_name}")
    create_se_resp = resource_manager.create_service_environment(
        {
            "serviceEnvironmentName": service_environment_name,
            "serviceEnvironmentType": "SAGEMAKER_TRAINING",
            "state": "ENABLED",
            "capacityLimits": [{"maxCapacity": max_capacity, "capacityUnit": "NUM_INSTANCES"}],
        }
    )
    logger.info("Waiting for ServiceEnvironment to transition to VALID...")
    resource_manager.await_service_environment_update(service_environment_name, "VALID", "ENABLED")

    # Create JobQueue
    logger.info(f"Creating JobQueue: {job_queue_name}")
    create_jq_response = resource_manager.create_job_queue(
        {
            "jobQueueName": job_queue_name,
            "jobQueueType": "SAGEMAKER_TRAINING",
            "state": "ENABLED",
            "priority": 1,
            "serviceEnvironmentOrder": [
                {"order": 1, "serviceEnvironment": create_se_resp["serviceEnvironmentName"]},
            ],
        }
    )
    logger.info("Waiting for JobQueue to transition to VALID...")
    resource_manager.await_job_queue_update(job_queue_name, "VALID", "ENABLED")

    resources = Resources(
        service_environment=Resource(
            name=create_se_resp["serviceEnvironmentName"],
            arn=create_se_resp["serviceEnvironmentArn"],
        ),
        job_queue=Resource(
            name=create_jq_response["jobQueueName"], arn=create_jq_response["jobQueueArn"]
        ),
    )

    logger.info(f"Resource creation complete: {resources}")
    return resources


def delete_resources(resource_manager: AwsBatchResourceManager, resources: Resources):
    """
    Delete AWS Batch resources.

    This function deletes the job queue first and then the service environment,
    following the proper order for resource cleanup.

    Args:
        resource_manager (AwsBatchResourceManager): The resource manager to use for deleting resources.
        resources (Resources): The Resources object containing the resources to delete.
    """
    if resources.job_queue:
        logger.info(f"Deleting JobQueue: {resources.job_queue.name}")
        resource_manager.delete_job_queue(resources.job_queue.name)

    if resources.service_environment:
        logger.info(f"Deleting ServiceEnvironment: {resources.service_environment.name}")
        resource_manager.delete_service_environment(resources.service_environment.name)
