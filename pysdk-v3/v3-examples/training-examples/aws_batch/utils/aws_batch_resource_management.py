import json
import logging
import time
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

import boto3
from botocore.exceptions import ClientError
from typing import List, Set, Tuple

from sagemaker.train.aws_batch.training_queue import TrainingQueue
from sagemaker.train.aws_batch.training_queued_job import TrainingQueuedJob

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
    scheduling_policy: Resource = None
    batch_role: Resource = None
    sagemaker_execution_role: Resource = None
    quota_shares: list = None


@dataclass
class QuotaShareConfig:
    """
    Configuration for an AWS Batch quota share.

    Attributes:
        name (str): Name of the quota share.
        capacity_unit (str): Unit of capacity (e.g., "ml.g5.xlarge").
        max_capacity (int): Maximum capacity for this quota share.
        in_share_preemption (bool): Whether in-share preemption is enabled.
        sharing_strategy (str): Resource sharing strategy (e.g., "RESERVE", "LEND", "LEND_AND_BORROW").
        borrow_limit (int, optional): Maximum capacity that can be borrowed, only applicable for LEND_AND_BORROW
          sharing strategy.
    """

    name: str
    capacity_unit: str
    max_capacity: int
    in_share_preemption: bool
    sharing_strategy: str
    borrow_limit: int = None

    def create_quota_share_request(self, job_queue_name: str):
        """
        Build the request dictionary for creating a quota share.

        Returns:
            dict: Request parameters for the create_quota_share API call.
        """
        return {
            "quotaShareName": self.name,
            "jobQueue": job_queue_name,
            "capacityLimits": [{
                "capacityUnit": self.capacity_unit,
                "maxCapacity": self.max_capacity,
            }],
            "resourceSharingConfiguration": {
                "strategy": self.sharing_strategy,
                **({"borrowLimit": self.borrow_limit} if self.borrow_limit else {})
            },
            "preemptionConfiguration": {
                "inSharePreemption": "ENABLED" if self.in_share_preemption else "DISABLED"
            },
            "state": "ENABLED"
        }


class JobStatus(str, Enum):
    """
    Enumeration of AWS Batch job statuses.

    Provides helper methods to get sets of statuses for common filtering operations.
    """

    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    RUNNABLE = "RUNNABLE"
    SCHEDULED = "SCHEDULED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @staticmethod
    def active() -> set:
        """
        Returns the set of job statuses indicating a job is still active (not yet completed).

        Returns:
            set: Job statuses for jobs that are queued or running (SUBMITTED, PENDING,
                RUNNABLE, SCHEDULED, STARTING, RUNNING).
        """
        return {
            JobStatus.SUBMITTED,
            JobStatus.PENDING,
            JobStatus.RUNNABLE,
            JobStatus.SCHEDULED,
            JobStatus.STARTING,
            JobStatus.RUNNING
        }

    @staticmethod
    def dispatched() -> set:
        """
        Returns the set of job statuses indicating a job has been dispatched to compute resources.

        Returns:
            set: Job statuses for jobs that have been scheduled or completed (SCHEDULED,
                STARTING, RUNNING, SUCCEEDED, FAILED).
        """
        return {
            JobStatus.SCHEDULED,
            JobStatus.STARTING,
            JobStatus.RUNNING,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED
        }

    @staticmethod
    def terminal() -> set:
        """
        Returns the set of job statuses indicating a job has reached a final state.

        Returns:
            set: Job statuses for completed jobs (SUCCEEDED, FAILED).
        """
        return {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED
        }

    @staticmethod
    def all() -> set:
        """
        Returns the set of all possible job statuses.

        Returns:
            set: All job statuses (SUBMITTED, PENDING, RUNNABLE, SCHEDULED, STARTING,
                RUNNING, SUCCEEDED, FAILED).
        """
        return {
            JobStatus.SUBMITTED,
            JobStatus.PENDING,
            JobStatus.RUNNABLE,
            JobStatus.SCHEDULED,
            JobStatus.STARTING,
            JobStatus.RUNNING,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED
        }


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
            service_environment_name (str): Name of the service environment to monitor.
            expected_status (str): The expected status to wait for (e.g., "VALID", "DELETED").
            expected_state (str): The expected state to wait for (e.g., "ENABLED", "DISABLED").

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
                    raise ValueError(f"Something went wrong!  {json.dumps(describe_response, indent=4)}")
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
            service_environment_name (str): Name of the service environment to delete.
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
            job_queue_name (str): Name of the job queue to delete.
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
            job_queue_name (str): Name of the job queue to monitor.
            expected_status (str): The expected status to wait for (e.g., "VALID", "DELETED").
            expected_state (str): The expected state to wait for (e.g., "ENABLED", "DISABLED").

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

    def create_scheduling_policy(self, create_sp_request: dict):
        """
        Create a new AWS Batch scheduling policy.

        If the scheduling policy already exists, returns the existing policy details.

        Args:
            create_sp_request (dict): Request parameters for creating a scheduling policy.
                Must contain 'name' key. Optional: 'quotaSharePolicy', 'tags'.

        Returns:
            dict: Response containing the scheduling policy name and ARN.

        Raises:
            ClientError: If there's an error creating the scheduling policy.
        """
        try:
            return self._batch_client.create_scheduling_policy(**create_sp_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":
                logger.info("SchedulingPolicy already exists, skipping creation")
                list_resp = self._batch_client.list_scheduling_policies()
                sp = [p for p in list_resp["schedulingPolicies"] if create_sp_request["name"] in p["arn"]][0]
                return {"name": create_sp_request["name"], "arn": sp["arn"]}
            logger.error(f"Error: {json.dumps(error.response, indent=4)}")
            raise error

    def delete_scheduling_policy(self, scheduling_policy_arn: str):
        """
        Delete an AWS Batch scheduling policy.

        Args:
            scheduling_policy_arn (str): ARN of the scheduling policy to delete.
        """
        logger.info(f"Deleting SchedulingPolicy {scheduling_policy_arn}")
        self._batch_client.delete_scheduling_policy(arn=scheduling_policy_arn)

    def create_quota_share(self, create_qs_request: dict):
        """
        Create a new AWS Batch quota share.

        Args:
            create_qs_request (dict): Request parameters for creating a quota share.
                Required: quotaShareName, jobQueueArn, capacityLimits,
                resourceSharingConfiguration, preemptionConfiguration.

        Returns:
            dict: Response containing the quota share name and ARN.
        """
        try:
            return self._batch_client.create_quota_share(**create_qs_request)
        except ClientError as error:
            if "already exists" in error.response["message"]:
                logger.info("QuotaShare already exists, skipping creation")
                desc_jqs_resp = self._batch_client.describe_job_queues(
                    jobQueues=[create_qs_request["jobQueue"]]
                )
                jq_arn = desc_jqs_resp["jobQueues"][0]["jobQueueArn"]
                quota_share_arn = f"{jq_arn}/quota-share/{create_qs_request['quotaShareName']}"
                return {
                    "quotaShareName": create_qs_request["quotaShareName"],
                    "quotaShareArn": quota_share_arn,
                }
            logger.error(f"Error: {json.dumps(error.response, indent=4)}")
            raise error

    def await_quota_share_update(
        self, quota_share_arn: str, expected_status: str, expected_state: str
    ):
        """
        Wait for a quota share to reach the expected status.

        Args:
            quota_share_arn (str): ARN of the quota share to monitor.
            expected_status (str): The expected status (VALID, DELETED, etc.).
            expected_state (str): The expected state to wait for (e.g., ENABLED, DISABLED).
        """
        while True:
            try:
                describe_response = self._batch_client.describe_quota_share(
                    quotaShareArn=quota_share_arn
                )

                state = describe_response["state"]
                status = describe_response["status"]

                if status == expected_status and state == expected_state:
                    break
                if describe_response["status"] == "INVALID":
                    raise ValueError(f"Something went wrong!  {json.dumps(describe_response, indent=4)}")
            except ClientError as error:
                if expected_status == "DELETED":
                    logger.info(f"QuotaShare {quota_share_arn} has been deleted")
                    break
                raise error

            time.sleep(5)

    def list_quota_shares(self, job_queue: str):
        """
        List quota shares for a job queue.

        Args:
            job_queue (str): Name or ARN of the job queue.

        Returns:
            list: List of quota share summaries.
        """
        return self._batch_client.list_quota_shares(jobQueue=job_queue).get("quotaShares", [])

    def delete_quota_share(self, quota_share_arn: str):
        """
        Delete an AWS Batch quota share.

        Args:
            quota_share_arn (str): ARN of the quota share to delete.
        """
        logger.info(f"Disabling QuotaShare {quota_share_arn}")
        self._batch_client.update_quota_share(quotaShareArn=quota_share_arn, state="DISABLED")

        logger.info("Waiting for QuotaShare update to finish...")
        self.await_quota_share_update(quota_share_arn, "VALID", "DISABLED")

        logger.info(f"Deleting QuotaShare {quota_share_arn}")
        self._batch_client.delete_quota_share(quotaShareArn=quota_share_arn)

        logger.info("Waiting for QuotaShare deletion to finish...")
        self.await_quota_share_update(quota_share_arn, "DELETED", "DISABLED")


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

    resources = Resources(batch_role=batch_role, sagemaker_execution_role=sagemaker_execution_role)

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


def await_jobs(job_groups: List[Tuple[List[TrainingQueuedJob], Set[JobStatus]]]):
    """
    Wait for jobs to reach desired statuses, polling in parallel.

    Args:
        job_groups: List of tuples, each containing a list of jobs and the set of
            statuses to wait for those jobs to reach.
    """

    def poll(job: TrainingQueuedJob, desired_status_set: Set[JobStatus]):
        while True:
            job_status = job.describe().get("status", "")

            if job_status in desired_status_set:
                logger.info(f"Job: {job.job_name} is {job_status}")
                break

            time.sleep(5)

    all_tasks = [(job, statuses) for jobs, statuses in job_groups for job in jobs]

    with ThreadPoolExecutor(max_workers=len(all_tasks)) as executor:
        futures = [executor.submit(poll, job, statuses) for job, statuses in all_tasks]
        for future in as_completed(futures):
            future.result()


def _status_message(job_detail) -> str:
    """
    Extract a status message from job details for logging.

    Args:
        job_detail: Job detail dictionary from describe_jobs response.

    Returns:
        str: Formatted status reason string, or empty string if none available.
    """
    status_reason: str = job_detail.get("statusReason", None)
    recent_preempted_attempts: list = job_detail.get("preemptionSummary", {}).get("recentPreemptedAttempts", [{}])
    preemption_status_reason: str = recent_preempted_attempts[0].get("statusReason", None)

    # Return most recent preempted attempt statusReason if it exists
    if preemption_status_reason and job_detail.get("status", "") not in {JobStatus.SCHEDULED, JobStatus.STARTING}:
        return f" ({preemption_status_reason})"

    # Return top level statusReason if no preempted attempts
    if status_reason:
        return f" ({status_reason})"

    # Return empty string if no statusReasons are set
    return ""


def list_jobs_by_quota_share(training_queue: TrainingQueue, quota_share_names: List[str], statuses: Set[JobStatus]):
    """
    Lists all jobs in a TrainingQueue grouped by their quota share and prints a formatted log statement showing each
    quota share and the details for each job.

    Args:
        training_queue (TrainingQueue): The TrainingQueue to query for jobs.
    """
    all_jobs = [
        job for status in statuses for qs_name in quota_share_names
        for job in training_queue.list_jobs_by_share(quota_share_name=qs_name, status=status.value)
    ]
    jobs_by_qs = {qs_name: [] for qs_name in quota_share_names}
    for job in all_jobs:
        job_detail = job.describe()
        jobs_by_qs.setdefault(job_detail["quotaShareName"], []).append(job_detail)

    log_lines = []
    for qs_name, job_details in jobs_by_qs.items():
        log_lines.append(f"QuotaShare: {qs_name}" + "".join(
            f"\n  -> {jd['jobName']} (priority: {jd['schedulingPriority']}): {jd['status']}{_status_message(jd)}"
            for jd in job_details
        ))

    logger.info("Listing jobs by QuotaShare:\n" + "\n".join(log_lines))


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
        job_queue_name (str): Name for the job queue.
        service_environment_name (str): Name for the service environment.
        max_capacity (int): Maximum instance capacity for the service environment. Defaults to 1.

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


def create_quota_management_resources(
    resource_manager: AwsBatchResourceManager,
    job_queue_name: str,
    service_environment_name: str,
    capacity_unit: str,
    max_capacity: int,
    scheduling_policy_name: str,
    quota_share_configs: List[QuotaShareConfig],
):
    """
    Create AWS Batch resources including a service environment, job queue, scheduling policy, and quota shares.

    Args:
        resource_manager (AwsBatchResourceManager): The resource manager to use for creating resources.
        job_queue_name (str): Name for the job queue.
        service_environment_name (str): Name for the service environment.
        capacity_unit (str): The capacity unit for the service environment (e.g., "NUM_INSTANCES").
        max_capacity (int): Maximum capacity for the service environment.
        scheduling_policy_name (str): Name for the scheduling policy.
        quota_share_configs (List[QuotaShareConfig]): List of QuotaShareConfig objects defining quota shares.

    Returns:
        Resources: A Resources object containing the created service environment, job queue,
            scheduling policy, and quota shares.
    """
    # Create SchedulingPolicy
    logger.info(f"Creating SchedulingPolicy: {scheduling_policy_name}")
    create_sp_resp = resource_manager.create_scheduling_policy({
        "name": scheduling_policy_name,
        "quotaSharePolicy": {"idleResourceAssignmentStrategy": "FIFO"},
    })
    scheduling_policy = Resource(name=create_sp_resp["name"], arn=create_sp_resp["arn"])

    # Create ServiceEnvironment
    logger.info(f"Creating ServiceEnvironment: {service_environment_name}")
    create_se_resp = resource_manager.create_service_environment(
        {
            "serviceEnvironmentName": service_environment_name,
            "serviceEnvironmentType": "SAGEMAKER_TRAINING",
            "state": "ENABLED",
            "capacityLimits": [{"maxCapacity": max_capacity, "capacityUnit": capacity_unit}],
        }
    )
    logger.info("Waiting for ServiceEnvironment to transition to VALID...")
    resource_manager.await_service_environment_update(service_environment_name, "VALID", "ENABLED")

    # Create JobQueue
    logger.info(f"Creating JobQueue: {job_queue_name}")
    create_jq_request = {
        "jobQueueName": job_queue_name,
        "jobQueueType": "SAGEMAKER_TRAINING",
        "state": "ENABLED",
        "priority": 1,
        "serviceEnvironmentOrder": [
            {"order": 1, "serviceEnvironment": create_se_resp["serviceEnvironmentName"]},
        ],
        "schedulingPolicyArn": scheduling_policy.arn
    }
    create_jq_response = resource_manager.create_job_queue(create_jq_request)
    logger.info("Waiting for JobQueue to transition to VALID...")
    resource_manager.await_job_queue_update(job_queue_name, "VALID", "ENABLED")

    # Create QuotaShares
    quota_shares = []
    for qs_config in quota_share_configs:
        logger.info(f"Creating QuotaShare: {qs_config.name}")
        create_qs_resp = resource_manager.create_quota_share(qs_config.create_quota_share_request(job_queue_name))
        resource_manager.await_quota_share_update(create_qs_resp["quotaShareArn"], "VALID", "ENABLED")
        quota_shares.append(Resource(name=create_qs_resp["quotaShareName"], arn=create_qs_resp["quotaShareArn"]))

    resources = Resources(
        service_environment=Resource(
            name=create_se_resp["serviceEnvironmentName"],
            arn=create_se_resp["serviceEnvironmentArn"],
        ),
        job_queue=Resource(
            name=create_jq_response["jobQueueName"],
            arn=create_jq_response["jobQueueArn"]
        ),
        scheduling_policy=scheduling_policy,
        quota_shares=quota_shares,
    )

    logger.info(f"Resource creation complete: {resources}")
    return resources


def delete_resources(resource_manager: AwsBatchResourceManager, resources: Resources):
    """
    Delete AWS Batch resources.

    This function deletes quota shares first, then the job queue, then the scheduling policy,
    and finally the service environment, following the proper order for resource cleanup.

    Args:
        resource_manager (AwsBatchResourceManager): The resource manager to use for deleting resources.
        resources (Resources): The Resources object containing the resources to delete.
    """
    if resources.quota_shares:
        for qs in resources.quota_shares:
            logger.info(f"Deleting QuotaShare: {qs.name}")
            resource_manager.delete_quota_share(qs.arn)

    if resources.job_queue:
        logger.info(f"Deleting JobQueue: {resources.job_queue.name}")
        resource_manager.delete_job_queue(resources.job_queue.name)

    if resources.scheduling_policy:
        logger.info(f"Deleting SchedulingPolicy: {resources.scheduling_policy.name}")
        resource_manager.delete_scheduling_policy(resources.scheduling_policy.arn)

    if resources.service_environment:
        logger.info(f"Deleting ServiceEnvironment: {resources.service_environment.name}")
        resource_manager.delete_service_environment(resources.service_environment.name)
