import json
import time
from botocore.exceptions import ClientError
from config import Resource


class TrainingQueueManager:
    TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED"}

    def __init__(self, batch_client):
        self._batch_client = batch_client
        self.log_create_msgs = True

    # Service Environments
    def create_service_env(self, create_se_request):
        try:
            return self._batch_client.create_service_environment(**create_se_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":

                desc_resp = self._batch_client.describe_service_environments(
                    serviceEnvironments=[create_se_request["serviceEnvironmentName"]]
                )
                se_name = desc_resp["serviceEnvironments"][0]["serviceEnvironmentName"]
                se_arn = desc_resp["serviceEnvironments"][0]["serviceEnvironmentArn"]
                if self.log_create_msgs:
                    print(f"Service environment already exists: {se_arn}")
                return {
                    "serviceEnvironmentName": se_name,
                    "serviceEnvironmentArn": se_arn,
                }

            print(f"ERROR: {json.dumps(error.response, indent=4)}")

    def wait_for_se_update(
        self, se_name: str, expected_status: str, expected_state: str = "ENABLED"
    ):
        while True:
            describe_se_response = self._batch_client.describe_service_environments(
                serviceEnvironments=[se_name]
            )
            if describe_se_response["serviceEnvironments"]:
                se = describe_se_response["serviceEnvironments"][0]
                if se["state"] == expected_state:
                    break
            elif expected_status == "DELETED":
                print(f"SE {se_name} has been deleted")
                break

            time.sleep(5)

    def delete_service_env(self, se_name: str):
        print(f"Setting SE {se_name} to DISABLED")
        self._batch_client.update_service_environment(
            serviceEnvironment=se_name, state="DISABLED"
        )

        print("Waiting for SE update to finish...")
        self.wait_for_se_update(se_name, "VALID", "DISABLED")

        print(f"Deleting SE {se_name}")
        self._batch_client.delete_service_environment(serviceEnvironment=se_name)

        print("Waiting for SE update to finish...")
        self.wait_for_se_update(se_name, "DELETED", "DISABLED")

    # Scheduling Policies
    def create_scheduling_policy(self, create_sp_request):
        try:
            return self._batch_client.create_scheduling_policy(**create_sp_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":
                sp_arn = None
                sp_name = None
                list_resp = self._batch_client.list_scheduling_policies()
                for sp in list_resp["schedulingPolicies"]:
                    sp_read_name_from_arn = sp["arn"].split("scheduling-policy/")[-1]
                    if sp_read_name_from_arn == create_sp_request["name"]:
                        sp_arn = sp["arn"]
                        sp_name = sp_read_name_from_arn
                if self.log_create_msgs:
                    print(f"Scheduling Policy already exists: {sp_arn}")
                return {
                    "name": sp_name,
                    "arn": sp_arn,
                }

            print(f"ERROR: {json.dumps(error.response, indent=4)}")

    # Job Queues
    def create_job_queue(self, create_jq_request):
        try:
            return self._batch_client.create_job_queue(**create_jq_request)
        except ClientError as error:
            if error.response["message"] == "Object already exists":
                desc_resp = self._batch_client.describe_job_queues(
                    jobQueues=[create_jq_request["jobQueueName"]]
                )
                jq_name = desc_resp["jobQueues"][0]["jobQueueName"]
                jq_arn = desc_resp["jobQueues"][0]["jobQueueArn"]
                if self.log_create_msgs:
                    print(f"Job queue already exists: {jq_arn}")
                return {
                    "jobQueueName": jq_name,
                    "jobQueueArn": jq_arn,
                }

            print(f"ERROR: {json.dumps(error.response, indent=4)}")

    def delete_scheduling_policy(self, sp_arn: str):
        self._batch_client.delete_scheduling_policy(arn=sp_arn)

    def delete_job_queue(self, jq_name: str):
        print(f"Disabling JQ {jq_name}")
        self._batch_client.update_job_queue(jobQueue=jq_name, state="DISABLED")

        print("Waiting for JQ update to finish...")
        self.wait_for_jq_update(jq_name, "VALID", "DISABLED")

        print(f"Deleting JQ {jq_name}")
        self._batch_client.delete_job_queue(jobQueue=jq_name)

        print("Waiting for JQ update to finish...")
        self.wait_for_jq_update(jq_name, "DELETED", "DISABLED")

    def wait_for_jq_update(
        self, jq_name: str, expected_status: str, expected_state: str = "ENABLED"
    ):
        while True:
            describe_jq_response = self._batch_client.describe_job_queues(
                jobQueues=[jq_name]
            )
            if describe_jq_response["jobQueues"]:
                jq = describe_jq_response["jobQueues"][0]

                state = jq["state"]
                status = jq["status"]

                if status == expected_status and state == expected_state:
                    break
                if status == "INVALID":
                    raise ValueError(
                        f"Something went wrong!  {json.dumps(jq, indent=4)}"
                    )
            elif expected_status == "DELETED":
                print(f"JQ {jq_name} has been deleted")
                break

            time.sleep(5)


def create_service_environments(jq_manager, service_env_config):
    resources = []
    for attr_name in dir(service_env_config):
        if attr_name.endswith('_CREATE_REQUEST') and not attr_name.startswith('_'):
            name_attr = attr_name.replace('_CREATE_REQUEST', '_NAME')
            if hasattr(service_env_config, name_attr):
                se_name = getattr(service_env_config, name_attr)
                create_request = getattr(service_env_config, attr_name)
                
                if jq_manager.log_create_msgs:
                    print(f"Creating service environment: {se_name}")
                create_se_resp = jq_manager.create_service_env(create_request)
                jq_manager.wait_for_se_update(create_se_resp["serviceEnvironmentName"], "VALID")
                resources.append(Resource(
                    name=create_se_resp["serviceEnvironmentName"],
                    arn=create_se_resp["serviceEnvironmentArn"]
                ))
    return resources


def create_scheduling_policy(jq_manager, scheduling_policy_config):
    resources = []
    for attr_name in dir(scheduling_policy_config):
        if attr_name.endswith('_CREATE_REQUEST') and not attr_name.startswith('_'):
            name_attr = attr_name.replace('_CREATE_REQUEST', '_NAME')
            if hasattr(scheduling_policy_config, name_attr):
                sp_name = getattr(scheduling_policy_config, name_attr)
                create_request = getattr(scheduling_policy_config, attr_name)
                
                if jq_manager.log_create_msgs:
                    print(f"Creating scheduling policy: {sp_name}")
                create_sp_resp = jq_manager.create_scheduling_policy(create_request)
                resources.append(Resource(name=create_sp_resp["name"], arn=create_sp_resp["arn"]))
    return resources


def create_training_queues(jq_manager, jq_config, resources):
    queue_resources = []
    
    for attr_name in dir(jq_config):
        if attr_name.endswith('_CREATE_REQUEST') and not attr_name.startswith('_'):
            name_attr = attr_name.replace('_CREATE_REQUEST', '_NAME')
            if hasattr(jq_config, name_attr):
                queue_name = getattr(jq_config, name_attr)
                create_request = getattr(jq_config, attr_name).copy()
                
                if jq_manager.log_create_msgs:
                    print(f"Creating training job queue: {queue_name}")
                
                # Handle scheduling policy if present
                if 'schedulingPolicyArn' in create_request:
                    sp_name = create_request['schedulingPolicyArn']
                    for sp_resource in resources.scheduling_policies:
                        if sp_resource.name == sp_name:
                            create_request['schedulingPolicyArn'] = sp_resource.arn
                            break
                
                create_jq_response = jq_manager.create_job_queue(create_request)
                jq_manager.wait_for_jq_update(create_jq_response["jobQueueName"], "VALID")
                queue_resources.append(Resource(
                    name=create_jq_response["jobQueueName"], 
                    arn=create_jq_response["jobQueueArn"]
                ))
    return queue_resources


def delete_training_queues(jq_manager, resources):
    # Delete job queues
    for queue in resources.job_queues:
        print(f"Deleting job queue: {queue.name}")
        jq_manager.delete_job_queue(queue.name)
    
    # Delete service environments
    for se in resources.service_environments:
        print(f"Deleting service environment: {se.name}")
        jq_manager.delete_service_env(se.name)
    
    # Delete scheduling policies
    for sp in resources.scheduling_policies:
        print(f"Deleting scheduling policy: {sp.name}")
        jq_manager.delete_scheduling_policy(sp.arn)
