import argparse
import config as config
from queue_manager import TrainingQueueManager
from queue_manager import (
    create_service_environments,
    create_training_queues,
    create_scheduling_policy,
)
from queue_manager import delete_training_queues


def create_resources(cleaning_up):
    # this function will return jq_manager instance and the created resources
    jq_manager = TrainingQueueManager(config.batch_client)
    # If we're cleaning up, don't bother logging messages stating resources 'already exist'.
    if cleaning_up:
        jq_manager.log_create_msgs = False
    # Create a data class container for our resources
    resources = config.Resources()
    # Creates the service environments
    resources.service_environments = create_service_environments(jq_manager, config.ServiceEnvs)
    # Creates the scheduling policy
    resources.scheduling_policies = create_scheduling_policy(jq_manager, config.SchedulingPolicies)
    # Creates the job queues
    resources.job_queues = create_training_queues(jq_manager, config.TrainingJobQueues, resources)
    # return the jq manager and resources
    return jq_manager, resources


def delete_resources(jq_manager, resources):
    delete_training_queues(jq_manager, resources)
    print("Resource removal complete.")


def main():
    # Provide a resource clean-up option. False by default.
    # True when user passes --clean via cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Deletes the resources created in this example",
    )
    args = parser.parse_args()

    # Create the service environments and job queues for our examples
    jq_manager, resources = create_resources(args.clean)

    # Delete resources and exit if clean argument is passed
    if args.clean:
        delete_resources(jq_manager, resources)


if __name__ == "__main__":
    main()
