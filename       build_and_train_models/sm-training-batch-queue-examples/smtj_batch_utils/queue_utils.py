import datetime
import logging
from sagemaker.aws_batch.boto_client import get_batch_boto_client
from sagemaker.aws_batch.training_queue import TrainingQueuedJob

logging.getLogger().setLevel(logging.ERROR)


def print_queue_state(queue_obj):
    # List the submitted jobs
    print(
        f"======== {queue_obj.queue_name} Submitted and Runnable  ===================================\n "
    )
    # List the jobs by a state match
    submitted_jobs = queue_obj.list_jobs(status="SUBMITTED")
    runnable_jobs = queue_obj.list_jobs(status="RUNNABLE")
    if len(submitted_jobs + runnable_jobs) == 0:
        print("    ... no submitted or runnable jobs ... ")
    # Now, loop over the jobs and print out the state
    for job in submitted_jobs + runnable_jobs:
        job_status = job.describe().get("status", "")
        print(f"Job : {job.job_name} is {job_status}")
    print(
        "\n======================================================================================= \n\n"
    )

    # Use JQ Snapshot to Display State of the Queue

    batch_client = get_batch_boto_client()
    jobs = batch_client.get_job_queue_snapshot(jobQueue=queue_obj.queue_name)[
        "frontOfQueue"
    ]["jobs"]

    job_order_index = 1
    print(
        f"======== {queue_obj.queue_name} Queue Head as of {datetime.datetime.now():%Y-%m-%d-%H-%M-%S} =========================\n "
    )
    if len(jobs) == 0:
        print("    ... no jobs in queue ... ")
    for job in jobs:
        job_name = batch_client.describe_service_job(jobId=job["jobArn"]).get("jobName")
        queued_job = TrainingQueuedJob(job["jobArn"], job_name)
        job_status = queued_job.describe().get("status", "")
        job_name = queued_job.describe().get("jobName", "")
        print(f"Queue Position {job_order_index} : {job_name} is {job_status}")
        job_order_index += 1
    print(
        f"\n======== {queue_obj.queue_name} Queue Tail =================================================== \n\n"
    )

    # List the starting and running jobs
    print(
        f"======== {queue_obj.queue_name} Starting and Running Jobs  ===================================\n "
    )
    # List the jobs by a state match
    starting_jobs = queue_obj.list_jobs(status="STARTING")
    running_jobs = queue_obj.list_jobs(status="RUNNING")
    if len(starting_jobs + running_jobs) == 0:
        print("    ... no running jobs ... ")
    # Now, loop over the jobs and print out the state
    for job in starting_jobs + running_jobs:
        job_status = job.describe().get("status", "")
        print(f"Job : {job.job_name} is {job_status}")
    print(
        "\n======================================================================================= \n"
    )


def print_completed_jobs(queue_obj):
    # List the completed jobs
    print(
        f"======== {queue_obj.queue_name} Completed Jobs  ==========================\n "
    )
    # List the jobs by a state match
    completed_jobs = queue_obj.list_jobs(status="SUCCEEDED")
    failed_jobs = queue_obj.list_jobs(status="FAILED")
    # Now, loop over the jobs and print out the state
    for job in completed_jobs + failed_jobs:
        job_status = job.describe().get("status", "")
        print(f"Job : {job.job_name} is {job_status}")
    print(
        "\n======================================================================================= \n\n"
    )
