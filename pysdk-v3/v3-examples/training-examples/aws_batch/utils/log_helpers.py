import botocore
import sys
import time
from sagemaker.core.logs import ColorWrap, Position, multi_stream_iter
from sagemaker.core.common_utils import (
    secondary_training_status_changed,
    secondary_training_status_message,
)

class LogState(object):
    """Placeholder docstring"""
    STARTING = 1
    WAIT_IN_PROGRESS = 2
    TAILING = 3
    JOB_COMPLETE = 4
    COMPLETE = 5

STATUS_CODE_TABLE = {
    "COMPLETED": "Completed",
    "INPROGRESS": "InProgress",
    "IN_PROGRESS": "InProgress",
    "FAILED": "Failed",
    "STOPPED": "Stopped",
    "STOPPING": "Stopping",
    "STARTING": "Starting",
    "PENDING": "Pending",
}


def wait_until(callable_fn, poll=5):
    """Placeholder docstring"""
    elapsed_time = 0
    result = None
    while result is None:
        try:
            elapsed_time += poll
            time.sleep(poll)
            result = callable_fn()
        except botocore.exceptions.ClientError as err:
            # For initial 5 mins we accept/pass AccessDeniedException.
            # The reason is to await tag propagation to avoid false AccessDenied claims for an
            # access policy based on resource tags, The caveat here is for true AccessDenied
            # cases the routine will fail after 5 mins
            if err.response["Error"]["Code"] == "AccessDeniedException" and elapsed_time <= 300:
                logger.warning(
                    "Received AccessDeniedException. This could mean the IAM role does not "
                    "have the resource permissions, in which case please add resource access "
                    "and retry. For cases where the role has tag based resource policy, "
                    "continuing to wait for tag propagation.."
                )
                continue
            raise err
    return result


def get_initial_job_state(description, status_key, wait):
    """Placeholder docstring"""
    status = description[status_key]
    job_already_completed = status in ("Completed", "Failed", "Stopped")
    return LogState.TAILING if wait and not job_already_completed else LogState.COMPLETE


def logs_init(boto_session, description, job):
    """Placeholder docstring"""
    if job == "Training":
        if "InstanceGroups" in description["ResourceConfig"]:
            instance_count = 0
            for instanceGroup in description["ResourceConfig"]["InstanceGroups"]:
                instance_count += instanceGroup["InstanceCount"]
        else:
            instance_count = description["ResourceConfig"]["InstanceCount"]
    elif job == "Transform":
        instance_count = description["TransformResources"]["InstanceCount"]
    elif job == "Processing":
        instance_count = description["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
    elif job == "AutoML":
        instance_count = 0

    stream_names = []  # The list of log streams
    positions = {}  # The current position in each stream, map of stream name -> position

    # Increase retries allowed (from default of 4), as we don't want waiting for a training job
    # to be interrupted by a transient exception.
    config = botocore.config.Config(retries={"max_attempts": 15})
    client = boto_session.client("logs", config=config)
    log_group = "/aws/sagemaker/" + job + "Jobs"

    dot = False

    color_wrap = ColorWrap()

    return instance_count, stream_names, positions, client, log_group, dot, color_wrap


def flush_log_streams(
    stream_names, instance_count, client, log_group, job_name, positions, dot, color_wrap
):
    """Placeholder docstring"""
    if len(stream_names) < instance_count:
        # Log streams are created whenever a container starts writing to stdout/err, so this list
        # may be dynamic until we have a stream for every instance.
        try:
            streams = client.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_name + "/",
                orderBy="LogStreamName",
                limit=min(instance_count, 50),
            )
            stream_names = [s["logStreamName"] for s in streams["logStreams"]]

            while "nextToken" in streams:
                streams = client.describe_log_streams(
                    logGroupName=log_group,
                    logStreamNamePrefix=job_name + "/",
                    orderBy="LogStreamName",
                    limit=50,
                )

                stream_names.extend([s["logStreamName"] for s in streams["logStreams"]])

            positions.update(
                [
                    (s, Position(timestamp=0, skip=0))
                    for s in stream_names
                    if s not in positions
                ]
            )
        except ClientError as e:
            # On the very first training job run on an account, there's no log group until
            # the container starts logging, so ignore any errors thrown about that
            err = e.response.get("Error", {})
            if err.get("Code", None) != "ResourceNotFoundException":
                raise

    if len(stream_names) > 0:
        if dot:
            print("")
            dot = False
        for idx, event in multi_stream_iter(
            client, log_group, stream_names, positions
        ):
            color_wrap(idx, event["message"])
            ts, count = positions[stream_names[idx]]
            if event["timestamp"] == ts:
                positions[stream_names[idx]] = Position(
                    timestamp=ts, skip=count + 1
                )
            else:
                positions[stream_names[idx]] = Position(
                    timestamp=event["timestamp"], skip=1
                )
    else:
        dot = True
        print(".", end="")
        sys.stdout.flush()


def rule_statuses_changed(current_statuses, last_statuses):
    """Checks the rule evaluation statuses for SageMaker Debugger and Profiler rules."""
    if not last_statuses:
        return True

    for current, last in zip(current_statuses, last_statuses):
        if (current["RuleConfigurationName"] == last["RuleConfigurationName"]) and (
            current["RuleEvaluationStatus"] != last["RuleEvaluationStatus"]
        ):
            return True

    return False


def check_job_status(job, desc, status_key_name):
    """Check to see if the job completed successfully.

    If not, construct and raise a exceptions. (UnexpectedStatusException).

    Args:
        job (str): The name of the job to check.
        desc (dict[str, str]): The result of ``describe_training_job()``.
        status_key_name (str): Status key name to check for.

    Raises:
        exceptions.CapacityError: If the training job fails with CapacityError.
        exceptions.UnexpectedStatusException: If the training job fails.
    """
    status = desc[status_key_name]
    # If the status is capital case, then convert it to Camel case
    status = STATUS_CODE_TABLE.get(status, status)

    if status == "Stopped":
        logger.warning(
            "Job ended with status 'Stopped' rather than 'Completed'. "
            "This could mean the job timed out or stopped early for some other reason: "
            "Consider checking whether it completed as you expect."
        )
    elif status != "Completed":
        reason = desc.get("FailureReason", "(No reason provided)")
        job_type = status_key_name.replace("JobStatus", " job")
        troubleshooting = (
            "https://docs.aws.amazon.com/sagemaker/latest/dg/"
            "sagemaker-python-sdk-troubleshooting.html"
        )
        message = (
            "Error for {job_type} {job_name}: {status}. Reason: {reason}. "
            "Check troubleshooting guide for common errors: {troubleshooting}"
        ).format(
            job_type=job_type,
            job_name=job,
            status=status,
            reason=reason,
            troubleshooting=troubleshooting,
        )
        if "CapacityError" in str(reason):
            raise exceptions.CapacityError(
                message=message,
                allowed_statuses=["Completed", "Stopped"],
                actual_status=status,
            )
        raise exceptions.UnexpectedStatusException(
            message=message,
            allowed_statuses=["Completed", "Stopped"],
            actual_status=status,
        )


def logs_for_job(
    model_trainer, wait=False, poll=10, log_type="All", timeout=None
):
    """Display logs for a given training job, optionally tailing them until job is complete.

    If the output is a tty or a Jupyter cell, it will be color-coded
    based on which instance the log entry is from.

    Args:
        model_trainer (sagemaker.train.ModelTrainer): The ModelTrainer used for the 
            training job
        wait (bool): Whether to keep looking for new log entries until the job completes
            (default: False).
        poll (int): The interval in seconds between polling for new log entries and job
            completion (default: 5).
        log_type ([str]): A list of strings specifying which logs to print. Acceptable
            strings are "All", "None", "Training", or "Rules". To maintain backwards
            compatibility, boolean values are also accepted and converted to strings.
        timeout (int): Timeout in seconds to wait until the job is completed. ``None`` by
            default.
    Returns:
        Last call to sagemaker DescribeTrainingJob
    Raises:
        exceptions.CapacityError: If the training job fails with CapacityError.
        exceptions.UnexpectedStatusException: If waiting and the training job fails.
    """
    sagemaker_session = model_trainer.sagemaker_session
    job_name = model_trainer._latest_training_job.training_job_name

    sagemaker_client = sagemaker_session.sagemaker_client
    request_end_time = time.time() + timeout if timeout else None
    description = wait_until(
        lambda: sagemaker_client.describe_training_job(TrainingJobName=job_name)
    )
    print(secondary_training_status_message(description, None), end="")

    instance_count, stream_names, positions, client, log_group, dot, color_wrap = logs_init(
        sagemaker_session.boto_session, description, job="Training"
    )

    state = get_initial_job_state(description, "TrainingJobStatus", wait)

    # The loop below implements a state machine that alternates between checking the job status
    # and reading whatever is available in the logs at this point. Note, that if we were
    # called with wait == False, we never check the job status.
    #
    # If wait == TRUE and job is not completed, the initial state is TAILING
    # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is
    # complete).
    #
    # The state table:
    #
    # STATE               ACTIONS                        CONDITION             NEW STATE
    # ----------------    ----------------               -----------------     ----------------
    # TAILING             Read logs, Pause, Get status   Job complete          JOB_COMPLETE
    #                                                    Else                  TAILING
    # JOB_COMPLETE        Read logs, Pause               Any                   COMPLETE
    # COMPLETE            Read logs, Exit                                      N/A
    #
    # Notes:
    # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to
    #   Cloudwatch after the job was marked complete.
    last_describe_job_call = time.time()
    last_description = description
    last_debug_rule_statuses = None
    last_profiler_rule_statuses = None

    while True:
        flush_log_streams(
            stream_names,
            instance_count,
            client,
            log_group,
            job_name,
            positions,
            dot,
            color_wrap,
        )
        if timeout and time.time() > request_end_time:
            print("Timeout Exceeded. {} seconds elapsed.".format(timeout))
            break

        if state == LogState.COMPLETE:
            break

        time.sleep(poll)

        if state == LogState.JOB_COMPLETE:
            state = LogState.COMPLETE
        elif time.time() - last_describe_job_call >= 30:
            description = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            last_describe_job_call = time.time()

            if secondary_training_status_changed(description, last_description):
                print()
                print(secondary_training_status_message(description, last_description), end="")
                last_description = description

            status = description["TrainingJobStatus"]

            if status in ("Completed", "Failed", "Stopped"):
                print()
                state = LogState.JOB_COMPLETE

            # Print prettified logs related to the status of SageMaker Debugger rules.
            debug_rule_statuses = description.get("DebugRuleEvaluationStatuses", {})
            if (
                debug_rule_statuses
                and rule_statuses_changed(debug_rule_statuses, last_debug_rule_statuses)
                and (log_type in {"All", "Rules"})
            ):
                for status in debug_rule_statuses:
                    rule_log = (
                        f"{status['RuleConfigurationName']}: {status['RuleEvaluationStatus']}"
                    )
                    print(rule_log)

                last_debug_rule_statuses = debug_rule_statuses

            # Print prettified logs related to the status of SageMaker Profiler rules.
            profiler_rule_statuses = description.get("ProfilerRuleEvaluationStatuses", {})
            if (
                profiler_rule_statuses
                and rule_statuses_changed(profiler_rule_statuses, last_profiler_rule_statuses)
                and (log_type in {"All", "Rules"})
            ):
                for status in profiler_rule_statuses:
                    rule_log = (
                        f"{status['RuleConfigurationName']}: {status['RuleEvaluationStatus']}"
                    )
                    print(rule_log)

                last_profiler_rule_statuses = profiler_rule_statuses

    if wait:
        check_job_status(job_name, description, "TrainingJobStatus")
        if dot:
            print()
        # Customers are not billed for hardware provisioning, so billable time is less than
        # total time
        training_time = description.get("TrainingTimeInSeconds")
        billable_time = description.get("BillableTimeInSeconds")
        if training_time is not None:
            print("Training seconds:", training_time * instance_count)
        if billable_time is not None:
            print("Billable seconds:", billable_time * instance_count)
            if description.get("EnableManagedSpotTraining"):
                saving = (1 - float(billable_time) / training_time) * 100
                print("Managed Spot Training savings: {:.1f}%".format(saving))
    return last_description