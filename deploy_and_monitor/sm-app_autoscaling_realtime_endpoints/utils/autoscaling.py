import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

from rich import print
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


# Function to update the user prompt in the messages list
def update_user_prompt(messages, prompt):
    for message in messages:
        if message["role"] == "user":
            message["content"] = prompt
    return messages


# helper function to record latency
def get_request_latency(payload, endpoint_name, sagemaker_runtime_client):
    start_time = time.time()
    _ = sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    # _ = predictor.predict(payload)
    end_time = time.time()
    latency = end_time - start_time
    # print(chat["choices"][0]["message"]["content"].strip())
    return latency


# Function to test concurrent requests with a given concurrency level
def test_concurrency_level(
    concurrency_level,
    prompts,
    messages,
    parameters,
    endpoint_name,
    sagemaker_runtime_client,
):
    payloads = [
        {"messages": update_user_prompt(messages, prompt), **parameters}
        for prompt in prompts * (concurrency_level // len(prompts))
    ]
    latencies = []
    with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [
            executor.submit(
                get_request_latency, payload, endpoint_name, sagemaker_runtime_client
            )
            for payload in payloads
        ]
        for future in as_completed(futures):
            try:
                latency = future.result()
                latencies.append(latency)
            except Exception as e:
                print(f"Request failed: {e}")

    avg_latency = mean(latencies)
    return avg_latency


# helper function to get the current instance count of the endpoint
def get_scaling_instance_counts(endpoint_name, sagemaker_client):
    endpoint_description = sagemaker_client.describe_endpoint(
        EndpointName=endpoint_name
    )
    current = endpoint_description["ProductionVariants"][0]["CurrentInstanceCount"]
    desired = endpoint_description["ProductionVariants"][0]["DesiredInstanceCount"]
    current_status = endpoint_description["EndpointStatus"]
    return current, desired, current_status


# Helper function to check if any alarm is in "InAlarm" state
def is_alarm_in_alarm_state(alarm_name, cloudwatch_client):
    alarm_state = cloudwatch_client.describe_alarms(AlarmNames=[alarm_name])[
        "MetricAlarms"
    ][0]["StateValue"]
    if alarm_state == "ALARM":
        return True
    return False


# Helper function to monitor the endpoint for scaling events
def monitor_scaling_events(
    endpoint_name, alarm_name, time_to_sleep, cloudwatch_client, sagemaker_client
):
    scaling_times = {}
    (
        current_instance_count,
        desired_instance_count,
        status,
    ) = get_scaling_instance_counts(endpoint_name, sagemaker_client)
    print(f"Initial instance count: {current_instance_count}", flush=True)
    print(f"Tracking Alarm: [i green]{alarm_name}[/i green]", flush=True)

    with Progress(
        SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()
    ) as progress:
        alarm_task = progress.add_task(
            "[green]Waiting for alarm to trigger...", total=None
        )

        alarm_timer_start = time.time()

        while True:
            if is_alarm_in_alarm_state(alarm_name, cloudwatch_client):
                start_time = time.time()
                alarm_timer_end = time.time()
                time_to_alarm = alarm_timer_end - alarm_timer_start
                progress.update(
                    alarm_task,
                    description=f"[bold red]Alarm triggered! Time to alarm trigger: {time_to_alarm:.2f} seconds.",
                    total=1,
                    completed=1,
                )
                # print(f"[bold red]Alarm triggered! Time to alarm trigger: {time_to_alarm:.2f} seconds.")
                break
            else:
                progress.update(alarm_task, advance=1)
                # Wait for time_to_sleep seconds before checking again
                time.sleep(time_to_sleep)

        scaling_task = progress.add_task(
            "[green]Waiting for scaling to complete...", total=None
        )

        while True:
            (
                current_instance_count,
                desired_instance_count,
                status,
            ) = get_scaling_instance_counts(endpoint_name, sagemaker_client)

            if current_instance_count == desired_instance_count:
                # Add sleep here as endpoint status doesn't change to `Updating` instantaneously
                time.sleep(time_to_sleep)
                if status == "InService":
                    end_time = time.time()
                    scaling_time = end_time - start_time
                    scaling_times[desired_instance_count] = scaling_time
                    progress.update(
                        scaling_task,
                        description=f"[bold green]Scaling to {desired_instance_count} instances completed in {scaling_time:.2f} seconds.",
                        total=1,
                        completed=1,
                    )
                    break
            progress.update(scaling_task, advance=1)
            # Wait for time_to_sleep seconds before checking again
            time.sleep(time_to_sleep)

    return scaling_times


# function to print scaling times in a table
def print_scaling_times(scaling_times):
    # Create a table
    table = Table(title="Scaling Times")

    # Add columns
    table.add_column(
        "Target Instance Count", justify="right", style="cyan", no_wrap=True
    )
    table.add_column("Scaling Time (seconds)", justify="right", style="magenta")

    # Add rows
    for target_instance_count, scaling_time in scaling_times.items():
        table.add_row(str(target_instance_count), f"{scaling_time:.2f}")

    return table
