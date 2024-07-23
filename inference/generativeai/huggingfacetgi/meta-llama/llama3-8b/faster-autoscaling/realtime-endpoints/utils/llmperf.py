import glob
import json
import os
import subprocess
import time

from rich import box, print
from rich.table import Table


# LLMPerf requires you to pass AWS Creds as ENV variables along with endpoint name as an arg
def trigger_auto_scaling(creds, region, endpoint_name, num_concurrent_requests):
    aws_access_key = creds.access_key
    aws_secret_key = creds.secret_key
    aws_session_token = creds.token

    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
    os.environ["AWS_SESSION_TOKEN"] = aws_session_token
    os.environ["AWS_REGION"] = f"{region}"
    os.environ["EP_NAME"] = endpoint_name

    # Define the command to run the traffic generation script
    command = f"""
    echo "Installing llmperf..."
    rm -rf llmperf && \
    git clone https://github.com/philschmid/llmperf.git && \
    uv pip install -e llmperf/

    DIR="results"

    if [ ! -d "$DIR" ]; then
      mkdir -p "$DIR"
      echo "Created $DIR directory."
    else
      echo "$DIR directory already exists."
    fi

    echo "Starting benchmarking scripts on endpoint {endpoint_name} ..."

    start_time=`date +%s`

    MESSAGES_API=true python llmperf/token_benchmark_ray.py \
    --model {endpoint_name} \
    --llm-api "sagemaker" \
    --max-num-completed-requests 1000 \
    --timeout 600 \
    --num-concurrent-requests {num_concurrent_requests} \
    --results-dir "results"

    end_time=`date +%s`
    echo execution time was `expr $end_time - $start_time` secs.
    """

    # print(command)

    # Run the command in the background; pass env variables to the shell
    print(f"Launching LLMPerf with {num_concurrent_requests} concurrent requests")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )

    # Print the process ID
    # print(f"Started background job with PID: {process.pid}")
    return process


# helper function to monitor the process
def monitor_process(proc):
    while True:
        retcode = proc.poll()  # Check if the process has terminated
        if retcode is not None:
            # Process has terminated
            print(f"Process {proc.pid} finished with return code {retcode}")
            break
        else:
            # Process is still running
            print(f"Process {proc.pid} is still running...")
            time.sleep(15)  # Check every 15 seconds


# helper function to print llmperf results
def print_llmperf_results(num_concurrent_requests):
    # Reads the summary.json file and prints the results
    with open(glob.glob("results/*summary.json")[0], "r") as file:
        data = json.load(file)

    # Create a table
    perf_table = Table(
        title="LLMPerf Endpoint Metrics",
        row_styles=["bold", "bold"],
        box=box.MINIMAL_DOUBLE_HEAD,
    )
    # Add columns
    perf_table.add_column("Metric", justify="right", style="green", no_wrap=True)
    perf_table.add_column("Units", justify="left", style="magenta")

    # Add rows
    perf_table.add_row("Concurrent requests", f"{num_concurrent_requests}")
    perf_table.add_row("Avg. Input token length", f"{data['mean_input_tokens']}")
    perf_table.add_row("Avg. Output token length", f"{data['mean_output_tokens']}")
    perf_table.add_row(
        "Avg. First-Time-To-Token", f"{data['results_ttft_s_mean']*1000:.2f}ms"
    )
    perf_table.add_row(
        "Avg. Thorughput",
        f"{data['results_mean_output_throughput_token_per_s']:.2f} tokens/sec",
    )
    perf_table.add_row(
        "Avg. Latency", f"{data['results_inter_token_latency_s_mean']*1000:.2f}ms/token"
    )

    # Print the table
    # console.print(perf_table)
    return perf_table
