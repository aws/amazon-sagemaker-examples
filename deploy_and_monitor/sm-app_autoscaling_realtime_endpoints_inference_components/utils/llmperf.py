import glob
import json
import os
import subprocess
import time

from rich import box, print
from rich.table import Table


# LLMPerf requires AWS Creds as ENV variables along with endpoint name
def trigger_auto_scaling(creds, region, endpoint_name, num_concurrent_requests):
    # Set environment variables
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    os.environ["AWS_SESSION_TOKEN"] = creds.token
    os.environ["AWS_REGION"] = region
    os.environ["EP_NAME"] = endpoint_name
    os.environ["NUM_CONCURRENT_REQUESTS"] = str(num_concurrent_requests)

    # Path to the shell script
    # script_path = "./trigger_autoscaling.sh"
    # current_dir = os.getcwd()
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "trigger_autoscaling.sh")
    )

    # print(f"Current working directory: {current_dir}")
    # print(f"Full path to script: {script_path}")

    # Check if the file exists
    if os.path.exists(script_path):
        print(f"Calling LLMPerf shell script: {script_path}")
    else:
        print(f"LLMPerf shell script file not found at {script_path}")

    # Make sure the script is executable
    # os.chmod(script_path, 0o755)

    # Run the shell script
    print(f"Launching LLMPerf with {num_concurrent_requests} concurrent requests")
    process = subprocess.Popen([script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return process


# helper function to monitor the process
def monitor_process(proc):
    while True:
        retcode = proc.poll()  # Check if the process has terminated
        if retcode is not None:
            # Process has terminated
            print(f"Process {proc.pid} finished with return code {retcode}")

            # Capture and print any output from the process
            stdout, stderr = proc.communicate()
            if stdout:
                print(f"Process output:\n{stdout.decode('utf-8')}")
            if stderr:
                print(f"Process errors:\n{stderr.decode('utf-8')}")

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
    perf_table.add_row("Avg. First-Time-To-Token", f"{data['results_ttft_s_mean']*1000:.2f}ms")
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
