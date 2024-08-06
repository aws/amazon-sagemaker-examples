from typing import Dict
import logging
import json
import datetime
import os

logger = logging.getLogger(__name__)

PROCESSING_JOB_CONFIG_FILE = '/opt/ml/config/processingjobconfig.json'

DEFAULT_ENDPOINT_AND_MONITORING_SCHEDULE = ('byoc_llm_default_endpoint', 'byoc_llm_default_monitoring_schedule')


class CloudWatchLogger:
    """
    The CloudWatchLogger is a service that writes evaluation metrics to CloudWatch.
    """

    def __init__(self):
        """
        Constructor.
        """

    def log(self, eval_results: Dict, destination: str):
        """
        Log the evaluation results to CloudWatch.
        :param eval_results: A dictionary of evaluation results.
        :param destination: The path to the file where the evaluation results will be written.
        :raises: ValueError if eval_results is not a dictionary.

        For formatting and other information, see here: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-cloudwatch.html
        """

        if eval_results is not None and not isinstance(eval_results, dict):
                raise ValueError("eval_results must be a dictionary")
        
        
        now = datetime.datetime.now(datetime.timezone.utc)
        metric_timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")


        endpoint_name, monitoring_schedule_name = get_endpoint_and_monitoring_schedule()
        logger.info(f"Endpoint: {endpoint_name}, Monitoring Schedule: {monitoring_schedule_name}")

        # Create the output directory if it doesn't exist
        formatted_data_dir = os.path.dirname(destination)
        if not os.path.exists(formatted_data_dir):
            os.makedirs(formatted_data_dir, exist_ok=True)

        try:
            with open(destination, 'w') as file:
                for metric_name, metric_value in eval_results.items():
                    metric_data = {
                        "MetricName": metric_name,
                        "Timestamp": metric_timestamp,
                        "Dimensions": [
                            {"Name": "Endpoint", "Value": endpoint_name},
                            {"Name": "MonitoringSchedule", "Value": monitoring_schedule_name} 
                        ],
                        "Value": metric_value
                    }
                    file.write(json.dumps(metric_data) + '\n')

                    logger.info(f"Logged metrics: {json.dumps(metric_data)}")
                    logger.info(f"Logged to {destination}")
        except PermissionError as e:
            logger.warning(f"Unable to write to {destination}")
            print(f"Error: {e}")

        print(f"Evaluation results logged to: {destination}")
    

def is_running_in_docker():
    """
    Checks whether we are running in a Docker container or not.
    :returns True if DOCKER_CONTAINER env variable is present, False otherwise.
    """
    return 'DOCKER_CONTAINER' in os.environ


def get_endpoint_and_monitoring_schedule():
    """
    Retrieves the endpoint name and monitoring schedule name from the processing job config file.
    If we are in a docker container, we are running a monitoring job, and the config file has 
    the endpoint name and monitoring schedule name.

    For information about processingjobcongfig.json file, see here: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-contract-inputs.html

    :returns A tuple containing the endpoint name and monitoring schedule name.
    """

    if is_running_in_docker():
        try:
            with open(PROCESSING_JOB_CONFIG_FILE, 'r') as config:
                params = json.load(config)
                logger.info("Reading Env params")
                endpoint_name = params["Environment"]["sagemaker_endpoint_name"]
                monitoring_schedule_name = params["Environment"]["sagemaker_monitoring_schedule_name"]

            return endpoint_name, monitoring_schedule_name
        except KeyError:
            logger.error(f"Environment does not have endpoint or monitoring schedule name. Ensure that this processing job is initiated by a monitoring schedule.")
            return DEFAULT_ENDPOINT_AND_MONITORING_SCHEDULE
        
    else:
        return DEFAULT_ENDPOINT_AND_MONITORING_SCHEDULE