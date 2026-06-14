import logging
import sys
import site
import json
import os
from components.data_loader import DataLoader
from components.evaluator import Evaluator
from components.cloudwatch_logger import CloudWatchLogger
from langkit import textstat
from whylogs.experimental.core.udf_schema import udf_schema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This is where our capture data is loaded to. MUST be same as "destination" field in EndointInput for deployed model.
INPUT_DATA_SOURCE = '/opt/ml/processing/input_data' 

# Destination for formatted and cleaned data in the container for evaluation.
CLEANED_DATA_DESTINATION = '/opt/ml/processing/internal/data.jsonl'

# Destination for metrics. These metrics MUST be stored at this location if they are to be published.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-cloudwatch.html
CLOUDWATCH_METRICS_DESTINATION = '/opt/ml/output/metrics/cloudwatch/cloudwatch_metrics.jsonl'

PROCESSING_JOB_CONFIG_FILE = '/opt/ml/config/processingjobconfig.json'

DEFAULT_EVAL_LIST = {"TOXICITY", "READABILITY", "RELEVANCE_AND_ACCURACY"}

def get_evaluations():
    """
    Retrieves the specified evaluations from the processing job config file.
    If we are in a docker container, we are running a monitoring job, and the config file has 
    the endpoint name and monitoring schedule name.

    For information about processingjobcongfig.json file, see here: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-contract-inputs.html

    :returns A tuple containing the endpoint name and monitoring schedule name.
    """

    if 'DOCKER_CONTAINER' in os.environ:
        try:
            with open(PROCESSING_JOB_CONFIG_FILE, 'r') as config:
                params = json.load(config)
                logger.info("Reading Env params")
                eval_list = set()

                if params["Environment"]["TOXICITY"] == "Enabled":
                    eval_list.add("TOXICITY")
                if params["Environment"]["READABILITY"] == "Enabled":
                    eval_list.add("READABILITY")
                if params["Environment"]["RELEVANCE_AND_ACCURACY"] == "Enabled":
                    eval_list.add("RELEVANCE_AND_ACCURACY")

            return eval_list
        except KeyError as e:
            logger.error(f"Environment does not have any evaluations enables.")
            raise e  
    else:
        return DEFAULT_EVAL_LIST

if __name__ == "__main__":

    try:
        evaluations = get_evaluations()
        data_loader = DataLoader()
        evaluator = Evaluator(eval_config=evaluations)
        cloudwatch_logger = CloudWatchLogger()
        
        data_loader.execute_etl(INPUT_DATA_SOURCE, CLEANED_DATA_DESTINATION)
        eval_results = evaluator.evaluate(CLEANED_DATA_DESTINATION)
        cloudwatch_logger.log(eval_results, CLOUDWATCH_METRICS_DESTINATION)
                                
    except Exception as e:
        logger.exception("Exception performing analysis: " + str(e))
        sys.exit(255)
