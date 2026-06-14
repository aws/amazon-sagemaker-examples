import logging
import sys
import site
from components.data_loader import DataLoader
from components.evaluator import Evaluator
from components.cloudwatch_logger import CloudWatchLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This is where our capture data is loaded to. MUST be same as "destination" field in EndointInput for deployed model.
INPUT_DATA_SOURCE = '/opt/ml/processing/input_data' 

# Destination for formatted and cleaned data in the container for evaluation.
CLEANED_DATA_DESTINATION = '/opt/ml/processing/internal/data.jsonl'

# Destination for metrics. These metrics MUST be stored at this location if they are to be published.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-cloudwatch.html
CLOUDWATCH_METRICS_DESTINATION = '/opt/ml/output/metrics/cloudwatch/cloudwatch_metrics.jsonl'

# These are all of the evaluations we can run. 
EVALUATIONS = {
        "toxicity", 
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
        "sexual_explicit"
        }

if __name__ == "__main__":
    try:
        data_loader = DataLoader()
        evaluator = Evaluator(EVALUATIONS)
        cloudwatch_logger = CloudWatchLogger()
        
        data_loader.execute_etl(INPUT_DATA_SOURCE, CLEANED_DATA_DESTINATION)
        eval_results = evaluator.evaluate(CLEANED_DATA_DESTINATION)
        cloudwatch_logger.log(eval_results, CLOUDWATCH_METRICS_DESTINATION)
                                
    except Exception as e:
        logger.exception("Exception performing analysis: " + str(e))
        sys.exit(255)
