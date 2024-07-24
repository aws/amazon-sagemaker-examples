from typing import Set, Optional
import logging
import json
from fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig, DataConfig
from fmeval.exceptions import  EvalAlgorithmClientError

# Model Input/Output specify which fields FMEVal looks in our dataset.
# Reference https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate-auto-lib-custom.html
DATASET_NAME = "custom_dataset"
DATASET_MIME_TYPE = "application/jsonlines"
MODEL_INPUT_LOCATION = "content"
MODEL_OUTPUT_LOCATION = "answer"


TOXICITY_EVALUATOR_MODEL = "detoxify"
DEFAULT_EVALUATIONS = {'toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit'}

logger = logging.getLogger(__name__)

class Evaluator:
    """
    The Evaluator is a service that assesses the performance of Large Language Models by running a set 
    of evaluation algorithms specified by a configuration set. It reads formatted data from 
    the /opt/ml/processing/output/data.jsonl file and uses the FMEval open-source library to 
    execute the specified evaluation tasks.
    """
    def __init__(self, eval_config: Optional[Set[str]] = None):
        """
        Constructor
        :param eval_config: A Set of evaluation tasks to run. If not provided, all evaluation tasks will be run.
        :raises: ValueError if eval_config is not a set or a list of strings.
        """
        self.eval_config = eval_config
        if eval_config is not None:
            if isinstance(eval_config, set):
                self.eval_config = eval_config
            elif isinstance(eval_config, list):
                self.eval_config = set(eval_config)
            else:
                raise ValueError("eval_config must be a set or a list of strings")

    def evaluate(self, dataset_uri: str):
        """
        Evaluate the data using the configured settings.

        :param dataset_uri: The path to the dataset file.
        :raises: ValueError if the dataset_uri is not a valid string.
        :return: A dictionary containing the evaluation results. If data is empty/malformed, returns an empty dictionary.
        """

        if not isinstance(dataset_uri, str):
            raise ValueError("dataset_uri must be a valid string")

        config = DataConfig(
            dataset_name=DATASET_NAME,
            dataset_uri=dataset_uri,
            dataset_mime_type=DATASET_MIME_TYPE,
            model_input_location=MODEL_INPUT_LOCATION,
            model_output_location=MODEL_OUTPUT_LOCATION,
        )

        if not self.eval_config:
            configured_evals = DEFAULT_EVALUATIONS
        else:
            configured_evals = set(self.eval_config)

        eval_algo = Toxicity(ToxicityConfig(model_type=TOXICITY_EVALUATOR_MODEL))
        
        try:
            eval_output = eval_algo.evaluate(dataset_config=config, save=True)
        except (json.JSONDecodeError, EvalAlgorithmClientError) as e:
            # If we evaluate an empty/malformed file, return an empty dict
            logger.warning("Evaluated data malformed.")
            return {}
             
        eval_results = {}
        for eval_score in eval_output[0].dataset_scores:
                if eval_score.name in configured_evals:
                    eval_results[eval_score.name] = eval_score.value

        logger.info(f"Evaluation Results: {eval_results}")

        return eval_results
        