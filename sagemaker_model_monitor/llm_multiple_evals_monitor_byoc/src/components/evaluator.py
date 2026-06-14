from typing import Set, Optional
import logging
from langkit import light_metrics, extract
from fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig, DataConfig
from fmeval.exceptions import  EvalAlgorithmClientError
from langchain_community.llms.gpt4all import GPT4All
from gpt4all import GPT4All as fileDownloader
from langchain.evaluation.scoring import ScoreStringEvalChain
import json
from json import JSONDecodeError
from typing import Any, Callable, Optional, Sequence, Tuple
import re
import os
import random

# Model Input/Output specify which fields FMEVal looks in our dataset.
# Reference https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-foundation-model-evaluate-auto-lib-custom.html
DATASET_NAME = "custom_dataset"
DATASET_MIME_TYPE = "application/jsonlines"
MODEL_INPUT_LOCATION = "content"
MODEL_OUTPUT_LOCATION = "answer"


TOXICITY_EVALUATOR_MODEL = "detoxify"
DEFAULT_EVALUATIONS = {'toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit'}

DEFAULT_REPORT_PATH = './tests/output'
READABILITY_REPORT_FILENAME = 'readability_eval_results.jsonl'
RELEVANCE_AND_ACCURACY_REPORT_FILENAME = 'relevance_and_accuracy_eval_results.jsonl'
REPORT_PATH = os.getenv("EVAL_RESULTS_PATH") if "EVAL_RESULTS_PATH" in os.environ else DEFAULT_REPORT_PATH

# These are all of the readability evaluations we can run. 
READABILITY_EVALUATIONS = {
        "flesch_reading_ease",
        "automated_readability_index",
        "aggregate_reading_level",
        "syllable_count",
        "lexicon_count",
        "sentence_count",
        "character_count",
        "letter_count",
        "polysyllable_count",
        "monosyllable_count",
        "difficult_words",
        }

# These are all of the toxicity evaluations we can run. 
TOXICITY_EVALUATIONS = {
        "toxicity", 
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
        "sexual_explicit"
        }

RELEVANCE_AND_ACCURACY_EVALUATIONS = {
    "relevance_and_accuracy_score"
}

ANSWER_RELEVANCY_MODEL = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

DEFAULT_EVALUATIONS = {"TOXICITY", "READABILITY", "RELEVANCE_AND_ACCURACY"}

logger = logging.getLogger(__name__)

class Evaluator:
    """
    The Evaluator is a service that assesses the performance of Large Language Models by running a set 
    of evaluation algorithms specified by a configuration set. It reads formatted data from 
    the /opt/ml/processing/output/data.jsonl file and uses the FMEval open-source library to 
    execute the specified evaluation tasks.
    """
    def __init__(self, eval_config: Optional[Set[str]] = DEFAULT_EVALUATIONS):
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
        
        if not isinstance(REPORT_PATH, str):
            raise ValueError("report_path must be a valid string")

        toxicity_results = {}
        readability_results = {}
        relevance_and_accuracy_results = {}
        if "TOXICITY" in self.eval_config:
            toxicity_results = self._evaluate_toxicity(dataset_uri)
        
        if "READABILITY" in self.eval_config:
            readability_results = self._evaluate_readability(dataset_uri)

        if "RELEVANCE_AND_ACCURACY" in self.eval_config:
            relevance_and_accuracy_results = self._evaluate_relevance_and_accuracy(dataset_uri)

        return {**toxicity_results, **readability_results, **relevance_and_accuracy_results}


    def _evaluate_toxicity(self, dataset_uri: str):
        """
        Evaluates the data for Toxicity using the FMEval library.

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

        eval_algo = Toxicity(ToxicityConfig(model_type=TOXICITY_EVALUATOR_MODEL))
        
        try:
            eval_output = eval_algo.evaluate(dataset_config=config, save=True)
        except (json.JSONDecodeError, EvalAlgorithmClientError) as e:
            # If we evaluate an empty/malformed file, return an empty dict
            logger.warning("Evaluated data malformed.")
            return {}
             
        eval_results = {}
        for eval_score in eval_output[0].dataset_scores:
                eval_results[eval_score.name] = eval_score.value

        logger.info(f"Evaluation Results: {eval_results}")

        return eval_results
    

    def _evaluate_readability(self, dataset_uri: str):
        """
        Evaluates the data for readability using the WhyLabs Langkit Library.

        :param dataset_uri: The path to the dataset file.
        :raises: ValueError if the dataset_uri is not a valid string.
        :return: A dictionary containing the evaluation results. If data is empty/malformed, returns an empty dictionary.
        """
        
        text_schema = light_metrics.init()

        line_count = 0
        try:
            with open(dataset_uri, 'r') as file:
                lines = file.readlines()
        except:
            logger.error("Could not read file.")
            return {}
        
        if len(lines) == 0:
            logger.info("No data to evaluate")
            return {}
        
        results = []
        totals = {field: 0 for field in READABILITY_EVALUATIONS}

        if len(lines) <= 100:
            sample_lines = lines
        else:
            sample_lines = random.sample(lines, 100)

        for line in sample_lines:
            try:
                data = json.loads(line)
                line_count += 1

                readability_evals = clean_readability_dict(extract({"prompt": data['answer']}, schema=text_schema))
                result_dict = {
                    "prompt": data["content"],
                    "response": data["answer"],
                    **readability_evals,
                }
                results.append(result_dict)
                for key, value in result_dict.items():
                    if key in totals:
                        totals[key] += value
            except (KeyError, JSONDecodeError) as e:
                logger.error(f"Data malformed. {e}")
                return {}

        report_filepath = os.path.join(REPORT_PATH, READABILITY_REPORT_FILENAME)

        logger.info(f"Writing readability evaluation results to {report_filepath}")
        write_eval_result_file(report_filepath, results)

        return {key: value / (line_count if line_count > 0 else 1) for key, value in totals.items()}
    
    def _evaluate_relevance_and_accuracy(self, dataset_uri: str):
        """
        Evaluates the data for relevance and accuracy using the FMEval library.

        :param dataset_uri: The path to the dataset file.
        :raises: ValueError if the dataset_uri is not a valid string.
        :return: A dictionary containing the evaluation results. If data is empty/malformed, returns an empty dictionary.
        """

        if not isinstance(dataset_uri, str):
            raise ValueError("dataset_uri must be a valid string")
        

        fileDownloader.retrieve_model(ANSWER_RELEVANCY_MODEL) # downloads / loads a 4.66GB LLM
        model = GPT4All(model=ANSWER_RELEVANCY_MODEL, verbose=False, n_batch=128, n_threads=36 if 'DOCKER_CONTAINER' in os.environ else None)
        evaluator_model = ScoreStringEvalChain.from_llm(
            llm=model, verbose=False
        )

        line_count = 0
        try:
            with open(dataset_uri, 'r') as file:
                lines = file.readlines()
        except:
            logger.error("Could not read file.")
            return {}
        
        if not lines:
            logger.info("No data to evaluate")
            return {}
        
        # Initialize our list of individualy response scores and summed total scores (for later averaging)
        results = []
        totals = {field: 0 for field in RELEVANCE_AND_ACCURACY_EVALUATIONS}
        # Randomly sample 10 prompt and responses for evaluation
        if len(lines) <= 10:
            sample_lines = lines
        else:
            sample_lines = random.sample(lines, 10)

        logger.info("Starting evaluation")
        for line in sample_lines:  
            try:
                data = json.loads(line)
                line_count += 1
                logger.info(f"Evaluating line: {line_count}")
                
                accuracy_relevance_eval_result = evaluator_model.evaluate_strings(
                    prediction=data["answer"],
                    input=data["content"],
                )

                result_dict = {
                    "prompt": data["content"],
                    "response": data["answer"],
                    "relevance_and_accuracy_analysis": accuracy_relevance_eval_result["reasoning"],
                    "relevance_and_accuracy_score": accuracy_relevance_eval_result["score"],
                }
                # Add all scores for this response to result list and sum total scores
                results.append(result_dict)
                for key, value in result_dict.items():
                    if key in totals:
                        totals[key] += value
            except ValueError as e:
                logger.warning(f"Error evaluating line, continuing: {e}")
                continue
            except (KeyError, JSONDecodeError) as e:
                logger.error(f"Data malformed {e}")
                return {}

        report_filepath = os.path.join(REPORT_PATH, RELEVANCE_AND_ACCURACY_REPORT_FILENAME)
        write_eval_result_file(report_filepath, results)

        # Returns average scores
        return {key: value / (line_count if line_count > 0 else 1) for key, value in totals.items()}
        

def clean_readability_dict(evals):
    """
    Cleans the readability dictionary by removing the 'prompt' and 'has_patterns' keys. Also, removes 'prompt.' prefix from fields which is 
    the default behavior of the LangKit extract function.
    :param evals: The dictionary to clean.
    :return: The cleaned dictionary.
    """
    evals.pop('prompt')

    # Remove 'prompt.' from every key
    new_evals = {}
    for key, value in evals.items():
        new_key = key.replace('prompt.', '')
        new_evals[new_key] = value

    try:
        new_evals.pop('has_patterns')
    except:
        logger.info("No patterns found")
        
    return new_evals

def write_eval_result_file(report_filepath, results):
    """
    Writes the evaluation results to a file in the specified directory.
    :param formatted_data_dir: The directory to write the file to.
    :param report_path: The directory to write the file to
    :param results: The evaluation results to write.
    :return: None
    """
    formatted_data_dir = os.path.dirname(report_filepath)
    os.makedirs(formatted_data_dir, exist_ok=True)
    with open(report_filepath, 'w') as output_file:
        for result_dict in results:
            output_file.write(json.dumps(result_dict) + '\n')