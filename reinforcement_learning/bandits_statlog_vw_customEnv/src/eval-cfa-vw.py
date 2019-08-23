import argparse
import json
import os
from pathlib import Path
import logging
import numpy as np

from vw_model import VWModel

from io_utils import extract_model, CSVReader, validate_experience, download_manifest_data
from vw_utils import EVAL_CHANNEL, MODEL_CHANNEL

logging.basicConfig(level=logging.INFO)


def main():
    """
    Evaluate a Vowpal Wabbit (VW) model by performing counterfactual analysis (CFA)
    """
    channel_names = json.loads(os.environ['SM_CHANNELS'])
    hyperparameters = json.loads(os.environ['SM_HPS'])
    local_mode_manifest = bool(hyperparameters.get("local_mode_manifest", False))
    num_arms = int(hyperparameters.get("num_arms", 0))
    cfa_type = hyperparameters.get("cfa_type", "dr")
    cfa_type_candidate = ["dr", "ips", "dm"]

    if num_arms is 0:
        raise ValueError("Customer Error: Please provide a non-zero value for 'num_arms'.")
    logging.info("channels %s" % channel_names)
    logging.info("hps: %s" % hyperparameters)

    # Load the model for evaluation
    model_folder = os.environ[f"SM_CHANNEL_{MODEL_CHANNEL.upper()}"]
    _, weights_path = extract_model(model_folder)
    vw_load_model_args = f"-i {weights_path}"
    vw_model = VWModel(cli_args=f"{vw_load_model_args}", 
                       model_path=None, test_only=False, quiet_mode=False)
    vw_model.start()

    # Different CFA policies in VW
    # https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Logged-Contextual-Bandit-Example
    if cfa_type not in cfa_type_candidate:
        raise ValueError(f"Customer Error: Counterfactual algorithm must be in {cfa_type_candidate}.")
    if cfa_type == "dm":
        logging.warning(f"Direct method can not be used for evaluation -- it is biased."
                       "Resetting to dr.")
        cfa_type = "dr"
    vw_cfa_args = f"--cb {num_arms} --eval --cb_type {cfa_type}"

    # Set test_only=False as VW differentiates "test" with "evaluation"
    vw_cfa = VWModel(cli_args=f"{vw_cfa_args}", test_only=False, quiet_mode=False) 
    vw_cfa.start()

    if EVAL_CHANNEL not in channel_names:
        logging.error("Evaluation channel not available. Please check container setting.")
    else:
        # Load the data for evaluation
        eval_data_dir = Path(os.environ["SM_CHANNEL_%s" % EVAL_CHANNEL.upper()])
        if local_mode_manifest:
            files = list(eval_data_dir.rglob("*"))
            if len(files) == 0:
                logging.info("No evaluation data available, aborting...")
                return
            else:
                manifest_file = files[0]
                logging.info(f"Trying to download files using manifest file {manifest_file}.")
                download_manifest_data(manifest_file, eval_data_dir)
        
        eval_files = [i for i in eval_data_dir.rglob("*") if i.is_file() and i.suffix == ".csv"]
        logging.info("Processing evaluation data: %s" % eval_files)
        
        data_reader = CSVReader(input_files=eval_files)
        data_iterator = data_reader.get_iterator()
        
        if MODEL_CHANNEL not in channel_names:
            raise ValueError("No model to be evaluated. Should at least provide current model.")
        
        # Perform counterfactual analysis
        count = 0
        for experience in data_iterator:
            is_valid = validate_experience(experience)
            if not is_valid:
                continue
            experience_context = json.loads(experience["observation"])
            predicted_action_probs = vw_model.predict(context_vector=experience_context)
            n_choices = len(predicted_action_probs)
            predicted_action = np.random.choice(n_choices, p=predicted_action_probs) + 1
            
            vw_cfa.evaluate(context_vector=experience_context,
                            action=experience["action"], 
                            cost=1 - experience["reward"],
                            probability=experience["action_prob"], 
                            label=predicted_action)
            count += 1

        vw_model.close(prediction_only=True)
        stdout = vw_cfa.close()
        print(stdout.decode())
        
        logging.info(f"Model evaluated using {count} data instances.")


if __name__ == '__main__':
    main()
