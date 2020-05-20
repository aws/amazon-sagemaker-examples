import argparse
import json
import os
from pathlib import Path
import logging

from vw_model import VWModel

from io_utils import extract_model, CSVReader, validate_experience
from vw_utils import TRAIN_CHANNEL, MODEL_CHANNEL, MODEL_OUTPUT_PATH, save_vw_model, save_vw_metadata

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    """ Train a Vowpal Wabbit (VW) model through C++ process. """
    
    channel_names = json.loads(os.environ['SM_CHANNELS'])
    hyperparameters = json.loads(os.environ['SM_HPS'])
    num_arms = int(hyperparameters.get("num_arms", 0))
    num_policies = int(hyperparameters.get("num_policies", 3))
    exploration_policy = hyperparameters.get("exploration_policy", "egreedy").lower()
    epsilon = float(hyperparameters.get("epsilon", 0))

    if num_arms is 0:
        raise ValueError("Customer Error: Please provide a non-zero value for 'num_arms'")
    logging.info("channels %s" % channel_names)
    logging.info("hps: %s" % hyperparameters)

    # Different exploration policies in VW
    # https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    valid_policies = ["egreedy", "bag", "cover"]
    if exploration_policy not in valid_policies:
        raise ValueError(f"Customer Error: exploration_policy must be one of {valid_policies}.")
    
    if exploration_policy == "egreedy":
        vw_args_base = f"--cb_explore {num_arms} --epsilon {epsilon}"
    else:
        vw_args_base = f"--cb_explore {num_arms} --{exploration_policy} {num_policies}"

    # No training data. Initialize and save a random model
    if TRAIN_CHANNEL not in channel_names:
        logging.info("No training data found. Saving a randomly initialized model!")
        vw_model = VWModel(cli_args=f"{vw_args_base} -f {MODEL_OUTPUT_PATH}",
                           model_path=None, test_only=False, quiet_mode=False)
        vw_model.start()
        vw_model.close()
        save_vw_metadata(meta=vw_args_base)
    
    # If training data is present
    else:
        if MODEL_CHANNEL not in channel_names:
            logging.info(f"No pre-trained model has been specified in channel {MODEL_CHANNEL}."
                         f"Training will start from scratch.")
            vw_args = f"{vw_args_base}"
        else:
            # Load the pre-trained model for training.
            model_folder = os.environ[f'SM_CHANNEL_{MODEL_CHANNEL.upper()}']
            _, weights_path = extract_model(model_folder)
            logging.info(f"Loading model from {weights_path}")
            vw_args = f"{vw_args_base} -i {weights_path}"
        
        # Init a class that communicates with C++ VW process using pipes
        vw_model = VWModel(cli_args=f"{vw_args} -f {MODEL_OUTPUT_PATH} --save_resume",
                           model_path=None, test_only=False, quiet_mode=False)
        vw_model.start()

        # Load training data
        training_data_dir = Path(os.environ["SM_CHANNEL_%s" % TRAIN_CHANNEL.upper()])
        training_files = [i for i in training_data_dir.rglob("*") if i.is_file() and i.suffix == ".csv"]
        logging.info("Processing training data: %s" % training_files)

        data_reader = CSVReader(input_files=training_files)
        data_iterator = data_reader.get_iterator()

        count = 0
        for experience in data_iterator:
            is_valid = validate_experience(experience)
            if not is_valid:
                continue
            vw_model.learn(context_vector=json.loads(experience["observation"]),
                           action=experience["action"],
                           cost=1 - experience["reward"],
                           probability=experience["action_prob"])
            count += 1
        
        stdout = vw_model.close()
        print(stdout.decode())
        save_vw_metadata(meta=vw_args_base)
        logging.info(f"Model learned using {count} training experiences.")


if __name__ == '__main__':
    main()
