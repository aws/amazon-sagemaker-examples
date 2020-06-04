import json
import logging
import os

from env import MovieLens100KEnv
from io_utils import extract_model
from vw_agent import VWAgent
from vw_utils import MODEL_CHANNEL, MODEL_OUTPUT_DIR, DATA_OUTPUT_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    """ Train a Vowpal Wabbit (VW) model through C++ process. """

    channel_names = json.loads(os.environ['SM_CHANNELS'])
    hyperparameters = json.loads(os.environ['SM_HPS'])

    # Fetch algorithm hyperparameters
    num_arms = int(hyperparameters.get("num_arms", 0))  # Used if arm features are not present
    num_policies = int(hyperparameters.get("num_policies", 3))
    exploration_policy = hyperparameters.get("exploration_policy", "egreedy").lower()
    epsilon = float(hyperparameters.get("epsilon", 0))
    mellowness = float(hyperparameters.get("mellowness", 0.01))
    arm_features_present = bool(hyperparameters.get("arm_features", True))

    # Fetch environment parameters
    item_pool_size = int(hyperparameters.get("item_pool_size", 0))
    top_k = int(hyperparameters.get("top_k", 5))
    max_users = int(hyperparameters.get("max_users", 3))
    total_interactions = int(hyperparameters.get("total_interactions", 1000))

    if not arm_features_present and num_arms is 0:
        raise ValueError("Customer Error: Please provide a non-zero value for 'num_arms'")
    logging.info("channels %s" % channel_names)
    logging.info("hps: %s" % hyperparameters)

    # Different exploration policies in VW
    # https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    valid_policies = ["egreedy", "bag", "regcbopt", "regcb"]
    if exploration_policy not in valid_policies:
        raise ValueError(f"Customer Error: exploration_policy must be one of {valid_policies}.")

    if exploration_policy == "egreedy":
        vw_args_base = f"--cb_explore_adf --cb_type mtr --epsilon {epsilon}"
    elif exploration_policy in ["regcbopt", "regcb"]:
        vw_args_base = f"--cb_explore_adf --cb_type mtr --{exploration_policy} --mellowness {mellowness}"
    else:
        vw_args_base = f"--cb_explore_adf --cb_type mtr --{exploration_policy} {num_policies}"

    # If pre-trained model is present
    if MODEL_CHANNEL not in channel_names:
        logging.info(f"No pre-trained model has been specified in channel {MODEL_CHANNEL}."
                     f"Training will start from scratch.")
        vw_agent = VWAgent(cli_args=vw_args_base,
                           output_dir=MODEL_OUTPUT_DIR,
                           model_path=None,
                           test_only=False,
                           quiet_mode=False,
                           adf_mode=arm_features_present,
                           num_actions=num_arms)
    else:
        # Load the pre-trained model for training.
        model_folder = os.environ[f'SM_CHANNEL_{MODEL_CHANNEL.upper()}']
        metadata_path, weights_path = extract_model(model_folder)
        logging.info(f"Loading model from {weights_path}")
        vw_agent = VWAgent.load_model(metadata_loc=metadata_path,
                                      weights_loc=weights_path,
                                      test_only=False,
                                      quiet_mode=False,
                                      output_dir=MODEL_OUTPUT_DIR)

    # Start the VW C++ process. This python program will communicate with the C++ process using PIPES
    vw_agent.start()

    if "movielens" not in channel_names:
        raise ValueError(
            "Cannot find `movielens` channel. Please make sure to provide the data as `movielens` channel.")

    # Initialize MovieLens environment
    env = MovieLens100KEnv(data_dir=os.environ['SM_CHANNEL_MOVIELENS'],
                           item_pool_size=item_pool_size,
                           top_k=top_k,
                           max_users=max_users)

    regrets = []
    random_regrets = []

    obs = env.reset()

    # Learn by interacting with the environment
    for i in range(total_interactions):
        user_features, items_features = obs
        actions, probs = vw_agent.choose_actions(shared_features=user_features,
                                                 candidate_arms_features=items_features,
                                                 user_id=env.current_user_id,
                                                 candidate_ids=env.current_item_pool,
                                                 top_k=5)

        clicks, regret, random_regret = env.get_feedback(actions)
        regrets.append(regret)
        random_regrets.append(random_regret)

        for index, reward in enumerate(clicks):
            vw_agent.learn(shared_features=user_features,
                           candidate_arms_features=items_features,
                           action_index=actions[index],
                           reward=reward,
                           user_id=env.current_user_id,
                           candidate_ids=env.current_item_pool,
                           action_prob=probs[index],
                           cost_fn=lambda x: -x)

        # Step the environment to pick next user and new list of candidate items
        obs, rewards, done, info = env.step(actions)
        if i % 500 == 0:
            logging.info(f"Processed {i} interactions")

    stdout = vw_agent.save_model(close=True)
    print(stdout.decode())
    logging.info(f"Model learned using {total_interactions} training experiences.")

    all_regrets = {"agent": regrets, "random": random_regrets}
    with open(os.path.join(DATA_OUTPUT_DIR, "output.json"), "w") as file:
        json.dump(all_regrets, file)


if __name__ == '__main__':
    main()
