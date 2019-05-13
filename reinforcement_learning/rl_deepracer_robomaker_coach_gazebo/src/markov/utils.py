import socket
import logging
import time
import os
import signal

logger = logging.getLogger(__name__)

import tensorflow as tf

SM_MODEL_OUTPUT_DIR = os.environ.get("ALGO_MODEL_DIR", "/opt/ml/model")


def get_ip_from_host(timeout=100):
    counter = 0
    ip_address = None

    host_name = socket.gethostname()
    logger.info("Hostname: %s" % host_name)
    while counter < timeout and not ip_address:
        try:
            ip_address = socket.gethostbyname(host_name)
            break
        except Exception as e:
            counter += 1
            time.sleep(1)

    if counter == timeout and not ip_address:
        error_string = "Platform Error: Could not retrieve IP address \
        for %s in past %s seconds" % (host_name, timeout)
        raise RuntimeError(error_string)

    return ip_address


def register_custom_environments():
    from gym.envs.registration import register

    MAX_STEPS = 1000

    register(
        id='DeepRacerRacetrack-v0',
        entry_point='custom_files.deepracer_racetrack_env:DeepRacerRacetrackEnv',
        max_episode_steps=MAX_STEPS,
        reward_threshold=200
    )

    register(
        id='DeepRacerRacetrackCustomActionSpaceEnv-v0',
        entry_point='markov.environments.deepracer_racetrack_env:DeepRacerRacetrackCustomActionSpaceEnv',
        max_episode_steps=MAX_STEPS,
        reward_threshold=200
    )


def write_frozen_graph(graph_manager):
    if not os.path.exists(SM_MODEL_OUTPUT_DIR):
        os.makedirs(SM_MODEL_OUTPUT_DIR)
    output_head = ['main_level/agent/main/online/network_1/ppo_head_0/policy']
    frozen = tf.graph_util.convert_variables_to_constants(graph_manager.sess, graph_manager.sess.graph_def, output_head)
    tf.train.write_graph(frozen, SM_MODEL_OUTPUT_DIR, 'model.pb', as_text=False)
    logger.info("Saved TF frozen graph!")

def load_model_metadata(s3_client, model_metadata_s3_key, model_metadata_local_path):
    """Loads the model metadata.
    """

    # Strip the s3://<bucket> prefix if it exists
    logger.info('s3_client: {} \n \ 
           model_metadata_s3_key: {} \n \ 
           model_metadata_local_path: {}'.format(
           s3_client,model_metadata_s3_key,model_metadata_local_path)) 
    
    # Try to download the custom model metadata from s3 first
    download_success = False;
    if not model_metadata_s3_key:
        logger.info("Custom model metadata key not provided, using defaults.")
    else:
        # Strip the s3://<bucket> prefix if it exists
        model_metadata_s3_key = model_metadata_s3_key.replace('s3://{}/'.format(s3_client.bucket), '')

        download_success = s3_client.download_file(s3_key=model_metadata_s3_key,
                                                   local_path=model_metadata_local_path)
        if download_success:
            logger.info("Successfully downloaded model metadata from {}.".format(model_metadata_s3_key))
        else:
            logger.warning("Could not download custom model metadata from {}, using defaults.".format(model_metadata_s3_key))

    # If the download was successful, validate the contents
    if download_success:
        try:
            with open(model_metadata_local_path, 'r') as f:
                model_metadata = json.load(f)
                if 'action_space' not in model_metadata:
                    logger.error("Custom model metadata does not define an action space.")
                    download_success = False
        except:
            logger.error("Could not download custom model metadata, using defaults.")

    # If the download was unsuccessful, load the default model metadata instead
    if not download_success:
        from markov.defaults import model_metadata
        with open(model_metadata_local_path, 'w') as f:
            json.dump(model_metadata, f, indent=4)
        logger.info("Loaded default action space.")

class DoorMan:
    def __init__(self):
        self.terminate_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.terminate_now = True
