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
        id='SilverstoneRacetrack-v0',
        entry_point='custom_files.silverstone_racetrack_env:SilverstoneRacetrackEnv',
        max_episode_steps=MAX_STEPS,
        reward_threshold=200
    )

    register(
        id='SilverstoneRacetrack-Discrete-v0',
        entry_point='custom_files.silverstone_racetrack_env:SilverstoneRacetrackDiscreteEnv',
        max_episode_steps=MAX_STEPS,
        reward_threshold=200
    )

    register(
        id='SilverstoneRacetrack-MultiDiscrete-v0',
        entry_point='custom_files.silverstone_racetrack_env:SilverstoneRacetrackMultiDiscreteEnv',
        max_episode_steps=MAX_STEPS,
        reward_threshold=200
    )


def write_frozen_graph(graph_manager):
    if not os.path.exists(SM_MODEL_OUTPUT_DIR):
        os.makedirs(SM_MODEL_OUTPUT_DIR)
    output_head = ['main_level/agent/main/online/network_1/ppo_head_0/policy']
    frozen = tf.graph_util.convert_variables_to_constants(graph_manager.sess, graph_manager.sess.graph_def, output_head)
    tf.train.write_graph(frozen, SM_MODEL_OUTPUT_DIR, 'model.pb', as_text=False)
    print("Saved TF frozen graph!")


class DoorMan:
    def __init__(self):
        self.terminate_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.terminate_now = True
