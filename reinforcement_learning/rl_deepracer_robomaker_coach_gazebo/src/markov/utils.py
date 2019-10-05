import json
import logging
import os
import sys
import signal
import socket
import time
import datetime
import inspect
from collections import OrderedDict

SIMAPP_VERSION="1.0"

SIMAPP_SIMULATION_WORKER_EXCEPTION = "simulation_worker.exceptions"
SIMAPP_TRAINING_WORKER_EXCEPTION = "training_worker.exceptions"
SIMAPP_S3_DATA_STORE_EXCEPTION = "s3_datastore.exceptions"
SIMAPP_ENVIRONMENT_EXCEPTION = "environment.exceptions"
SIMAPP_MEMORY_BACKEND_EXCEPTION = "memory_backend.exceptions"

SIMAPP_EVENT_SYSTEM_ERROR = "system_error"
SIMAPP_EVENT_USER_ERROR = "user_error"

SIMAPP_EVENT_ERROR_CODE_500 = "500"
SIMAPP_EVENT_ERROR_CODE_503 = "503"
SIMAPP_EVENT_ERROR_CODE_400 = "400"
SIMAPP_EVENT_ERROR_CODE_401 = "401"

class Logger(object):
    counter = 0
    """
    Logger class for all DeepRacer Simulation Application logging
    """
    def __init__(self, logger_name=__name__, log_level=logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        self.logger.addHandler(handler)

    def get_logger(self):
        """
        Returns the logger object with all the required log settings.
        """
        return self.logger

logger = Logger(__name__, logging.INFO).get_logger()

import tensorflow as tf

SM_MODEL_OUTPUT_DIR = os.environ.get("ALGO_MODEL_DIR", "/opt/ml/model")

def json_format_logger (msg, *args, **kwargs):
    dict_obj = OrderedDict()
    json_format_log = dict()
    log_error = False

    message = msg.format(args)
    dict_obj['version'] = SIMAPP_VERSION
    dict_obj['date'] = str(datetime.datetime.now())
    dict_obj['function'] = inspect.stack()[1][3]
    dict_obj['message'] = message
    for key, value in kwargs.items():
        if key == "log_level":
            log_error = kwargs[key] == "ERROR"
        else:
            dict_obj[key] = value
    if log_error:
        json_format_log["simapp_exception"] = dict_obj
        logger.error (json.dumps(json_format_log))
    else:
        json_format_log["simapp_info"] = dict_obj
        logger.info (json.dumps(json_format_log))

def build_system_error_dict(exception_type, errcode):
    """
    Creates system exception dictionary to be printed in the logs
    """
    return {"exceptionType":exception_type,\
            "eventType":SIMAPP_EVENT_SYSTEM_ERROR,\
            "errorCode":errcode, "log_level":"ERROR"}

def build_user_error_dict(exception_type, errcode):
    """
    Creates user exception dictionary to be printed in the logs
    """
    return {"exceptionType":exception_type,\
            "eventType":SIMAPP_EVENT_USER_ERROR,\
            "errorCode":errcode, "log_level":"ERROR"}

def get_ip_from_host(timeout=100):
    counter = 0
    ip_address = None

    host_name = socket.gethostname()
    logger.debug("Hostname: %s" % host_name)
    while counter < timeout and not ip_address:
        try:
            ip_address = socket.gethostbyname(host_name)
            break
        except Exception as e:
            counter += 1
            time.sleep(1)

    if counter == timeout and not ip_address:
        error_string = "Environment Error: Could not retrieve IP address \
        for %s in past %s seconds. Job failed!" % (host_name, timeout)
        json_format_logger (error_string,
                            **build_system_error_dict(SIMAPP_ENVIRONMENT_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_503))
        sys.exit(1)

    return ip_address

def write_frozen_graph(graph_manager):
    if not os.path.exists(SM_MODEL_OUTPUT_DIR):
        os.makedirs(SM_MODEL_OUTPUT_DIR)
    output_head = ['main_level/agent/main/online/network_1/ppo_head_0/policy']
    frozen = tf.graph_util.convert_variables_to_constants(graph_manager.sess, graph_manager.sess.graph_def, output_head)
    tf.train.write_graph(frozen, SM_MODEL_OUTPUT_DIR, 'model.pb', as_text=False)


def load_model_metadata(s3_client, model_metadata_s3_key, model_metadata_local_path):
    """Loads the model metadata.
    """

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
           logger.info("Could not download custom model metadata from {}, using defaults.".format(model_metadata_s3_key))

    # If the download was successful, validate the contents
    if download_success:
        try:
            with open(model_metadata_local_path, 'r') as f:
                model_metadata = json.load(f)
                if 'action_space' not in model_metadata:
                    logger.info("Custom model metadata does not define an action space.")
                    download_success = False
        except:
            logger.info("Could not download custom model metadata, using defaults.")

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
