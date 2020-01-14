import threading
import json
import logging
import os
import io
import re
import signal
import socket
import time
import datetime
import inspect
from collections import OrderedDict
import traceback
import boto3
import botocore
import numpy as np

SIMAPP_VERSION="2.0"
DEFAULT_COLOR="Black"

SIMAPP_SIMULATION_WORKER_EXCEPTION = "simulation_worker.exceptions"
SIMAPP_TRAINING_WORKER_EXCEPTION = "training_worker.exceptions"
SIMAPP_S3_DATA_STORE_EXCEPTION = "s3_datastore.exceptions"
SIMAPP_ENVIRONMENT_EXCEPTION = "environment.exceptions"
SIMAPP_MEMORY_BACKEND_EXCEPTION = "memory_backend.exceptions"

SIMAPP_EVENT_SYSTEM_ERROR = "system_error"
SIMAPP_EVENT_USER_ERROR = "user_error"

SIMAPP_EVENT_ERROR_CODE_500 = "500"
SIMAPP_EVENT_ERROR_CODE_400 = "400"
# The robomaker team has asked us to wait 5 minutes to let their workflow cancel
# the simulation job
ROBOMAKER_CANCEL_JOB_WAIT_TIME = 60 * 5
# The current checkpoint key
CHKPNT_KEY_SUFFIX = "model/.coach_checkpoint"

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

def cancel_simulation_job(simulation_job_arn, aws_region):
    logger.info("cancel_simulation_job: make sure to shutdown simapp first")
    session = boto3.session.Session()
    robomaker_client = session.client('robomaker', region_name=aws_region)
    robomaker_client.cancel_simulation_job(job=simulation_job_arn)
    time.sleep(ROBOMAKER_CANCEL_JOB_WAIT_TIME)

def str2bool(flag):
    """ bool: convert flag to boolean if it is string and return it else return its initial bool value """
    if not isinstance(flag, bool):
        if flag.lower() == 'false':
            flag = False
        elif flag.lower() == 'true':
            flag = True
    return flag

def json_format_logger(msg, *args, **kwargs):
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

def log_and_exit(msg, error_source, error_code):
    ''' Helper method that logs an exception and exits the application
        msg - The message to be logged
        error_source - The source of the error, training worker, rolloutworker, etc
        error_code - 4xx or 5xx error
    '''
    error_dict = build_user_error_dict(error_source, error_code) \
        if error_code == SIMAPP_EVENT_ERROR_CODE_400 else build_system_error_dict(error_source, error_code)
    json_format_logger(msg, **error_dict)
    simapp_exit_gracefully()

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
        for %s in past %s seconds." % (host_name, timeout)
        json_format_logger (error_string,
                            **build_system_error_dict(SIMAPP_ENVIRONMENT_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500))
        simapp_exit_gracefully()

    return ip_address

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

SIMAPP_DONE_EXIT=0
SIMAPP_ERROR_EXIT=-1
def simapp_exit_gracefully(simapp_exit=SIMAPP_ERROR_EXIT):
    #simapp exception leading to exiting the system
    # -close the running processes
    # -upload simtrace data to S3
    logger.info("simapp_exit_gracefully: simapp_exit-{}".format(simapp_exit))
    logger.info("Terminating simapp simulation...")
    stack_trace = traceback.format_exc()
    logger.info ("deepracer_racetrack_env - callstack={}".format(stack_trace))
    if simapp_exit == SIMAPP_ERROR_EXIT:
        os._exit(1)

def do_model_selection(s3_bucket, s3_prefix, region):
    '''Sets the chekpoint file to point at the best model based on reward and progress
       s3_bucket - DeepRacer s3 bucket
       s3_prefix - Prefix for the training job for which to select the best model for
       region - Name of the aws region where the job ran
    '''
    session = boto3.Session()
    s3_client = session.client('s3', region_name=region)
    # Download training metrics
    training_metrics_json = os.path.join(os.getcwd(), 'training_metrics.json')
    try:
        s3_client.download_file(Bucket=s3_bucket,
                                Key=os.path.join(s3_prefix, 'training_metrics.json'),
                                Filename=training_metrics_json)
    except botocore.exceptions.ClientError as e:
        log_and_exit("Can't download traing metrics {}, {}".format(s3_bucket,
                                                                   e.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as e:
        log_and_exit("Can't download traing metrics {}".format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)

    # Download hyperparameters
    hyperparameters_json = os.path.join(os.getcwd(), 'hyperparameters.json')
    try:
        s3_client.download_file(Bucket=s3_bucket,
                                Key=os.path.join(s3_prefix, 'ip/hyperparameters.json'),
                                Filename=hyperparameters_json)
    except botocore.exceptions.ClientError as e:
        log_and_exit("Can't download hyperparameters {}, {}".format(s3_bucket,
                                                                    e.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as e:
        log_and_exit("Can't download hyperparameters {}".format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)
    try:
        with open(training_metrics_json) as training_metrics_file:
            training_metrics = json.load(training_metrics_file)["metrics"]
        with open(hyperparameters_json) as hyperparameters_file:
            n_episodes_per_rollout = json.load(hyperparameters_file)["num_episodes_between_training"]
        # We only care about completed rollouts
        training_metrics = training_metrics[:len(training_metrics)-(len(training_metrics) % n_episodes_per_rollout)]
        # Extract the relevent a data
        metrics_per_episode = np.array([[metric["completion_percentage"], metric["reward_score"]]
                                        for metric in training_metrics])
        metrics_per_rollout = np.reshape(metrics_per_episode, (-1, n_episodes_per_rollout, 2))
        # Compute the averages per rollout
        means = np.mean(metrics_per_rollout, axis=1)
        scores = np.sum(means**2, axis=1)
        # Find the best episode
        best_episode_score = np.amax(scores)
        # First checkpoint has no training metrics associated with it because it is random
        best_episode_index = np.asscalar(np.where(scores == best_episode_score)[0]) + 1
        # Delete the local copies
        os.remove(training_metrics_json)
        os.remove(hyperparameters_json)
    except Exception as e:
        log_and_exit("Unable calculate best policy{}".format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)

    chkpnt_index = None
    ckpnt_total_step = None
    # Look at the checkpoints available in the s3 bucket to see if the optimal policy is present
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket,
                                             Prefix=os.path.join(s3_prefix, "model"))
        for obj in response['Contents']:
            _, tail = os.path.split(obj['Key'])
            file_name = tail.split('_')
            if file_name[0] == str(best_episode_index) and len(file_name) > 1:
                ckpnt_candiate = file_name[1].split('.')
                if 'ckpt' in ckpnt_candiate:
                    chkpnt_index = best_episode_index
                    ckpnt_total_step = ckpnt_candiate[0].split('-')[1]
                    break
    except botocore.exceptions.ClientError as e:
        log_and_exit("Unable to list checkpoint files: {}, {}".format(s3_bucket,
                                                                      e.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as e:
        log_and_exit("Unable to list checkpoint files: {}".format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)
    # Write and upload a new ckeck point file if the optimal policy is available for download
    try:
        local_path = os.path.abspath(os.path.join(os.getcwd(), 'coach_checkpoint'))
        if chkpnt_index is not None:
            with open(local_path, '+w') as new_ckpnt:
                optimal_model = '{}_Step-{}.ckpt'.format(chkpnt_index, ckpnt_total_step)
                logger.info('Detected best policy: {}'.format(optimal_model))
                new_ckpnt.write(optimal_model)
            s3_client.upload_file(Filename=local_path, Bucket=s3_bucket,
                                  Key=os.path.join(s3_prefix, CHKPNT_KEY_SUFFIX))
            # Delete the local copy
            os.remove(local_path)
        else:
            logger.info("Optimal Policy not stored in models s3 bucket")
    except botocore.exceptions.ClientError as e:
        log_and_exit("Unable to upload checkpoint: {}, {}".format(s3_bucket,
                                                                  e.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as e:
        log_and_exit("Unable to upload checkpoint: {}".format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)


def has_current_ckpnt_name(s3_bucket, s3_prefix, region):
    '''This method checks if a given s3 bucket contains the current checkpoint key
       s3_bucket - DeepRacer s3 bucket
       s3_prefix - Prefix for the training job for which to select the best model for
       region - Name of the aws region where the job ran
    '''
    try:
        session = boto3.Session()
        s3_client = session.client('s3', region_name=region)
        response = s3_client.list_objects_v2(Bucket=s3_bucket,
                                             Prefix=os.path.join(s3_prefix, "model"))
        if 'Contents' not in response:
            # Customer deleted checkpoint file.
            log_and_exit("No objects found: {}".format(s3_bucket),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)

        _, ckpnt_name = os.path.split(CHKPNT_KEY_SUFFIX)
        return any(list(map(lambda obj: os.path.split(obj['Key'])[1] == ckpnt_name,
                            response['Contents'])))
    except botocore.exceptions.ClientError as e:
        log_and_exit("No objects found: {}, {}".format(s3_bucket, e.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as e:
        log_and_exit("No objects found: {}".format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)

def make_compatible(s3_bucket, s3_prefix, region, ready_file):
    '''Moves and creates all the necessary files to make models trained by coach 0.11
       compatible with coach 1.0
       s3_bucket - DeepRacer s3 bucket
       s3_prefix - Prefix for the training job for which to select the best model for
       region - Name of the aws region where the job ran
    '''
    try:
        session = boto3.Session()
        s3_client = session.client('s3', region_name=region)

        old_checkpoint = os.path.join(os.getcwd(), 'checkpoint')
        s3_client.download_file(Bucket=s3_bucket,
                                Key=os.path.join(s3_prefix, 'model/checkpoint'),
                                Filename=old_checkpoint)

        with open(old_checkpoint) as old_checkpoint_file:
            chekpoint = re.findall(r'"(.*?)"', old_checkpoint_file.readline())
        if len(chekpoint) != 1:
            log_and_exit("No checkpoint file found", SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        os.remove(old_checkpoint)
        # Upload ready file so that the system can gab the checkpoints
        s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                 Bucket=s3_bucket,
                                 Key=os.path.join(s3_prefix, "model/{}").format(ready_file))
        # Upload the new checkpoint file
        new_checkpoint = os.path.join(os.getcwd(), 'coach_checkpoint')
        with open(new_checkpoint, 'w+') as new_checkpoint_file:
            new_checkpoint_file.write(chekpoint[0])
        s3_client.upload_file(Filename=new_checkpoint, Bucket=s3_bucket,
                              Key=os.path.join(s3_prefix, CHKPNT_KEY_SUFFIX))
        os.remove(new_checkpoint)
    except botocore.exceptions.ClientError as e:
        log_and_exit("Unable to make model compatible: {}, {}".format(s3_bucket,
                                                                      e.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as e:
        log_and_exit("Unable to make model compatible: {}".format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)

def is_error_bad_ckpnt(error):
    ''' Helper method that determines whether a value error is caused by an invalid checkpoint
        by looking for keywords in the exception message
        error - Python exception object, which produces a message by implementing __str__
    '''
    # These are the key words, which if present indicate that the tensorflow saver was unable
    # to restore a checkpoint because it does not match the graph.
    keys = ['tensor', 'shape', 'checksum', 'checkpoint']
    return any(key in str(error).lower() for key in keys)

class DoorMan:
    def __init__(self):
        self.terminate_now = False
        logger.info("DoorMan: installing SIGINT, SIGTERM")
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.terminate_now = True
        logger.info("DoorMan: received signal {}".format(signum))
        simapp_exit_gracefully(SIMAPP_DONE_EXIT)

class DoubleBuffer(object):
    def __init__(self, clear_data_on_get=True):
        self.read_buffer = None
        self.write_buffer = None
        self.clear_data_on_get = clear_data_on_get
        self.cv = threading.Condition()

    def clear(self):
        with self.cv:
            self.read_buffer = None
            self.write_buffer = None

    def put(self, data):
        with self.cv:
            self.write_buffer = data
            self.write_buffer, self.read_buffer = self.read_buffer, self.write_buffer
            self.cv.notify()

    def get(self, block=True):
        with self.cv:
            if not block:
                if self.read_buffer is None:
                    raise DoubleBuffer.Empty
            else:
                while self.read_buffer is None:
                    self.cv.wait()
            data = self.read_buffer
            if self.clear_data_on_get:
              self.read_buffer = None
            return data

    def get_nowait(self):
        return self.get(block=False)

    class Empty(Exception):
        pass
