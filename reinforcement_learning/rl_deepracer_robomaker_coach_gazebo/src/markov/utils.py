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
import shutil

SIMAPP_VERSION = "2.0"
DEFAULT_COLOR = "Black"

SIMAPP_SIMULATION_WORKER_EXCEPTION = "simulation_worker.exceptions"
SIMAPP_TRAINING_WORKER_EXCEPTION = "training_worker.exceptions"
SIMAPP_VALIDATION_WORKER_EXCEPTION = "validation_worker.exceptions"
SIMAPP_S3_DATA_STORE_EXCEPTION = "s3_datastore.exceptions"
SIMAPP_ENVIRONMENT_EXCEPTION = "environment.exceptions"
SIMAPP_MEMORY_BACKEND_EXCEPTION = "memory_backend.exceptions"
SIMAPP_SIMULATION_SAVE_TO_MP4_EXCEPTION = "save_to_mp4.exceptions"
SIMAPP_SIMULATION_KINESIS_VIDEO_CAMERA_EXCEPTION = "kinesis_video_camera.exceptions"

SIMAPP_EVENT_SYSTEM_ERROR = "system_error"
SIMAPP_EVENT_USER_ERROR = "user_error"

SIMAPP_EVENT_ERROR_CODE_500 = "500"
SIMAPP_EVENT_ERROR_CODE_400 = "400"
# The robomaker team has asked us to wait 5 minutes to let their workflow cancel
# the simulation job
ROBOMAKER_CANCEL_JOB_WAIT_TIME = 60 * 5
# The current checkpoint key
CHKPNT_KEY_SUFFIX = "model/.coach_checkpoint"
# This is the key for the best checkpoint
DEEPRACER_CHKPNT_KEY_SUFFIX = "model/deepracer_checkpoints.json"
# The number of times to retry a failed boto call
NUM_RETRIES = 5

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

def force_list(val):
    if type(val) is not list:
        val = [val]
    return val

def get_boto_config():
    '''Returns a botocore config object which specifies the number of times to retry'''
    return botocore.config.Config(retries=dict(max_attempts=NUM_RETRIES))

def cancel_simulation_job(simulation_job_arn, aws_region):
    logger.info("cancel_simulation_job: make sure to shutdown simapp first")
    session = boto3.session.Session()
    robomaker_client = session.client('robomaker', region_name=aws_region)
    robomaker_client.cancel_simulation_job(job=simulation_job_arn)
    time.sleep(ROBOMAKER_CANCEL_JOB_WAIT_TIME)

def restart_simulation_job(simulation_job_arn, aws_region):
    logger.info("restart_simulation_job: make sure to shutdown simapp first")
    session = boto3.session.Session()
    robomaker_client = session.client('robomaker', region_name=aws_region)
    robomaker_client.restart_simulation_job(job=simulation_job_arn)

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
    dict_obj['function'] = inspect.stack()[2][3]
    dict_obj['message'] = message
    for key, value in kwargs.items():
        if key == "log_level":
            log_error = kwargs[key] == "ERROR"
        else:
            dict_obj[key] = value
    if log_error:
        json_format_log["simapp_exception"] = dict_obj
        logger.error(json.dumps(json_format_log))
    else:
        json_format_log["simapp_info"] = dict_obj
        logger.info(json.dumps(json_format_log))

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
        except Exception:
            counter += 1
            time.sleep(1)

    if counter == timeout and not ip_address:
        error_string = "Environment Error: Could not retrieve IP address \
        for %s in past %s seconds." % (host_name, timeout)
        log_and_exit(error_string, SIMAPP_ENVIRONMENT_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)
    return ip_address

def load_model_metadata(s3_client, model_metadata_s3_key, model_metadata_local_path):
    """Loads the model metadata.
    """

    # Try to download the custom model metadata from s3 first
    download_success = False
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
        except Exception:
            logger.info("Could not download custom model metadata, using defaults.")

    # If the download was unsuccessful, load the default model metadata instead
    if not download_success:
        from markov.defaults import model_metadata
        with open(model_metadata_local_path, 'w') as f:
            json.dump(model_metadata, f, indent=4)
        logger.info("Loaded default action space.")

SIMAPP_DONE_EXIT = 0
SIMAPP_ERROR_EXIT = -1
def simapp_exit_gracefully(simapp_exit=SIMAPP_ERROR_EXIT):
    #simapp exception leading to exiting the system
    # -close the running processes
    # -upload simtrace data to S3
    logger.info("simapp_exit_gracefully: simapp_exit-{}".format(simapp_exit))
    logger.info("Terminating simapp simulation...")
    stack_trace = traceback.format_exc()
    logger.info("deepracer_racetrack_env - callstack={}".format(stack_trace))
    if simapp_exit == SIMAPP_ERROR_EXIT:
        os._exit(1)


BEST_CHECKPOINT = 'best_checkpoint'
LAST_CHECKPOINT = 'last_checkpoint'


def get_best_checkpoint_num(s3_bucket, s3_prefix, region):
    """Get the checkpoint number of the best checkpoint if its available, else return last checkpoint
    Args:
        s3_bucket (str): S3 bucket where the deepracer_checkpoints.json is stored
        s3_prefix (str): S3 prefix where the deepracer_checkpoints.json is stored
        region (str): AWS region where the deepracer_checkpoints.json is stored
    Returns:
        int: Best checkpoint or last checkpoint number if found else return -1
    """
    checkpoint_num = -1
    best_checkpoint_name = get_best_checkpoint(s3_bucket, s3_prefix, region)
    if best_checkpoint_name and len(best_checkpoint_name.split("_Step")) > 0:
        checkpoint_num = int(best_checkpoint_name.split("_Step")[0])
    else:
        logger.info("Unable to find the best checkpoint number. Getting the last checkpoint number")
        checkpoint_num = get_last_checkpoint_num(s3_bucket, s3_prefix, region)
    return checkpoint_num


def get_last_checkpoint_num(s3_bucket, s3_prefix, region):
    """Get the checkpoint number of the last checkpoint.
    Args:
        s3_bucket (str): S3 bucket where the deepracer_checkpoints.json is stored
        s3_prefix (str): S3 prefix where the deepracer_checkpoints.json is stored
        region (str): AWS region where the deepracer_checkpoints.json is stored
    Returns:
        int: Last checkpoint number if found else return -1
    """
    checkpoint_num = -1
    # Get the last checkpoint name from the deepracer_checkpoints.json file
    last_checkpoint_name = get_last_checkpoint(s3_bucket, s3_prefix, region)
    # Verify if the last checkpoint name is present and is in right format
    if last_checkpoint_name and len(last_checkpoint_name.split("_Step")) > 0:
        checkpoint_num = int(last_checkpoint_name.split("_Step")[0])
    else:
        logger.info("Unable to find the last checkpoint number.")
    return checkpoint_num


def copy_best_frozen_model_to_sm_output_dir(s3_bucket, s3_prefix, region,
                                            source_dir, dest_dir):
    """Copy the frozen model for the current best checkpoint from soure directory to the destination directory.
    Args:
        s3_bucket (str): S3 bucket where the deepracer_checkpoints.json is stored
        s3_prefix (str): S3 prefix where the deepracer_checkpoints.json is stored
        region (str): AWS region where the deepracer_checkpoints.json is stored
        source_dir (str): Source directory where the frozen models are present
        dest_dir (str): Sagemaker output directory where we store the frozen models for best checkpoint
    """
    dest_dir_pb_files = [filename for filename in os.listdir(dest_dir)
                         if os.path.isfile(os.path.join(dest_dir, filename)) and filename.endswith(".pb")]
    source_dir_pb_files = [filename for filename in os.listdir(source_dir)
                           if os.path.isfile(os.path.join(source_dir, filename)) and filename.endswith(".pb")]
    best_checkpoint_num_s3 = get_best_checkpoint_num(s3_bucket,
                                                     s3_prefix,
                                                     region)
    last_checkpoint_num_s3 = get_last_checkpoint_num(s3_bucket,
                                                     s3_prefix,
                                                     region)
    logger.info("Best checkpoint number: {}, Last checkpoint number: {}"
                .format(best_checkpoint_num_s3, last_checkpoint_num_s3))
    best_model_name = 'model_{}.pb'.format(best_checkpoint_num_s3)
    last_model_name = 'model_{}.pb'.format(last_checkpoint_num_s3)
    if len(source_dir_pb_files) < 1:
        log_and_exit("Could not find any frozen model file in the local directory",
                     SIMAPP_S3_DATA_STORE_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)
    # Could not find the deepracer_checkpoints.json file or there are no model.pb files in destination
    if best_checkpoint_num_s3 == -1 or len(dest_dir_pb_files) == 0:
        if len(source_dir_pb_files) > 1:
            logger.info("More than one model.pb found in the source directory. Choosing the "
                        "first one to copy to destination: {}".format(source_dir_pb_files[0]))
        # copy the frozen model present in the source directory
        logger.info("Copying the frozen checkpoint from {} to {}.".format(
                    os.path.join(source_dir, source_dir_pb_files[0]), os.path.join(dest_dir, "model.pb")))
        shutil.copy(os.path.join(source_dir, source_dir_pb_files[0]), os.path.join(dest_dir, "model.pb"))
    else:
        # Delete the current .pb files in the destination direcory
        """
        for filename in dest_dir_pb_files:
            os.remove(os.path.join(dest_dir, filename))
        """

        # Copy the frozen model for the current best checkpoint to the destination directory
        logger.info("Copying the frozen checkpoint from {} to {}.".format(
                    os.path.join(source_dir, best_model_name), os.path.join(dest_dir, "model.pb")))
        shutil.copy(os.path.join(source_dir, best_model_name), os.path.join(dest_dir, "model.pb"))

        # Loop through the current list of frozen models in source directory and
        # delete the iterations lower than last_checkpoint_iteration except best_model
        """
        for filename in source_dir_pb_files:
            if filename not in [best_model_name, last_model_name]:
                if len(filename.split("_")[1]) > 1 and len(filename.split("_")[1].split(".pb")):
                    file_iteration = int(filename.split("_")[1].split(".pb")[0])
                    if file_iteration < last_checkpoint_num_s3:
                        os.remove(os.path.join(source_dir, filename))
                else:
                    logger.error("Frozen model name not in the right format in the source directory: {}, {}"
                                 .format(filename, source_dir))
        """

def get_best_checkpoint(s3_bucket, s3_prefix, region):
    return get_deepracer_checkpoint(s3_bucket=s3_bucket,
                                    s3_prefix=s3_prefix,
                                    region=region,
                                    checkpoint_type=BEST_CHECKPOINT)


def get_last_checkpoint(s3_bucket, s3_prefix, region):
    return get_deepracer_checkpoint(s3_bucket=s3_bucket,
                                    s3_prefix=s3_prefix,
                                    region=region,
                                    checkpoint_type=LAST_CHECKPOINT)


def get_deepracer_checkpoint(s3_bucket, s3_prefix, region, checkpoint_type):
    '''Returns the best checkpoint stored in the best checkpoint json
       s3_bucket - DeepRacer s3 bucket
       s3_prefix - Prefix for the training job for which to select the best model for
       region - Name of the aws region where the job ran
       checkpoint_type - BEST_CHECKPOINT/LAST_CHECKPOINT
    '''
    try:
        session = boto3.Session()
        s3_client = session.client('s3', region_name=region, config=get_boto_config())
        # Download the best model if available
        deepracer_checkpoint_json = os.path.join(os.getcwd(), 'deepracer_checkpoints.json')
        s3_client.download_file(Bucket=s3_bucket,
                                Key=os.path.join(s3_prefix, DEEPRACER_CHKPNT_KEY_SUFFIX),
                                Filename=deepracer_checkpoint_json)
    except botocore.exceptions.ClientError as err:
        if err.response['Error']['Code'] == "404":
            logger.info("Unable to find best model data, using last model")
            return None
        else:
            log_and_exit("Unable to download best checkpoint: {}, {}".\
                         format(s3_bucket, err.response['Error']['Code']),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as ex:
        log_and_exit("Can't download best checkpoint {}".format(ex),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)
    try:
        with open(deepracer_checkpoint_json) as deepracer_checkpoint_file:
            checkpoint = json.load(deepracer_checkpoint_file)[checkpoint_type]["name"]
            if not checkpoint:
                raise Exception("No checkpoint recorded")
        os.remove(deepracer_checkpoint_json)
    except Exception as ex:
        logger.info("Unable to parse best checkpoint data: {}, using last \
                    checkpoint instead".format(ex))
        return None
    return checkpoint


def do_model_selection(s3_bucket, s3_prefix, region, checkpoint_type=BEST_CHECKPOINT):
    '''Sets the chekpoint file to point at the best model based on reward and progress
       s3_bucket - DeepRacer s3 bucket
       s3_prefix - Prefix for the training job for which to select the best model for
       region - Name of the aws region where the job ran

       :returns status of model selection. True if successfully selected model otherwise false.
    '''
    try:
        model_checkpoint = get_deepracer_checkpoint(s3_bucket=s3_bucket,
                                                    s3_prefix=s3_prefix,
                                                    region=region,
                                                    checkpoint_type=checkpoint_type)
        if model_checkpoint is None:
            return False
        local_path = os.path.abspath(os.path.join(os.getcwd(), 'coach_checkpoint'))
        with open(local_path, '+w') as new_ckpnt:
            new_ckpnt.write(model_checkpoint)
        s3_client = boto3.Session().client('s3', region_name=region, config=get_boto_config())
        s3_client.upload_file(Filename=local_path,
                              Bucket=s3_bucket,
                              Key=os.path.join(s3_prefix, CHKPNT_KEY_SUFFIX))
        os.remove(local_path)
        return True
    except botocore.exceptions.ClientError as err:
        log_and_exit("Unable to upload checkpoint: {}, {}".format(s3_bucket,
                                                                  err.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as ex:
        log_and_exit("Unable to upload checkpoint: {}".format(ex),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)

def has_current_ckpnt_name(s3_bucket, s3_prefix, region):
    '''This method checks if a given s3 bucket contains the current checkpoint key
       s3_bucket - DeepRacer s3 bucket
       s3_prefix - Prefix for the training job for which to select the best model for
       region - Name of the aws region where the job ran
    '''
    try:
        session = boto3.Session()
        s3_client = session.client('s3', region_name=region, config=get_boto_config())
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
        s3_client = session.client('s3', region_name=region, config=get_boto_config())

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
