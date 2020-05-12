import threading
import json
import logging
import os
import io
import re
import signal
import socket
import time
import cProfile
import pstats
from itertools import count
from markov.log_handler.constants import (SIMAPP_ENVIRONMENT_EXCEPTION,
                                          SIMAPP_EVENT_ERROR_CODE_500,
                                          SIMAPP_S3_DATA_STORE_EXCEPTION,
                                          SIMAPP_EVENT_ERROR_CODE_400,
                                          SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_DONE_EXIT)
from markov.log_handler.logger import Logger
from markov.log_handler.exception_handler import log_and_exit, simapp_exit_gracefully
from markov.log_handler.deepracer_exceptions import GenericException
from markov.constants import (ROBOMAKER_CANCEL_JOB_WAIT_TIME,
                              CHKPNT_KEY_SUFFIX, DEEPRACER_CHKPNT_KEY_SUFFIX,
                              NUM_RETRIES, BEST_CHECKPOINT, LAST_CHECKPOINT,
                              SAGEMAKER_S3_KMS_CMK_ARN, ROBOMAKER_S3_KMS_CMK_ARN,
                              S3_KMS_CMK_ARN_ENV, HYPERPARAMETERS, SAGEMAKER_IS_PROFILER_ON,
                              SAGEMAKER_PROFILER_S3_BUCKET, SAGEMAKER_PROFILER_S3_PREFIX,
                              S3KmsEncryption)
import boto3
import botocore
import shutil

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
    if simulation_job_arn:
        session = boto3.session.Session()
        robomaker_client = session.client('robomaker', region_name=aws_region)
        robomaker_client.cancel_simulation_job(job=simulation_job_arn)
        time.sleep(ROBOMAKER_CANCEL_JOB_WAIT_TIME)
    else:
        simapp_exit_gracefully()

def restart_simulation_job(simulation_job_arn, aws_region):
    logger.info("restart_simulation_job: make sure to shutdown simapp first")
    if simulation_job_arn:
        session = boto3.session.Session()
        robomaker_client = session.client('robomaker', region_name=aws_region)
        robomaker_client.restart_simulation_job(job=simulation_job_arn)
    else:
        simapp_exit_gracefully()

def str2bool(flag):
    """ bool: convert flag to boolean if it is string and return it else return its initial bool value """
    if not isinstance(flag, bool):
        if flag.lower() == 'false':
            flag = False
        elif flag.lower() == 'true':
            flag = True
    return flag

def str_to_done_condition(done_condition):
    if done_condition == any or done_condition == all:
        return done_condition
    elif done_condition.lower().strip() == 'all':
        return all
    return any

def pos_2d_str_to_list(list_pos_str):
    if list_pos_str and isinstance(list_pos_str[0], str):
        return [tuple(map(float, pos_str.split(","))) for pos_str in list_pos_str]
    return list_pos_str

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
        log_and_exit(error_string, 
                     SIMAPP_ENVIRONMENT_EXCEPTION, 
                     SIMAPP_EVENT_ERROR_CODE_500)
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
    try:
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
            for filename in dest_dir_pb_files:
                os.remove(os.path.join(dest_dir, filename))

            # Copy the frozen model for the current best checkpoint to the destination directory
            logger.info("Copying the frozen checkpoint from {} to {}.".format(
                        os.path.join(source_dir, best_model_name), os.path.join(dest_dir, "model.pb")))
            shutil.copy(os.path.join(source_dir, best_model_name), os.path.join(dest_dir, "model.pb"))

            # Loop through the current list of frozen models in source directory and
            # delete the iterations lower than last_checkpoint_iteration except best_model
            for filename in source_dir_pb_files:
                if filename not in [best_model_name, last_model_name]:
                    if len(filename.split("_")[1]) > 1 and len(filename.split("_")[1].split(".pb")):
                        file_iteration = int(filename.split("_")[1].split(".pb")[0])
                        if file_iteration < last_checkpoint_num_s3:
                            os.remove(os.path.join(source_dir, filename))
                    else:
                        logger.error("Frozen model name not in the right format in the source directory: {}, {}"
                                     .format(filename, source_dir))
    except FileNotFoundError as err:
        log_and_exit("No such file or directory: {}".format(err),
                     SIMAPP_S3_DATA_STORE_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_400)


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
                         SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                         SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as ex:
        log_and_exit("Can't download best checkpoint: {}"
                         .format(ex),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                     SIMAPP_EVENT_ERROR_CODE_500)
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
        s3_extra_args = get_s3_kms_extra_args()
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
                              Key=os.path.join(s3_prefix, CHKPNT_KEY_SUFFIX),
                              ExtraArgs=s3_extra_args)
        os.remove(local_path)
        return True
    except botocore.exceptions.ClientError as err:
        log_and_exit("Unable to upload checkpoint: {}, {}"
                        .format(s3_bucket, err.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                     SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as ex:
        log_and_exit("Exception in uploading checkpoint: {}"
                         .format(ex),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                     SIMAPP_EVENT_ERROR_CODE_500)

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
            log_and_exit("No objects found: {}"
                             .format(s3_bucket),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                         SIMAPP_EVENT_ERROR_CODE_400)

        _, ckpnt_name = os.path.split(CHKPNT_KEY_SUFFIX)
        return any(list(map(lambda obj: os.path.split(obj['Key'])[1] == ckpnt_name,
                            response['Contents'])))
    except botocore.exceptions.ClientError as e:
        log_and_exit("No objects found: {}, {}"
                         .format(s3_bucket, e.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                     SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as e:
        log_and_exit("Exception in checking for current checkpoint key: {}"
                         .format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                     SIMAPP_EVENT_ERROR_CODE_500)

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
        s3_extra_args = get_s3_kms_extra_args()
        old_checkpoint = os.path.join(os.getcwd(), 'checkpoint')
        s3_client.download_file(Bucket=s3_bucket,
                                Key=os.path.join(s3_prefix, 'model/checkpoint'),
                                Filename=old_checkpoint)

        with open(old_checkpoint) as old_checkpoint_file:
            chekpoint = re.findall(r'"(.*?)"', old_checkpoint_file.readline())
        if len(chekpoint) != 1:
            log_and_exit("No checkpoint file found", 
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        os.remove(old_checkpoint)
        # Upload ready file so that the system can gab the checkpoints
        s3_client.upload_fileobj(Fileobj=io.BytesIO(b''),
                                 Bucket=s3_bucket,
                                 Key=os.path.join(s3_prefix, "model/{}").format(ready_file),
                                 ExtraArgs=s3_extra_args)
        # Upload the new checkpoint file
        new_checkpoint = os.path.join(os.getcwd(), 'coach_checkpoint')
        with open(new_checkpoint, 'w+') as new_checkpoint_file:
            new_checkpoint_file.write(chekpoint[0])
        s3_client.upload_file(Filename=new_checkpoint, Bucket=s3_bucket,
                              Key=os.path.join(s3_prefix, CHKPNT_KEY_SUFFIX),
                              ExtraArgs=s3_extra_args)
        os.remove(new_checkpoint)
    except botocore.exceptions.ClientError as e:
        log_and_exit("Unable to make model compatible: {}, {}"
                         .format(s3_bucket, e.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                     SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as e:
        log_and_exit("Exception in making model compatible: {}"
                         .format(e),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, 
                     SIMAPP_EVENT_ERROR_CODE_500)

def is_error_bad_ckpnt(error):
    ''' Helper method that determines whether a value error is caused by an invalid checkpoint
        by looking for keywords in the exception message
        error - Python exception object, which produces a message by implementing __str__
    '''
    # These are the key words, which if present indicate that the tensorflow saver was unable
    # to restore a checkpoint because it does not match the graph.
    keys = ['tensor', 'shape', 'checksum', 'checkpoint']
    return any(key in str(error).lower() for key in keys)

def get_video_display_name():
    """ Based on the job type display the appropriate name on the mp4.
    For the RACING job type use the racer alias name and for others use the model name.

    Returns:
        list: List of the display name. In head2head there would be two values else one value.
    """
    #
    # The import rospy statement is here because the util is used by the sagemaker and it fails because ROS is not installed.
    # Also the rollout_utils.py fails because its PhaseObserver is python3 compatable and not python2.7 because of 
    # def __init__(self, topic: str, sink: RunPhaseSubject) -> None:
    #
    import rospy
    video_job_type = rospy.get_param("VIDEO_JOB_TYPE", "")
    # TODO: This code should be removed when the cloud service starts providing VIDEO_JOB_TYPE YAML parameter
    if not video_job_type:
        return force_list(rospy.get_param("DISPLAY_NAME", ""))
    if video_job_type == "RACING":
        return force_list(rospy.get_param("RACER_NAME", ""))
    return force_list(rospy.get_param("MODEL_NAME", ""))

def get_racecar_names(racecar_num):
    """Return the racer names based on the number of racecars given.

    Arguments:
        racecar_num (int): The number of race cars
    Return:
        [] - the list of race car names
    """
    racer_names = []
    if racecar_num == 1:
        racer_names.append('racecar')
    else:
        for idx in range(racecar_num):
            racer_names.append('racecar_' + str(idx))
    return racer_names

def get_racecar_idx(racecar_name):
    try:
        racecar_name_list = racecar_name.split("_")
        if len(racecar_name_list) == 1:
            return None
        racecar_num = racecar_name_list[1]
        return int(racecar_num)
    except Exception as ex:
        log_and_exit("racecar name should be in format racecar_x. However, get {}".\
                        format(racecar_name),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)

def get_s3_kms_extra_args():
    """ Since the SageS3Client class is called by both robomaker and sagemaker. One has to know
    first if its coming from sagemaker or robomaker. Then alone I could decide to fetch the kms arn
    to encrypt all the S3 upload object. Return the extra args that is required to encrypt the s3 object with KMS key
    If the KMS key not passed then returns empty dict

    Returns:
        dict: With the kms encryption arn if passed else empty dict
    """
    #
    # TODO:
    # 1. I am avoiding using of os.environ.get("NODE_TYPE", "") because we can depricate this env which is hardcoded in
    # training build image. This is currently used in the multi-part s3 upload.
    # 2. After refactoring I hope there wont be need to check if run on sagemaker or robomaker.
    # 3. For backward compatability and do not want to fail if the cloud team does not pass these kms arn
    #
    # SM_TRAINING_ENV env is only present in sagemaker
    hyperparams = os.environ.get('SM_TRAINING_ENV', '')

    # Validation worker will store KMS Arn to S3_KMS_CMK_ARN_ENV environment variable
    # if value is passed from cloud service.
    s3_kms_cmk_arn = os.environ.get(S3_KMS_CMK_ARN_ENV, None)
    if not s3_kms_cmk_arn:
        if hyperparams:
            hyperparams_dict = json.loads(hyperparams)
            if HYPERPARAMETERS in hyperparams_dict and SAGEMAKER_S3_KMS_CMK_ARN in hyperparams_dict[HYPERPARAMETERS]:
                s3_kms_cmk_arn = hyperparams_dict[HYPERPARAMETERS][SAGEMAKER_S3_KMS_CMK_ARN]
        else:
            # Having a try catch block if sagemaker drops SM_TRAINING_ENV and for some reason this is empty in sagemaker
            try:
                # The import rospy statement will fail in sagemaker because ROS is not installed
                import rospy
                s3_kms_cmk_arn = rospy.get_param(ROBOMAKER_S3_KMS_CMK_ARN, None)
            except Exception:
                pass
    s3_extra_args = {S3KmsEncryption.ACL.value: S3KmsEncryption.BUCKET_OWNER_FULL_CONTROL.value}
    if s3_kms_cmk_arn:
        s3_extra_args[S3KmsEncryption.SERVER_SIDE_ENCRYPTION.value] = S3KmsEncryption.AWS_KMS.value
        s3_extra_args[S3KmsEncryption.SSE_KMS_KEY_ID.value] = s3_kms_cmk_arn
    return s3_extra_args

def test_internet_connection(aws_region):
    """
    Recently came across faults because of old VPC stacks trying to use the deepracer service.
    When tried to download the model_metadata.json. The s3 fails with connection time out.
    To avoid this and give the user proper message, having this logic.
    """
    try:
        session = boto3.session.Session()
        ec2_client = session.client('ec2', aws_region)
        logger.info('Checking internet connection...')
        response = ec2_client.describe_vpcs()
        if not response['Vpcs']:
            log_and_exit("No VPC attached to instance",
                         SIMAPP_ENVIRONMENT_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
        logger.info('Verified internet connection')
    except botocore.exceptions.EndpointConnectionError:
        log_and_exit("No Internet connection or ec2 service unavailable",
                     SIMAPP_ENVIRONMENT_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)
    except botocore.exceptions.ClientError as ex:
        log_and_exit("Issue with your current VPC stack and IAM roles.\
                      You might need to reset your account resources: {}".format(ex),
                     SIMAPP_ENVIRONMENT_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_400)
    except botocore.exceptions.ConnectTimeoutError as ex:
        log_and_exit("Issue with your current VPC stack and IAM roles.\
                      You might need to reset your account resources: {}".format(ex),
                     SIMAPP_ENVIRONMENT_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as ex:
        log_and_exit("Issue with your current VPC stack and IAM roles.\
                      You might need to reset your account resources: {}".format(ex),
                     SIMAPP_ENVIRONMENT_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)

def get_sagemaker_profiler_env():
    """ Read the sagemaker profiler environment variables """
    is_profiler_on, profiler_s3_bucker, profiler_s3_prefix = False, None, None
    hyperparams = os.environ.get('SM_TRAINING_ENV', '')
    hyperparams_dict = json.loads(hyperparams)
    if HYPERPARAMETERS in hyperparams_dict and SAGEMAKER_IS_PROFILER_ON in hyperparams_dict[HYPERPARAMETERS]:
        is_profiler_on = hyperparams_dict[HYPERPARAMETERS][SAGEMAKER_IS_PROFILER_ON]
        profiler_s3_bucker = hyperparams_dict[HYPERPARAMETERS][SAGEMAKER_PROFILER_S3_BUCKET]
        profiler_s3_prefix = hyperparams_dict[HYPERPARAMETERS][SAGEMAKER_PROFILER_S3_PREFIX]
    return (is_profiler_on, profiler_s3_bucker, profiler_s3_prefix)

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

class Profiler(object):
    """ Class to profile the specific code.
    """
    _file_count = count(0)
    _profiler = None
    _profiler_owner = None

    def __init__(self, s3_bucket, s3_prefix, output_local_path, enable_profiling=False):
        self._enable_profiling = enable_profiling
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.output_local_path = output_local_path
        self.file_count = next(self._file_count)

    def __enter__(self):
        self.start()

    def __exit__(self, type_val, value, traceback):
        self.stop()

    @property
    def enable_profiling(self):
        """ Property to enable profiling

        Returns:
            (bool): True if profiler is enabled
        """
        return self._enable_profiling

    @enable_profiling.setter
    def enable_profiling(self, val):
        self._enable_profiling = val

    def start(self):
        """ Start the profiler """
        if self.enable_profiling:
            if not self._profiler:
                self._profiler = cProfile.Profile()
                self._profiler.enable()
                self._profiler_owner = self
            else:
                raise GenericException('Profiler is in use!')

    def stop(self):
        """ Stop the profiler and upload the data to S3 """
        if self._profiler_owner == self:
            if self._profiler:
                self._profiler.disable()
                self._profiler.dump_stats(self.output_local_path)
                s3_file_name = "{}-{}.txt".format(os.path.splitext(os.path.basename(self.output_local_path))[0],
                                                  self.file_count)
                with open(s3_file_name, 'w') as filepointer:
                    pstat_obj = pstats.Stats(self.output_local_path, stream=filepointer)
                    pstat_obj.sort_stats('cumulative')
                    pstat_obj.print_stats()
                self._upload_profile_stats_to_s3(s3_file_name)
            self._profiler = None
            self._profiler_owner = None

    def _upload_profile_stats_to_s3(self, s3_file_name):
        """ Upload the profiler information to s3 bucket

        Arguments:
            s3_file_name (str): File name of the profiler in S3
        """
        try:
            session = boto3.Session()
            s3_client = session.client('s3', config=get_boto_config())
            s3_extra_args = get_s3_kms_extra_args()
            s3_client.upload_file(Filename=s3_file_name, Bucket=self.s3_bucket,
                                  Key=os.path.join(self.s3_prefix, s3_file_name),
                                  ExtraArgs=s3_extra_args)
        except botocore.exceptions.ClientError as ex:
            log_and_exit("Unable to upload profiler data: {}, {}".format(self.s3_prefix,
                                                                         ex.response['Error']['Code']),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception as ex:
            log_and_exit("Unable to upload profiler data: {}".format(ex),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
