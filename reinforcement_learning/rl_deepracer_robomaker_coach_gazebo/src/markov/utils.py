import cProfile
import io
import json
import logging
import os
import pstats
import random
import re
import shutil
import signal
import socket
import threading
import time
from itertools import count

import boto3
import botocore
from markov.constants import (
    BEST_CHECKPOINT,
    CONNECT_TIMEOUT,
    HYPERPARAMETERS,
    LAST_CHECKPOINT,
    NUM_RETRIES,
    ROBOMAKER_CANCEL_JOB_WAIT_TIME,
    ROBOMAKER_S3_KMS_CMK_ARN,
    S3_KMS_CMK_ARN_ENV,
    SAGEMAKER_IS_PROFILER_ON,
    SAGEMAKER_PROFILER_S3_BUCKET,
    SAGEMAKER_PROFILER_S3_PREFIX,
    SAGEMAKER_S3_KMS_CMK_ARN,
    S3KmsEncryption,
)
from markov.log_handler.constants import (
    SIMAPP_DONE_EXIT,
    SIMAPP_ENVIRONMENT_EXCEPTION,
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_S3_DATA_STORE_EXCEPTION,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericException
from markov.log_handler.exception_handler import log_and_exit, simapp_exit_gracefully
from markov.log_handler.logger import Logger

logger = Logger(__name__, logging.INFO).get_logger()


def is_int_repr(val):
    """check whether input is int or not

    Args:
        val (str): str input

    Returns:
        bool: True if string can be convert to int, False otherwise
    """
    try:
        int(val)
        return True
    except ValueError:
        return False


def force_list(val):
    if type(val) is not list:
        val = [val]
    return val


def get_boto_config():
    """Returns a botocore config object which specifies the number of times to retry"""
    return botocore.config.Config(
        retries=dict(max_attempts=NUM_RETRIES), connect_timeout=CONNECT_TIMEOUT
    )


def cancel_simulation_job(backoff_time_sec=1.0, max_retry_attempts=5):
    """ros service call to cancel simulation job

    Args:
        backoff_time_sec(float): backoff time in seconds
        max_retry_attempts (int): maximum number of retry
    """
    import rospy
    from markov.rospy_wrappers import ServiceProxyWrapper
    from robomaker_simulation_msgs.srv import Cancel

    requestCancel = ServiceProxyWrapper("/robomaker/job/cancel", Cancel)

    try_count = 0
    while True:
        try_count += 1
        response = requestCancel()
        logger.info(
            "cancel_simulation_job from ros service call response: {}".format(response.success)
        )
        if response and response.success:
            time.sleep(ROBOMAKER_CANCEL_JOB_WAIT_TIME)
            return
        if try_count > max_retry_attempts:
            simapp_exit_gracefully()
        backoff_time = (pow(try_count, 2) + random.random()) * backoff_time_sec
        time.sleep(backoff_time)


def str2bool(flag):
    """bool: convert flag to boolean if it is string and return it else return its initial bool value"""
    if not isinstance(flag, bool):
        if flag.lower() == "false":
            flag = False
        elif flag.lower() == "true":
            flag = True
    return flag


def str_to_done_condition(done_condition):
    if done_condition == any or done_condition == all:
        return done_condition
    elif done_condition.lower().strip() == "all":
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
        error_string = (
            "Environment Error: Could not retrieve IP address \
        for %s in past %s seconds."
            % (host_name, timeout)
        )
        log_and_exit(error_string, SIMAPP_ENVIRONMENT_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)
    return ip_address


def is_user_error(error):
    """Helper method that determines whether a value error is caused by an invalid checkpoint
    or model_metadata by looking for keywords in the exception message
    error - Python exception object, which produces a message by implementing __str__
    """
    # These are the key words, which if present indicate that the tensorflow saver was unable
    # to restore a checkpoint because it does not match the graph or unable to load
    # model_metadata due to invalid json format
    keys = ["tensor", "shape", "checksum", "checkpoint", "model_metadata"]
    return any(key in str(error).lower() for key in keys)


def get_video_display_name():
    """Based on the job type display the appropriate name on the mp4.
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
        racer_names.append("racecar")
    else:
        for idx in range(racecar_num):
            racer_names.append("racecar_" + str(idx))
    return racer_names


def get_racecar_idx(racecar_name):
    try:
        racecar_name_list = racecar_name.split("_")
        if len(racecar_name_list) == 1:
            return None
        racecar_num = racecar_name_list[1]
        return int(racecar_num)
    except Exception as ex:
        log_and_exit(
            "racecar name should be in format racecar_x. However, get {}".format(racecar_name),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )


def get_s3_extra_args(s3_kms_cmk_arn=None):
    """Generate the s3 extra arg dict with the s3 kms cmk arn passed in.

    Args:
        s3_kms_cmk_arn (str, optional): The kms arn to use for contructing
                                        the s3 extra arguments.
                                        Defaults to None.

    Returns:
        dict: A dictionary for s3 extra arguments.
    """
    s3_extra_args = {S3KmsEncryption.ACL.value: S3KmsEncryption.BUCKET_OWNER_FULL_CONTROL.value}
    if s3_kms_cmk_arn is not None:
        s3_extra_args[S3KmsEncryption.SERVER_SIDE_ENCRYPTION.value] = S3KmsEncryption.AWS_KMS.value
        s3_extra_args[S3KmsEncryption.SSE_KMS_KEY_ID.value] = s3_kms_cmk_arn
    return s3_extra_args


def get_s3_kms_extra_args():
    """Since the S3Client class is called by both robomaker and sagemaker. One has to know
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
    hyperparams = os.environ.get("SM_TRAINING_ENV", "")

    # Validation worker will store KMS Arn to S3_KMS_CMK_ARN_ENV environment variable
    # if value is passed from cloud service.
    s3_kms_cmk_arn = os.environ.get(S3_KMS_CMK_ARN_ENV, None)
    if not s3_kms_cmk_arn:
        if hyperparams:
            hyperparams_dict = json.loads(hyperparams)
            if (
                HYPERPARAMETERS in hyperparams_dict
                and SAGEMAKER_S3_KMS_CMK_ARN in hyperparams_dict[HYPERPARAMETERS]
            ):
                s3_kms_cmk_arn = hyperparams_dict[HYPERPARAMETERS][SAGEMAKER_S3_KMS_CMK_ARN]
        else:
            # Having a try catch block if sagemaker drops SM_TRAINING_ENV and for some reason this is empty in sagemaker
            try:
                # The import rospy statement will fail in sagemaker because ROS is not installed
                import rospy

                s3_kms_cmk_arn = rospy.get_param(ROBOMAKER_S3_KMS_CMK_ARN, None)
            except Exception:
                pass
    return get_s3_extra_args(s3_kms_cmk_arn)


def test_internet_connection(aws_region):
    """
    Recently came across faults because of old VPC stacks trying to use the deepracer service.
    When tried to download the model_metadata.json. The s3 fails with connection time out.
    To avoid this and give the user proper message, having this logic.
    """
    try:
        session = boto3.session.Session()
        ec2_client = session.client("ec2", aws_region)
        logger.info("Checking internet connection...")
        response = ec2_client.describe_vpcs()
        if not response["Vpcs"]:
            log_and_exit(
                "No VPC attached to instance",
                SIMAPP_ENVIRONMENT_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        logger.info("Verified internet connection")
    except botocore.exceptions.EndpointConnectionError:
        log_and_exit(
            "No Internet connection or ec2 service unavailable",
            SIMAPP_ENVIRONMENT_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
    except botocore.exceptions.ClientError as ex:
        log_and_exit(
            "Issue with your current VPC stack and IAM roles.\
                      You might need to reset your account resources: {}".format(
                ex
            ),
            SIMAPP_ENVIRONMENT_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
    except botocore.exceptions.ConnectTimeoutError as ex:
        log_and_exit(
            "Issue with your current VPC stack and IAM roles.\
                      You might need to reset your account resources: {}".format(
                ex
            ),
            SIMAPP_ENVIRONMENT_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
    except Exception as ex:
        log_and_exit(
            "Issue with your current VPC stack and IAM roles.\
                      You might need to reset your account resources: {}".format(
                ex
            ),
            SIMAPP_ENVIRONMENT_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )


def get_sagemaker_profiler_env():
    """Read the sagemaker profiler environment variables"""
    is_profiler_on, profiler_s3_bucker, profiler_s3_prefix = False, None, None
    hyperparams = os.environ.get("SM_TRAINING_ENV", "")
    hyperparams_dict = json.loads(hyperparams)
    if (
        HYPERPARAMETERS in hyperparams_dict
        and SAGEMAKER_IS_PROFILER_ON in hyperparams_dict[HYPERPARAMETERS]
    ):
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
        simapp_exit_gracefully(simapp_exit=SIMAPP_DONE_EXIT)


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
    """Class to profile the specific code."""

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
        """Property to enable profiling

        Returns:
            (bool): True if profiler is enabled
        """
        return self._enable_profiling

    @enable_profiling.setter
    def enable_profiling(self, val):
        self._enable_profiling = val

    def start(self):
        """Start the profiler"""
        if self.enable_profiling:
            if not self._profiler:
                self._profiler = cProfile.Profile()
                self._profiler.enable()
                self._profiler_owner = self
            else:
                raise GenericException("Profiler is in use!")

    def stop(self):
        """Stop the profiler and upload the data to S3"""
        if self._profiler_owner == self:
            if self._profiler:
                self._profiler.disable()
                self._profiler.dump_stats(self.output_local_path)
                s3_file_name = "{}-{}.txt".format(
                    os.path.splitext(os.path.basename(self.output_local_path))[0], self.file_count
                )
                with open(s3_file_name, "w") as filepointer:
                    pstat_obj = pstats.Stats(self.output_local_path, stream=filepointer)
                    pstat_obj.sort_stats("cumulative")
                    pstat_obj.print_stats()
                self._upload_profile_stats_to_s3(s3_file_name)
            self._profiler = None
            self._profiler_owner = None

    def _upload_profile_stats_to_s3(self, s3_file_name):
        """Upload the profiler information to s3 bucket

        Arguments:
            s3_file_name (str): File name of the profiler in S3
        """
        try:
            session = boto3.Session()
            s3_client = session.client("s3", config=get_boto_config())
            s3_extra_args = get_s3_kms_extra_args()
            s3_client.upload_file(
                Filename=s3_file_name,
                Bucket=self.s3_bucket,
                Key=os.path.join(self.s3_prefix, s3_file_name),
                ExtraArgs=s3_extra_args,
            )
        except botocore.exceptions.ClientError as ex:
            log_and_exit(
                "Unable to upload profiler data: {}, {}".format(
                    self.s3_prefix, ex.response["Error"]["Code"]
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Unable to upload profiler data: {}".format(ex),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
