"""This module implements rl coach coach_checkpoint file specifically"""

import logging
import os
import re

import botocore
from markov.boto.s3.constants import (
    BEST_CHECKPOINT,
    COACH_CHECKPOINT_LOCAL_PATH_FORMAT,
    COACH_CHECKPOINT_POSTFIX,
    LAST_CHECKPOINT,
    OLD_COACH_CHECKPOINT_LOCAL_PATH_FORMAT,
    OLD_COACH_CHECKPOINT_POSTFIX,
    TEMP_COACH_CHECKPOINT_LOCAL_PATH_FORMAT,
)
from markov.boto.s3.s3_client import S3Client
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.utils import get_s3_kms_extra_args
from rl_coach.checkpoint import CheckpointStateFile

LOG = Logger(__name__, logging.INFO).get_logger()


class RLCoachCheckpoint:
    """This class is for RL coach checkpoint file"""

    def __init__(
        self,
        bucket,
        s3_prefix,
        region_name="us-east-1",
        local_dir="./checkpoint/agent",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
        log_and_cont: bool = False,
    ):
        """This class is for RL coach checkpoint file

        Args:
            bucket (str): S3 bucket string.
            s3_prefix (str): S3 prefix string.
            region_name (str): S3 region name.
                               Defaults to 'us-east-1'.
            local_dir (str, optional): Local file directory.
                                       Defaults to '.checkpoint/agent'.
            max_retry_attempts (int, optional): Maximum number of retry attempts for S3 download/upload.
                                                Defaults to 5.
            backoff_time_sec (float, optional): Backoff second between each retry.
                                                Defaults to 1.0.
            log_and_cont (bool, optional): Log the error and continue with the flow.
                                           Defaults to False.
        """
        if not bucket or not s3_prefix:
            log_and_exit(
                "checkpoint S3 prefix or bucket not available for S3. \
                         bucket: {}, prefix {}".format(
                    bucket, s3_prefix
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._bucket = bucket
        # coach checkpoint s3 key
        self._s3_key = os.path.normpath(os.path.join(s3_prefix, COACH_CHECKPOINT_POSTFIX))
        # coach checkpoint local path
        self._local_path = os.path.normpath(COACH_CHECKPOINT_LOCAL_PATH_FORMAT.format(local_dir))
        # coach checkpoint local temp path
        self._temp_local_path = os.path.normpath(
            TEMP_COACH_CHECKPOINT_LOCAL_PATH_FORMAT.format(local_dir)
        )
        # old coach checkpoint s3 key to handle backward compatibility
        self._old_s3_key = os.path.normpath(os.path.join(s3_prefix, OLD_COACH_CHECKPOINT_POSTFIX))
        # old coach checkpoint local path to handle backward compatibility
        self._old_local_path = os.path.normpath(
            OLD_COACH_CHECKPOINT_LOCAL_PATH_FORMAT.format(local_dir)
        )
        # coach checkpoint state file from rl coach
        self._coach_checkpoint_state_file = CheckpointStateFile(os.path.dirname(self._local_path))
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec, log_and_cont)

    @property
    def s3_dir(self):
        """Return s3 directory in string"""
        return os.path.dirname(self._s3_key)

    @property
    def local_path(self):
        """Return local path in string"""
        return self._local_path

    @property
    def coach_checkpoint_state_file(self):
        """Return RL coach CheckpointStateFile class instance"""
        return self._coach_checkpoint_state_file

    def get(self):
        """get rl coach checkpoint"""
        self._download()

    def _download(self):
        """download rl coach checkpoint from s3 bucket"""
        local_dir = os.path.dirname(self._local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        self._s3_client.download_file(
            bucket=self._bucket, s3_key=self._s3_key, local_path=self._local_path
        )
        LOG.info(
            "[s3] Successfully downloaded rl coach checkpoint from \
                 s3 key {} to local {}.".format(
                self._s3_key, self._local_path
            )
        )

    def persist(self, s3_kms_extra_args):
        """upload rl coach checkpoint to s3 bucket

        Args:
            s3_kms_extra_args (dict): s3 key management service extra argument
        """
        self._s3_client.upload_file(
            bucket=self._bucket,
            s3_key=self._s3_key,
            local_path=self._local_path,
            s3_kms_extra_args=s3_kms_extra_args,
        )
        LOG.info(
            "[s3] Successfully uploaded coach checkpoint to \
                  s3 bucket {} with s3 key {}.".format(
                self._bucket, self._s3_key
            )
        )

    def _persist_temp_coach_checkpoint(self, s3_kms_extra_args):
        """upload rl temp coach checkpoint to s3 bucket for tensorflow model selection
        and compatibility

        Args:
            s3_kms_extra_args (dict): s3 key management service extra argument
        """
        self._s3_client.upload_file(
            bucket=self._bucket,
            s3_key=self._s3_key,
            local_path=self._temp_local_path,
            s3_kms_extra_args=s3_kms_extra_args,
        )
        LOG.info(
            "[s3] Successfully uploaded temp coach checkpoint to \
                  s3 bucket {} with s3 key {}.".format(
                self._bucket, self._s3_key
            )
        )

    def update(self, model_checkpoint_name, s3_kms_extra_args):
        """update local coach checkpoint file and upload to s3 bucket

        Args:
            model_checkpoint_name (str): model checkpoint string
            s3_kms_extra_args (dict): s3 key management service extra argument

        Returns:
            bool: True if update rl coach checkpoint successfully, False, otherwise.
            This is mainly for validation worker to validate the model.
        """
        try:
            # check model checkpoint is present and is type string
            if model_checkpoint_name is None or not isinstance(model_checkpoint_name, str):
                LOG.info(
                    "Exit because model_checkpoint_name is {} of type {}".format(
                        model_checkpoint_name, type(model_checkpoint_name)
                    )
                )
                return False
            with open(self._temp_local_path, "+w") as new_ckpnt:
                new_ckpnt.write(model_checkpoint_name)
            # upload local temp rl coach checkpoint
            self._persist_temp_coach_checkpoint(s3_kms_extra_args=s3_kms_extra_args)
            # remove local temp rl coach checkpoint
            os.remove(self._temp_local_path)
            return True
        except botocore.exceptions.ClientError as err:
            log_and_exit(
                "Unable to upload checkpoint: {}, {}".format(
                    self._bucket, err.response["Error"]["Code"]
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in uploading checkpoint: {}".format(ex),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def is_compatible(self):
        """check whether rl coach checkpoint is compatiable by checking
        whether there is a .coach_checkpoint file presetn in the expected s3 bucket

        Returns:
            bool: True is coach checkpoint is compatiable, False otherwise
        """
        try:
            coach_checkpoint_dir, coach_checkpoint_filename = os.path.split(self._s3_key)
            response = self._s3_client.list_objects_v2(
                bucket=self._bucket, prefix=coach_checkpoint_dir
            )
            if "Contents" not in response:
                # Customer deleted checkpoint file.
                log_and_exit(
                    "No objects found: {}".format(self._bucket),
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_400,
                )

            return any(
                list(
                    map(
                        lambda obj: os.path.split(obj["Key"])[1] == coach_checkpoint_filename,
                        response["Contents"],
                    )
                )
            )
        except botocore.exceptions.ClientError as e:
            log_and_exit(
                "No objects found: {}, {}".format(self._bucket, e.response["Error"]["Code"]),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as e:
            log_and_exit(
                "Exception in checking for current checkpoint key: {}".format(e),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def make_compatible(self, syncfile_ready):
        """update coach checkpoint file to make it compatible

        Args:
            syncfile_ready (RlCoachSyncFile): RlCoachSyncFile class instance for .ready file
        """
        try:
            # download old coach checkpoint
            self._s3_client.download_file(
                bucket=self._bucket, s3_key=self._old_s3_key, local_path=self._old_local_path
            )
            # parse old coach checkpoint
            with open(self._old_local_path) as old_coach_checkpoint_file:
                coach_checkpoint_value = re.findall(
                    r'"(.*?)"', old_coach_checkpoint_file.readline()
                )
            if len(coach_checkpoint_value) != 1:
                log_and_exit(
                    "No checkpoint file found",
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_400,
                )
            # remove old local coach checkpoint
            os.remove(self._old_local_path)
            # Upload ready file so that the system can gab the checkpoints
            syncfile_ready.persist(s3_kms_extra_args=get_s3_kms_extra_args())
            # write new temp coach checkpoint file
            with open(self._temp_local_path, "w+") as new_coach_checkpoint_file:
                new_coach_checkpoint_file.write(coach_checkpoint_value[0])
            # upload new temp coach checkpoint file
            self._persist_temp_coach_checkpoint(s3_kms_extra_args=get_s3_kms_extra_args())
            # remove new temp local coach checkpoint
            os.remove(self._temp_local_path)
        except botocore.exceptions.ClientError as e:
            log_and_exit(
                "Unable to make model compatible: {}, {}".format(
                    self._bucket, e.response["Error"]["Code"]
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as e:
            log_and_exit(
                "Exception in making model compatible: {}".format(e),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
