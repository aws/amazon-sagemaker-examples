"""This module implements deepracer checkpoint json file specifically"""

import json
import logging
import os

import botocore
from markov.boto.s3.constants import (
    BEST_CHECKPOINT,
    DEEPRACER_CHECKPOINT_KEY_POSTFIX,
    DEEPRACER_CHECKPOINT_LOCAL_PATH_FORMAT,
    LAST_CHECKPOINT,
)
from markov.boto.s3.s3_client import S3Client
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericNonFatalException
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class DeepracerCheckpointJson:
    """This class is for deepracer checkpoint json file upload and download"""

    def __init__(
        self,
        bucket,
        s3_prefix,
        region_name="us-east-1",
        local_dir=".checkpoint/agent",
        max_retry_attempts=0,
        backoff_time_sec=1.0,
        log_and_cont: bool = False,
    ):
        """This class is for deepracer checkpoint json file upload and download

        Args:
            bucket (str): S3 bucket string.
            s3_prefix (str): S3 prefix string.
            region_name (str): S3 region name.
                               Defaults to 'us-east-1'.
            local_dir (str, optional): Local file directory.
                                       Defaults to '.checkpoint/agent'.
            max_retry_attempts (int, optional): Maximum number of retry attempts for S3 download/upload.
                                                Defaults to 0.
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
        # deepracer checkpoint json s3 key
        self._s3_key = os.path.normpath(os.path.join(s3_prefix, DEEPRACER_CHECKPOINT_KEY_POSTFIX))
        # deepracer checkpoint json local path
        self._local_path = os.path.normpath(
            DEEPRACER_CHECKPOINT_LOCAL_PATH_FORMAT.format(local_dir)
        )
        self._s3_client = S3Client(
            region_name, max_retry_attempts, backoff_time_sec, log_and_cont=log_and_cont
        )

    def _get_deepracer_checkpoint(self, checkpoint_type):
        """Returns the deepracer checkpoint stored in the checkpoint json

        Args:
           checkpoint_type (str): BEST_CHECKPOINT/LAST_CHECKPOINT string
        """
        try:
            # Download deepracer checkpoint
            self._download()
        except GenericNonFatalException as ex:
            raise ex
        except botocore.exceptions.ClientError as err:
            if err.response["Error"]["Code"] == "404":
                LOG.info("Unable to find deepracer checkpoint json")
                return None
            else:
                log_and_exit(
                    "Unable to download deepracer checkpoint json: {}, {}".format(
                        self._bucket, err.response["Error"]["Code"]
                    ),
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_400,
                )
        except Exception as ex:
            log_and_exit(
                "Can't download deepracer checkpoint json: {}".format(ex),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        try:
            with open(self._local_path) as deepracer_checkpoint_file:
                checkpoint_name = json.load(deepracer_checkpoint_file)[checkpoint_type]["name"]
                if not checkpoint_name:
                    raise Exception("No deepracer checkpoint json recorded")
            os.remove(self._local_path)
        except Exception as ex:
            LOG.info("Unable to parse deepracer checkpoint json: {}".format(ex))
            return None
        return checkpoint_name

    def get_deepracer_best_checkpoint(self):
        """get the best deepracer checkpoint name

        Returns:
            str: best checkpoint name string
        """
        return self._get_deepracer_checkpoint(BEST_CHECKPOINT)

    def get_deepracer_last_checkpoint(self):
        """get the last deepracer checkpoint name

        Returns:
            str: last checkpoint name string
        """
        return self._get_deepracer_checkpoint(LAST_CHECKPOINT)

    def get_deepracer_best_checkpoint_number(self):
        """get the best deepracer checkpoint number. If there is no best checkpoint,
        it will return the last checkpoint. If there is no last checkpoint, it will return -1.

        Returns:
            int: best checkpoint number in integer
        """
        checkpoint_num = -1
        best_checkpoint_name = self._get_deepracer_checkpoint(BEST_CHECKPOINT)
        if best_checkpoint_name and len(best_checkpoint_name.split("_Step")) > 0:
            checkpoint_num = int(best_checkpoint_name.split("_Step")[0])
        else:
            LOG.info(
                "Unable to find the best deepracer checkpoint number. Getting the last checkpoint number"
            )
            checkpoint_num = self.get_deepracer_last_checkpoint_number()
        return checkpoint_num

    def get_deepracer_last_checkpoint_number(self):
        """get the last checkpoint number. If there is not last checkpoint, it will return -1

        Returns:
            int: last checkpoint number in integer
        """
        checkpoint_num = -1
        last_checkpoint_name = self._get_deepracer_checkpoint(LAST_CHECKPOINT)
        # Verify if the last checkpoint name is present and is in right format
        if last_checkpoint_name and len(last_checkpoint_name.split("_Step")) > 0:
            checkpoint_num = int(last_checkpoint_name.split("_Step")[0])
        else:
            LOG.info("Unable to find the last deepracer checkpoint number.")
        return checkpoint_num

    def _download(self):
        """download deepracer checkpoint json file from s3 bucket"""
        local_dir = os.path.dirname(self._local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        self._s3_client.download_file(
            bucket=self._bucket, s3_key=self._s3_key, local_path=self._local_path
        )
        LOG.info(
            "[s3] Successfully downloaded deepracer checkpoint json from \
                 s3 key {} to local {}.".format(
                self._s3_key, self._local_path
            )
        )

    def persist(self, body, s3_kms_extra_args):
        """upload metrics into s3 bucket

        Args:
            body (str): s3 upload string
            s3_kms_extra_args (dict): s3 key management service extra argument

        """
        self._s3_client.put_object(
            bucket=self._bucket,
            s3_key=self._s3_key,
            body=bytes(body, encoding="utf-8"),
            s3_kms_extra_args=s3_kms_extra_args,
        )
        LOG.info(
            "[s3] Successfully uploaded deepracer checkpoint to \
                 s3 bucket {} with s3 key {}.".format(
                self._bucket, self._s3_key
            )
        )
