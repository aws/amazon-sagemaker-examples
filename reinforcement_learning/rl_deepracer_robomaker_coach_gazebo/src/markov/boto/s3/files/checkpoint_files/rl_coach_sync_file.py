"""This module implements rl coach sync file specifically"""

import io
import logging
import os

import botocore
from markov.boto.s3.constants import SYNC_FILES_LOCAL_PATH_FORMAT_DICT, SYNC_FILES_POSTFIX_DICT
from markov.boto.s3.s3_client import S3Client
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_S3_DATA_STORE_EXCEPTION,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class RlCoachSyncFile:
    """This class is for rl coach sync file: .finished, .lock, and .ready"""

    def __init__(
        self,
        syncfile_type,
        bucket,
        s3_prefix,
        region_name="us-east-1",
        local_dir="./checkpoint",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
    ):
        """This class is for rl coach sync file: .finished, .lock, and .ready

        Args:
            syncfile_type (str): sync file type
            bucket (str): S3 bucket string
            s3_prefix (str): S3 prefix string
            local_dir (str): local file directory
            checkpoint_dir (str): checkpoint directory
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry
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
        self._syncfile_type = syncfile_type
        self._bucket = bucket
        # deepracer checkpoint json s3 key
        self._s3_key = os.path.normpath(
            os.path.join(s3_prefix, SYNC_FILES_POSTFIX_DICT[syncfile_type])
        )
        # deepracer checkpoint json local path
        self._local_path = os.path.normpath(
            SYNC_FILES_LOCAL_PATH_FORMAT_DICT[syncfile_type].format(local_dir)
        )
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)

    @property
    def local_path(self):
        """Return local path in string"""
        return self._local_path

    def list(self):
        """List sync file"""
        return self._s3_client.list_objects_v2(bucket=self._bucket, prefix=self._s3_key)

    def persist(self, s3_kms_extra_args):
        """persist sync file into s3 bucket by writing it locally first

        Args:
            s3_kms_extra_args (dict): s3 key management service extra argument
        """
        try:
            # make local dir is missing
            local_dir = os.path.dirname(self._local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # persist to s3
            self._s3_client.upload_fileobj(
                bucket=self._bucket,
                s3_key=self._s3_key,
                fileobj=io.BytesIO(b""),
                s3_kms_extra_args=s3_kms_extra_args,
            )
            LOG.info(
                "[s3] Successfully uploaded {} to \
                     s3 bucket {} with s3 key {}.".format(
                    self._syncfile_type, self._bucket, self._s3_key
                )
            )
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to upload {} file".format(self._syncfile_type),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in uploading {} file {}".format(self._syncfile_type, ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def download(self):
        """download the sync file from s3 bucket"""
        self._download()

    def delete(self):
        """delete the sync file from  s3 bucket"""
        self._s3_client.delete_object(bucket=self._bucket, s3_key=self._s3_key)

    def _download(self):
        """download file from s3 bucket"""
        local_dir = os.path.dirname(self._local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        self._s3_client.download_file(
            bucket=self._bucket, s3_key=self._s3_key, local_path=self._local_path
        )
        LOG.info(
            "[s3] Successfully downloaded {} from \
                 s3 key {} to local {}.".format(
                self._syncfile_type, self._s3_key, self._local_path
            )
        )
