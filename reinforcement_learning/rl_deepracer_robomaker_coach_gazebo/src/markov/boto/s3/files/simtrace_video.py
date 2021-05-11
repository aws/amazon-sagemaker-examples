"""This module implements s3 client for simtrace, pip, 45degrees, and topview videos"""

import logging
import os

from markov.boto.s3.constants import SIMTRACE_VIDEO_POSTFIX_DICT
from markov.boto.s3.s3_client import S3Client
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class SimtraceVideo:
    """This class is for all s3 simtrace and video upload"""

    def __init__(
        self,
        upload_type,
        bucket,
        s3_prefix,
        region_name="us-east-1",
        local_path="./custom_files/iteration_data/\
                 agent/file",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
    ):
        """This class is for all s3 simtrace and video upload

        Args:
            upload_type (str): upload simtrace or video type
            bucket (str): S3 bucket string
            s3_prefix (str): S3 prefix string
            region_name (str): S3 region name
            local_path (str): file local path
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry

        """
        self._upload_type = upload_type
        self._bucket = bucket
        self._s3_key = os.path.normpath(
            os.path.join(s3_prefix, SIMTRACE_VIDEO_POSTFIX_DICT[self._upload_type])
        )
        self._local_path = local_path
        self._upload_num = 0
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)

    def persist(self, s3_kms_extra_args):
        """persist simtrace or video into s3 bucket

        Args:
            s3_kms_extra_args(dict): s3 key management service extra argument

        """
        # upload sim trace or video
        # if retry failed, s3_client upload_file will log and exit 500
        self._s3_client.upload_file(
            bucket=self._bucket,
            s3_key=self._s3_key.format(self._upload_num),
            local_path=self._local_path,
            s3_kms_extra_args=s3_kms_extra_args,
        )
        LOG.info(
            "[s3] Successfully uploaded {} to \
             s3 bucket {} with s3 key {}.".format(
                self._upload_type, self._bucket, self._s3_key.format(self._upload_num)
            )
        )
        self._upload_num += 1
        # remove local file after upload simtrace or video to s3 bucket
        os.remove(self._local_path)
