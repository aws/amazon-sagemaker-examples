"""This module implements s3 client for metrics"""

import logging

from markov.boto.s3.s3_client import S3Client
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class Metrics:
    """metrics upload"""

    def __init__(
        self, bucket, s3_key, region_name="us-east-1", max_retry_attempts=5, backoff_time_sec=1.0
    ):
        """metrics upload

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            region_name (str): s3 region name
            max_retry_attempts (int): maximum retry attempts
            backoff_time_sec (float): retry backoff time in seconds
        """
        if not s3_key or not bucket:
            log_and_exit(
                "Metrics S3 key or bucket not available for S3. \
                         bucket: {}, key: {}".format(
                    bucket, s3_key
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._bucket = bucket
        self._s3_key = s3_key
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)

    def persist(self, body, s3_kms_extra_args):
        """upload metrics into s3 bucket

        Args:
            body (json): text body in json format
            s3_kms_extra_args (dict): s3 key management service extra argument

        """
        # if retry failed, s3_client put_object will log and exit 500
        self._s3_client.put_object(
            bucket=self._bucket,
            s3_key=self._s3_key,
            body=bytes(body, encoding="utf-8"),
            s3_kms_extra_args=s3_kms_extra_args,
        )
        LOG.info(
            "[s3] Successfully uploaded metrics to \
                 s3 bucket {} with s3 key {}.".format(
                self._bucket, self._s3_key
            )
        )
