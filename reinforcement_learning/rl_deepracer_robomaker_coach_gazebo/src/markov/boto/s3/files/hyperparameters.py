"""This module implements hyperparameters file"""

import io
import json
import logging
import os

import botocore
from markov.boto.s3.s3_client import S3Client
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class Hyperparameters:
    """hyperparameters download and upload"""

    def __init__(
        self,
        bucket,
        s3_key,
        region_name="us-east-1",
        local_path="./custom_files/agent/hyperparameters.json",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
    ):
        """Hyperparameters upload, download, and parse

        Args:
            bucket (str): S3 bucket string
            s3_key (str): S3 key string
            local_path (str): file local path
            region_name (str): S3 region name
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry
        """

        # check s3 key and bucket exist for hyperparamer
        if not s3_key or not bucket:
            log_and_exit(
                "hyperparameters S3 key or bucket not available for S3. \
                         bucket: {}, key: {}".format(
                    bucket, s3_key
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._bucket = bucket
        # Strip the s3://<bucket> from uri, if s3_key past in as uri
        self._s3_key = s3_key.replace("s3://{}/".format(self._bucket), "")
        self._local_path = local_path
        self._hyperparameters = None
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)

    def get_hyperparameters_dict(self):
        """return the hyperparameters

        Returns:
            dict: post-processed hyperparameters as dict

        """

        if self._hyperparameters is None:
            # download hyperparameters.json
            self._download()
        return self._hyperparameters

    def persist(self, hyperparams_json, s3_kms_extra_args):
        """upload local hyperparams_json into S3 bucket

        Args:
            hyperparams_json (str): json dump format string
            s3_kms_extra_args (dict): s3 key management service extra argument

        """

        # upload hyperparameter with retry
        # if retry failed, s3_client upload_fileobj will log and exit 500
        self._s3_client.upload_fileobj(
            bucket=self._bucket,
            s3_key=self._s3_key,
            fileobj=io.BytesIO(hyperparams_json.encode()),
            s3_kms_extra_args=s3_kms_extra_args,
        )
        LOG.info(
            "[s3] Successfully uploaded hyperparameters to \
                 s3 bucket {} with s3 key {}.".format(
                self._bucket, self._s3_key
            )
        )

    def _download(self):
        """download hyperparameters.json with retry from s3 bucket"""

        # check and make local directory
        local_dir = os.path.dirname(self._local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # download hyperparameter with retry
        try:
            self._s3_client.download_file(
                bucket=self._bucket, s3_key=self._s3_key, local_path=self._local_path
            )
            LOG.info(
                "[s3] Successfully downloaded hyperparameters from \
                     s3 key {} to local {}.".format(
                    self._s3_key, self._local_path
                )
            )
        except botocore.exceptions.ClientError as err:
            log_and_exit(
                "Failed to download hyperparameters file: s3_bucket: {}, s3_key: {}, {}".format(
                    self._bucket, self._s3_key, err
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        # parse local hyperparameters.json at into dict if download successful
        try:
            with open(self._local_path) as filepointer:
                self._hyperparameters = json.load(filepointer)
        except Exception as e:
            log_and_exit(
                "Failed to open and load hyperparameters: {}".format(e),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
