"""This module implements s3 client for virtual event best sector time"""

import json
import logging
import os
import sys

import botocore
from markov.boto.s3.constants import (
    PYTHON_2,
    PYTHON_3,
    SECTOR_TIME_FORMAT_DICT,
    SECTOR_TIME_S3_POSTFIX,
    SECTOR_X_FORMAT,
    TrackSectorTime,
)
from markov.boto.s3.s3_client import S3Client
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_EVENT_SYSTEM_ERROR,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericNonFatalException
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class VirtualEventBestSectorTime:
    """virtual event best sector time upload and download"""

    def __init__(
        self,
        bucket,
        s3_key,
        region_name="us-east-1",
        local_path="./custom_files/best_sector_time.json",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
    ):
        """virtual event best sector time upload and download

        Args:
            bucket (str): s3 bucket
            s3_prefix (str): s3 prefix
            s3_key (str): s3 key
            region_name (str): s3 region name
            local_path (str): best sector time json file local path
            max_retry_attempts (int): maximum retry attempts
            backoff_time_sec (float): retry backoff time in seconds
        """
        if not s3_key or not bucket:
            log_and_exit(
                "virtual event best sector time S3 key or bucket not available for S3. \
                         bucket: {}, key: {}".format(
                    bucket, s3_key
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._python_version = sys.version_info[0]
        self._bucket = bucket
        self._s3_key = s3_key
        self._local_path = local_path
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)
        self._best_sector_time = dict()

    def list(self):
        """List best sector time json file"""
        return self._s3_client.list_objects_v2(bucket=self._bucket, prefix=self._s3_key)

    def persist(self, body, s3_kms_extra_args):
        """upload virtual event best sector time into s3 bucket

        Args:
            body (json): text body in json format
            s3_kms_extra_args (dict): s3 key management service extra argument

        """
        try:
            # for python 2 and 3 compatibility
            if self._python_version <= PYTHON_2:
                body = bytes(body).encode("utf-8")
            else:
                body = bytes(body, "utf-8")
            # if retry failed, s3_client put_object will log and exit 500
            self._s3_client.put_object(
                bucket=self._bucket,
                s3_key=self._s3_key,
                body=body,
                s3_kms_extra_args=s3_kms_extra_args,
            )
            LOG.info(
                "[s3] Successfully uploaded virtual event best sector time to \
                     s3 bucket {} with s3 key {}.".format(
                    self._bucket, self._s3_key
                )
            )
        except Exception as ex:
            LOG.error("Failed to upload virtual event best sector time {}.".format(ex))

    def get_sector_time(self, num_sectors):
        """always download lateset best sector time from s3

        Args:
            num_sectors (int): number of sectors

        Returns:
            dict: best sector times in dict with key of sector1, sector2, and sector3
            and value of milli secont in integer

        """
        self._download(num_sectors)
        return self._best_sector_time

    def _download(self, num_sectors):
        """download best sector time json file

        Best sector time will be downloaded from s3 at the very
        beginning of each racer starting to race.

        If download from s3 is successful,
        we will return the value dict from s3. Then, during sector color overlay,
        we will compare best session, best personal, and current personal to decide
        the overlay color. However, if download failed, we will initialize all best
        sector time to None and then we will not overlay any sector color for that
        specific racer. When the next racer come in, we will download again.

        Args:
            num_sectors (int): total number of sectors to display

        """

        # check and make local directory
        local_dir = os.path.dirname(self._local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Download the best sector time with retry
        try:
            self._s3_client.download_file(
                bucket=self._bucket, s3_key=self._s3_key, local_path=self._local_path
            )
            LOG.info(
                "[s3] Successfully downloaded best sector time from \
                 s3 key {} to local {}.".format(
                    self._s3_key, self._local_path
                )
            )
            with open(self._local_path) as file:
                times = json.load(file)
                for idx in range(num_sectors):
                    sector = SECTOR_X_FORMAT.format(idx + 1)
                    self._best_sector_time[
                        SECTOR_TIME_FORMAT_DICT[TrackSectorTime.BEST_SESSION].format(sector)
                    ] = times[sector]
                return self._best_sector_time
        except Exception as ex:
            LOG.error(
                "[s3] Exception occurred while getting best sector time from s3 %s use default", ex
            )
            # default sector times to None and not plot sectors if download failed
            for idx in range(num_sectors):
                sector = SECTOR_X_FORMAT.format(idx + 1)
                self._best_sector_time[
                    SECTOR_TIME_FORMAT_DICT[TrackSectorTime.BEST_SESSION].format(sector)
                ] = None
            return self._best_sector_time
