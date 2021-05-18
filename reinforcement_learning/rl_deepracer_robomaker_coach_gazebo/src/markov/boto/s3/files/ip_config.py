"""This module implements s3 client for ip config"""

import io
import json
import logging
import os
import socket
import time

import botocore
from markov.boto.s3.constants import IP_ADDRESS_POSTFIX, IP_DONE_POSTFIX, SAGEMAKER_WAIT_TIME
from markov.boto.s3.s3_client import S3Client
from markov.log_handler.constants import (
    SIMAPP_ENVIRONMENT_EXCEPTION,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_S3_DATA_STORE_EXCEPTION,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class IpConfig:
    """ip upload, download, and parse"""

    def __init__(
        self,
        bucket,
        s3_prefix,
        region_name="us-east-1",
        local_path="./custom_files/agent/ip.json",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
    ):
        """ip upload, download, and parse

        Args:
            bucket (str): s3 bucket
            s3_prefix (str): s3 prefix
            region_name (str): s3 region name
            local_path (str): ip addres json file local path
            max_retry_attempts (int): maximum retry attempts
            backoff_time_sec (float): retry backoff time in seconds

        """
        if not s3_prefix or not bucket:
            log_and_exit(
                "Ip config S3 prefix or bucket not available for S3. \
                         bucket: {}, prefix: {}".format(
                    bucket, s3_prefix
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        self._bucket = bucket
        self._s3_ip_done_key = os.path.normpath(os.path.join(s3_prefix, IP_DONE_POSTFIX))
        self._s3_ip_address_key = os.path.normpath(os.path.join(s3_prefix, IP_ADDRESS_POSTFIX))
        self._local_path = local_path
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)
        self._ip_file = None

    def get_ip_config(self):
        """download ip config address is not exist and then return the value

        Returns:
            str: redis ip config address

        """
        if not self._ip_file:
            self._download()
        return self._ip_file

    def persist(self, s3_kms_extra_args):
        """persist ip done flag and ip config addres into s3 bucket

        Args:
            s3_kms_extra_args (dict): s3 key management service extra argument

        """
        # persist ip address first
        ip_address = {"IP": self.get_ip_from_host()}
        ip_address_json = json.dumps(ip_address)
        # persist ip address
        # if retry failed, s3_client upload_fileobj will log and exit 500
        self._s3_client.upload_fileobj(
            bucket=self._bucket,
            s3_key=self._s3_ip_address_key,
            fileobj=io.BytesIO(ip_address_json.encode()),
            s3_kms_extra_args=s3_kms_extra_args,
        )
        LOG.info(
            "[s3] Successfully uploaded ip address to \
                 s3 bucket {} with s3 key {}.".format(
                self._bucket, self._s3_ip_address_key
            )
        )
        # persist done second
        # if retry failed, s3_client upload_fileobj will log and exit 500
        self._s3_client.upload_fileobj(
            bucket=self._bucket,
            s3_key=self._s3_ip_done_key,
            fileobj=io.BytesIO(b"done"),
            s3_kms_extra_args=s3_kms_extra_args,
        )
        LOG.info(
            "[s3] Successfully uploaded ip done to \
                 s3 bucket {} with s3 key {}.".format(
                self._bucket, self._s3_ip_done_key
            )
        )

    def _download(self):
        """wait for ip config to be ready first and then download it"""

        # check and make local directory
        local_dir = os.path.dirname(self._local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Download the ip file with retry
        try:
            # Wait for sagemaker to produce the redis ip
            self._wait_for_ip_config()
            self._s3_client.download_file(
                bucket=self._bucket, s3_key=self._s3_ip_address_key, local_path=self._local_path
            )
            LOG.info(
                "[s3] Successfully downloaded ip config from \
                 s3 key {} to local {}.".format(
                    self._s3_ip_address_key, self._local_path
                )
            )
            with open(self._local_path) as file:
                self._ip_file = json.load(file)["IP"]
        except botocore.exceptions.ClientError as err:
            log_and_exit(
                "Failed to download ip file: s3_bucket: {}, s3_key: {}, {}".format(
                    self._bucket, self._s3_ip_address_key, err
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def _wait_for_ip_config(self):
        """wait for ip config to be ready"""
        time_elapsed = 0
        while time_elapsed < SAGEMAKER_WAIT_TIME:
            # if retry failed, s3_client list_objects_v2 will log and exit 500
            response = self._s3_client.list_objects_v2(
                bucket=self._bucket, prefix=self._s3_ip_done_key
            )
            if "Contents" in response:
                break
            time.sleep(1)
            time_elapsed += 1
            if time_elapsed % 5 == 0:
                LOG.info(
                    "Waiting for SageMaker Redis server IP: Time elapsed: %s seconds", time_elapsed
                )
        if time_elapsed >= SAGEMAKER_WAIT_TIME:
            log_and_exit(
                "Timed out while attempting to retrieve the Redis IP",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    @staticmethod
    def get_ip_from_host(timeout=100):
        """get ip address for host

        Args:
            timeout (int): timeout in second for get ip from host

        Returns:
            str: ip address
        """
        counter = 0
        ip_address = None
        host_name = socket.gethostname()
        LOG.info("Hostname: %s" % host_name)
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
