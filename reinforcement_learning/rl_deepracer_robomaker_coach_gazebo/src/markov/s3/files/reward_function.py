'''This module implements s3 client for reward function'''

import os
import logging
import future_fstrings
import botocore

from markov.s3.s3_client import S3Client
from markov.log_handler.logger import Logger
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.constants import (SIMAPP_EVENT_ERROR_CODE_500, SIMAPP_EVENT_ERROR_CODE_400,
                                          SIMAPP_S3_DATA_STORE_EXCEPTION,
                                          SIMAPP_SIMULATION_WORKER_EXCEPTION)

LOG = Logger(__name__, logging.INFO).get_logger()

class RewardFunction():
    '''reward function upload, download, and parse
    '''
    def __init__(self, bucket, s3_key, region_name="us-east-1",
                 s3_endpoint_url=None,
                 local_path="./custom_files/agent/customer_reward_function.py",
                 max_retry_attempts=5, backoff_time_sec=1.0):
        '''reward function upload, download, and parse

        Args:
            bucket (str): S3 bucket string
            s3_key (str): S3 key string
            region_name (str): S3 region name
            local_path (str): file local path
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry
        '''

        # check s3 key and bucket exist for reward function
        if not s3_key or not bucket:
            log_and_exit("Reward function code S3 key or bucket not available for S3. \
                         bucket: {}, key: {}".format(bucket, s3_key),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
        self._bucket = bucket
        # Strip the s3://<bucket> from uri, if s3_key past in as uri
        self._s3_key = s3_key.replace('s3://{}/'.format(self._bucket), '')
        self._local_path_processed = local_path
        # if _local_path_processed is test.py then _local_path_preprocessed is test_preprocessed.py
        self._local_path_preprocessed = ("_preprocessed.py").join(local_path.split(".py"))
        # if local _local_path_processed is ./custom_files/agent/customer_reward_function.py,
        # then the import path should be custom_files.agent.customer_reward_function by
        # remove ".py", remove "./", and replace "/" and "."
        self._import_path = local_path.replace(".py", "").replace("./", "").replace("/", ".")
        self._reward_function = None
        self._s3_client = S3Client(region_name,
                                   s3_endpoint_url,
                                   max_retry_attempts,
                                   backoff_time_sec)

    def get_reward_function(self):
        '''Download reward function, import it's module into code, and return it module

        Returns:
            module: reward function module

        '''

        if not self._reward_function:
            self._download()
            try:
                reward_function_module = __import__(self._import_path, fromlist=[None])
                self._reward_function = reward_function_module.reward_function
                LOG.info("Succeed to import user's reward function")
            except Exception as e:
                log_and_exit("Failed to import user's reward_function: {}".format(e),
                             SIMAPP_SIMULATION_WORKER_EXCEPTION,
                             SIMAPP_EVENT_ERROR_CODE_400)
        return self._reward_function

    def _download(self):
        '''Download customer reward function from s3 with retry logic'''

        # check and make local directory
        local_dir = os.path.dirname(self._local_path_preprocessed)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # download customer reward function with retry
        try:
            self._s3_client.download_file(bucket=self._bucket,
                                          s3_key=self._s3_key,
                                          local_path=self._local_path_preprocessed)
        except botocore.exceptions.ClientError as err:
            log_and_exit("Failed to download reward function: s3_bucket: {}, s3_key: {}, {}"\
                .format(self._bucket, self._s3_key, err),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

        LOG.info("[s3] Successfully downloaded reward function from s3 key {} to local \
                 {}.".format(self._s3_key, self._local_path_preprocessed))

        self._fstring_decoded_reward_function()

    def _fstring_decoded_reward_function(self):
        """ python 3.6 supports fstring and console lambda function validates using python3.6.
        But all the simapp code is runs in python3.5 which does not support fstring. This funciton
        support fstring in python 3.5"""

        try:
            with open(self._local_path_preprocessed, 'rb') as filepointer:
                text, _ = future_fstrings.fstring_decode(filepointer.read())
            with open(self._local_path_processed, 'wb') as filepointer:
                filepointer.write(text.encode('UTF-8'))
        except Exception as e:
            log_and_exit("Failed to decode the fstring format in reward function: {}".format(e),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
