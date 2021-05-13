"""This module implement deepracer boto client"""

import abc
import logging
import random
import time

import boto3
import botocore
from markov.boto.constants import BOTO_ERROR_MSG_FORMAT
from markov.constants import CONNECT_TIMEOUT, NUM_RETRIES
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class DeepRacerBotoClient(object):
    """Deepracer boto client class"""

    def __init__(
        self,
        region_name="us-east-1",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
        boto_client_name=None,
        session=None,
    ):
        """Deepracer boto client class

        Args:
          region_name (str): aws region name
          max_retry_attempts (int): max retry attempts for client call
          backoff_time_sec (float): exp back off time between call
          boto_client_name (str): boto client name
          session (boto3.Session): An alternative session to use.
                                   Defaults to None.
        """
        self._region_name = region_name
        self._max_retry_attempts = max_retry_attempts
        self._backoff_time_sec = backoff_time_sec
        self._boto_client_name = boto_client_name
        self._session = session

    def _get_boto_config(self):
        """Returns a botocore config object which specifies the number of times to retry"""

        return botocore.config.Config(
            retries=dict(max_attempts=NUM_RETRIES), connect_timeout=CONNECT_TIMEOUT
        )

    def get_client(self):
        """Return boto client"""
        if self._session:
            # auto refresh session
            s3_client = self._session.client(
                self._boto_client_name,
                region_name=self._region_name,
                config=self._get_boto_config(),
            )
        else:
            # new session per get client call
            s3_client = boto3.Session().client(
                self._boto_client_name,
                region_name=self._region_name,
                config=self._get_boto_config(),
            )
        return s3_client

    def exp_backoff(self, action_method, **kwargs):
        """retry on action_method

        Args:
            action_method (method) : specific action method
            **kwargs: argument for action_method
        """

        # download with retry
        try_count = 0
        while True:
            try:
                return action_method(**kwargs)
            except Exception as e:
                try_count += 1
                if try_count > self._max_retry_attempts:
                    raise e
                # use exponential backoff
                backoff_time = (pow(try_count, 2) + random.random()) * self._backoff_time_sec
                error_message = BOTO_ERROR_MSG_FORMAT.format(
                    self._boto_client_name,
                    backoff_time,
                    str(try_count),
                    str(self._max_retry_attempts),
                    e,
                )
                LOG.info(error_message)
                time.sleep(backoff_time)
