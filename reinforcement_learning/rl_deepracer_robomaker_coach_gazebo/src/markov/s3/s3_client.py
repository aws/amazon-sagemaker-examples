'''This module implement s3 client'''

import os
import json
import time
import logging
import botocore
import boto3
import random

from markov.log_handler.logger import Logger
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.constants import (SIMAPP_EVENT_ERROR_CODE_500, SIMAPP_EVENT_ERROR_CODE_400,
                                          SIMAPP_S3_DATA_STORE_EXCEPTION)
from markov.constants import (NUM_RETRIES, CONNECT_TIMEOUT, S3_KMS_CMK_ARN_ENV,
                              HYPERPARAMETERS, SAGEMAKER_S3_KMS_CMK_ARN,
                              ROBOMAKER_S3_KMS_CMK_ARN, S3KmsEncryption)

S3_ERROR_MSG_FORMAT = "S3 failed, retry after {0} seconds. Re-try count: {1}/{2}: {3}"

logger = Logger(__name__, logging.INFO).get_logger()


class S3Client():
    def __init__(self, region_name="us-east-1", s3_endpoint_url=None, max_retry_attempts=5, backoff_time_sec=1.0):
        '''S3 client

        Args:
            region_name (str): aws region name
            max_retry_attempts (int): maximum number of retry
            backoff_time_sec (float): backoff second between each retry
        '''

        self._region_name = region_name
        self._s3_endpoint_url = s3_endpoint_url
        self._max_retry_attempts = max_retry_attempts
        self._backoff_time_sec = backoff_time_sec

    def _get_boto_config(self):
        '''Returns a botocore config object which specifies the number of times to retry'''

        return botocore.config.Config(retries=dict(max_attempts=NUM_RETRIES),
                                      connect_timeout=CONNECT_TIMEOUT)

    def _get_s3_client(self):
        '''Return boto s3 client'''

        s3_client = boto3.Session().client('s3',
                                           region_name=self._region_name,
                                           endpoint_url=self._s3_endpoint_url,
                                           config=self._get_boto_config())
        return s3_client

    def _exp_backoff(self, action_method, **kwargs):
        ''' retry on action_method

        Args:
            action_method (method) : specific action method
            **kwargs: argument for action_method

        Raises:
            e: [description]
        '''

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
                error_message = S3_ERROR_MSG_FORMAT.format(backoff_time,
                                                           str(try_count),
                                                           str(self._max_retry_attempts),
                                                           e)
                logger.info(error_message)
                time.sleep(backoff_time)

    def download_file(self, bucket, s3_key, local_path):
        '''download file from s3 with retry logic

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            local_path (str): file local path

        '''

        try:
            self._exp_backoff(action_method=self._get_s3_client().download_file,
                              Bucket=bucket,
                              Key=s3_key,
                              Filename=local_path)
        except botocore.exceptions.ClientError as err:
            # It is possible that the file isn't there in which case we should
            # raise exception and let the client decide the next action
            raise err
        except botocore.exceptions.ConnectTimeoutError as ex:
            log_and_exit("Issue with your current VPC stack and IAM roles.\
                          You might need to reset your account resources: {}".format(ex),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception as ex:
            log_and_exit("Exception in downloading file (s3bucket: {} s3_key: {}): {}".format(bucket,
                                                                                              s3_key,
                                                                                              ex),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def upload_file(self, bucket, s3_key, local_path, s3_kms_extra_args):
        '''upload file to s3 with retry logic

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            local_path (str): file local path
            s3_kms_extra_args (dict): s3 key management service extra argument

        '''

        try:
            self._exp_backoff(action_method=self._get_s3_client().upload_file,
                              Filename=local_path,
                              Bucket=bucket,
                              Key=s3_key,
                              ExtraArgs=s3_kms_extra_args)
        except botocore.exceptions.ClientError:
            log_and_exit("Unable to upload file",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
        except Exception as ex:
            log_and_exit("Exception in uploading file: {}".format(ex),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def upload_fileobj(self, bucket, s3_key, fileobj, s3_kms_extra_args):
        '''upload fileobj to s3 with retry logic

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            fileobj (Fileobj): file object
            s3_kms_extra_args (dict): s3 key management service extra argument

        '''

        try:
            self._exp_backoff(action_method=self._get_s3_client().upload_fileobj,
                              Fileobj=fileobj,
                              Bucket=bucket,
                              Key=s3_key,
                              ExtraArgs=s3_kms_extra_args)
        except botocore.exceptions.ClientError:
            log_and_exit("Unable to upload fileobj",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
        except Exception as ex:
            log_and_exit("Exception in uploading fileobj: {}".format(ex),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def list_objects_v2(self, bucket, prefix):
        '''list object v2 from s3 with retry logic

        Args:
          bucket (str): s3 bucket
          prefix (str): s3 prefix

        '''

        try:
            return self._exp_backoff(action_method=self._get_s3_client().list_objects_v2,
                                     Bucket=bucket,
                                     Prefix=prefix)
        except botocore.exceptions.ClientError:
            log_and_exit("Unable to list objects",
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception as ex:
            log_and_exit("Exception in listing objects: {}".format(ex),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def put_object(self, bucket, s3_key, body, s3_kms_extra_args):
        '''put object into s3 with retry logic

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            body (b'bytes'): text body in byte
            s3_kms_extra_args (dict): s3 key management service extra argument

        '''
        try:
            self._exp_backoff(action_method=self._get_s3_client().put_object,
                              Bucket=bucket,
                              Key=s3_key,
                              Body=body,
                              **s3_kms_extra_args)
        except botocore.exceptions.ClientError as err:
            log_and_exit("Unable to put object to s3: bucket: {}, error: {}"
                         .format(bucket, err.response['Error']['Code']),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
        except Exception as ex:
            log_and_exit("Exception in putting objects: {}".format(ex),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def delete_object(self, bucket, s3_key):
        '''delete files specified by s3_key from s3 bucket

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
        '''
        try:
            self._exp_backoff(action_method=self._get_s3_client().delete_object,
                              Bucket=bucket,
                              Key=s3_key)
        except botocore.exceptions.ClientError as err:
            log_and_exit("Unable to delete object from s3: bucket: {}, error: {}"
                         .format(bucket, err.response['Error']['Code']),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception as ex:
            log_and_exit("Unable to delete object from s3, exception: {}".format(ex),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def paginate(self, bucket, prefix):
        '''get paginator for list_objects_v2

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key

        Returns:
            iter: page iterator
        '''
        try:
            # get a paginator for list_objects_v2
            kwargs = {'Bucket': bucket,
                      'Prefix': prefix}
            paginator = self._get_s3_client().get_paginator("list_objects_v2")
            # paginate based on the kwargs
            return paginator.paginate(**kwargs)
        except botocore.exceptions.ClientError as err:
            log_and_exit("Unable to paginate from s3, error: {}"
                         .format(err.response['Error']['Code']),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_400)
        except Exception as ex:
            log_and_exit("Unable to paginate from s3, exception: {}".format(ex),
                         SIMAPP_S3_DATA_STORE_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
