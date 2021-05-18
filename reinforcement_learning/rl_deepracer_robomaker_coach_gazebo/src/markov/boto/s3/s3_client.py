"""This module implement s3 client"""

import botocore
from markov.boto.constants import BotoClientNames
from markov.boto.deepracer_boto_client import DeepRacerBotoClient
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_EVENT_SYSTEM_ERROR,
    SIMAPP_EVENT_USER_ERROR,
    SIMAPP_S3_DATA_STORE_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericNonFatalException
from markov.log_handler.exception_handler import log_and_exit


class S3Client(DeepRacerBotoClient):
    """S3 Boto Client"""

    name = BotoClientNames.S3.value

    def __init__(
        self,
        region_name="us-east-1",
        max_retry_attempts=5,
        backoff_time_sec=1.0,
        session=None,
        log_and_cont=False,
    ):
        """S3 client

        Args:
            region_name (str): aws region name
            max_retry_attempts (int): maximum number of retry
            backoff_time_sec (float): backoff second between each retry
            session (boto3.Session): An alternative session to use.
                                     Defaults to None.
            log_and_cont (bool, optional): Log the error and continue with the flow.
                                           Defaults to False.
        """
        super(S3Client, self).__init__(
            region_name=region_name,
            max_retry_attempts=max_retry_attempts,
            backoff_time_sec=backoff_time_sec,
            boto_client_name=self.name,
            session=session,
        )
        self._log_and_cont = log_and_cont

    def download_file(self, bucket, s3_key, local_path):
        """download file from s3 with retry logic

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            local_path (str): file local path

        """

        try:
            self.exp_backoff(
                action_method=self.get_client().download_file,
                Bucket=bucket,
                Key=s3_key,
                Filename=local_path,
            )
        except botocore.exceptions.ClientError as err:
            # It is possible that the file isn't there in which case we should
            # raise exception and let the client decide the next action
            if self._log_and_cont:
                error_msg = "[s3] ClientError: Unable to download file from \
                            bucket {} with key {}. {}".format(
                    bucket, s3_key, ex
                )
                raise GenericNonFatalException(
                    error_msg=error_msg,
                    error_code=SIMAPP_EVENT_ERROR_CODE_400,
                    error_name=SIMAPP_EVENT_USER_ERROR,
                )
            raise err
        except botocore.exceptions.ConnectTimeoutError as ex:
            if self._log_and_cont:
                error_msg = "[s3] ConnectTimeoutError: Unable to download file from \
                            bucket {} with key {}. {}".format(
                    bucket, s3_key, ex
                )
                raise GenericNonFatalException(
                    error_msg=error_msg,
                    error_code=SIMAPP_EVENT_ERROR_CODE_400,
                    error_name=SIMAPP_EVENT_USER_ERROR,
                )
            log_and_exit(
                "Issue with your current VPC stack and IAM roles.\
                          You might need to reset your account resources: {}".format(
                    ex
                ),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            if self._log_and_cont:
                error_msg = "[s3] SystemError: Unable to download file from \
                            bucket {} with key {}. {}".format(
                    bucket, s3_key, ex
                )
                raise GenericNonFatalException(
                    error_msg=error_msg,
                    error_code=SIMAPP_EVENT_ERROR_CODE_500,
                    error_name=SIMAPP_EVENT_SYSTEM_ERROR,
                )
            log_and_exit(
                "Exception in downloading file (s3bucket: {} s3_key: {}): {}".format(
                    bucket, s3_key, ex
                ),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def upload_file(self, bucket, s3_key, local_path, s3_kms_extra_args):
        """upload file to s3 with retry logic

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            local_path (str): file local path
            s3_kms_extra_args (dict): s3 key management service extra argument

        """

        try:
            self.exp_backoff(
                action_method=self.get_client().upload_file,
                Filename=local_path,
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs=s3_kms_extra_args,
            )
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to upload file", SIMAPP_S3_DATA_STORE_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500
            )
        except Exception as ex:
            log_and_exit(
                "Exception in uploading file: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def upload_fileobj(self, bucket, s3_key, fileobj, s3_kms_extra_args):
        """upload fileobj to s3 with retry logic

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            fileobj (Fileobj): file object
            s3_kms_extra_args (dict): s3 key management service extra argument

        """

        try:
            self.exp_backoff(
                action_method=self.get_client().upload_fileobj,
                Fileobj=fileobj,
                Bucket=bucket,
                Key=s3_key,
                ExtraArgs=s3_kms_extra_args,
            )
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to upload fileobj",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in uploading fileobj: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def list_objects_v2(self, bucket, prefix):
        """list object v2 from s3 with retry logic

        Args:
          bucket (str): s3 bucket
          prefix (str): s3 prefix

        """

        try:
            return self.exp_backoff(
                action_method=self.get_client().list_objects_v2, Bucket=bucket, Prefix=prefix
            )
        except botocore.exceptions.ClientError:
            log_and_exit(
                "Unable to list objects",
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in listing objects: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def put_object(self, bucket, s3_key, body, s3_kms_extra_args):
        """put object into s3 with retry logic

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
            body (b'bytes'): text body in byte
            s3_kms_extra_args (dict): s3 key management service extra argument

        """
        try:
            self.exp_backoff(
                action_method=self.get_client().put_object,
                Bucket=bucket,
                Key=s3_key,
                Body=body,
                **s3_kms_extra_args
            )
        except botocore.exceptions.ClientError as err:
            log_and_exit(
                "Unable to put object to s3: bucket: {}, error: {}".format(
                    bucket, err.response["Error"]["Code"]
                ),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        except Exception as ex:
            log_and_exit(
                "Exception in putting objects: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def delete_object(self, bucket, s3_key):
        """delete files specified by s3_key from s3 bucket

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key
        """
        try:
            self.exp_backoff(
                action_method=self.get_client().delete_object, Bucket=bucket, Key=s3_key
            )
        except botocore.exceptions.ClientError as err:
            log_and_exit(
                "Unable to delete object from s3: bucket: {}, error: {}".format(
                    bucket, err.response["Error"]["Code"]
                ),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Unable to delete object from s3, exception: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def paginate(self, bucket, prefix):
        """get paginator for list_objects_v2

        Args:
            bucket (str): s3 bucket
            s3_key (str): s3 key

        Returns:
            iter: page iterator
        """
        try:
            # get a paginator for list_objects_v2
            kwargs = {"Bucket": bucket, "Prefix": prefix}
            paginator = self.get_client().get_paginator("list_objects_v2")
            # paginate based on the kwargs
            return paginator.paginate(**kwargs)
        except botocore.exceptions.ClientError as err:
            log_and_exit(
                "Unable to paginate from s3, error: {}".format(err.response["Error"]["Code"]),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        except Exception as ex:
            log_and_exit(
                "Unable to paginate from s3, exception: {}".format(ex),
                SIMAPP_S3_DATA_STORE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
