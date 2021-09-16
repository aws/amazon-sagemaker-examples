"""This module houses the constants for the boto client"""
from enum import Enum

BOTO_ERROR_MSG_FORMAT = "{0} failed, retry after {1} seconds. Re-try count: {2}/{3}: {4}"


class BotoClientNames(Enum):
    """Enum contains  all boto client for DeepRacer SimApp"""

    S3 = "s3"
    SQS = "sqs"
