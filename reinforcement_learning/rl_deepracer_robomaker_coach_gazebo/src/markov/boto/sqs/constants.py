"""This module houses the constants for the sqs client"""
from enum import Enum


class StatusIndicator(Enum):
    """Enum containing the integers signifing the
    the return status code for error accumulation.
    """

    SUCCESS = 0
    CLIENT_ERROR = 1
    SYSTEM_ERROR = 2
