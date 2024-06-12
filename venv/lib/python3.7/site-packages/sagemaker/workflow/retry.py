# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Pipeline parameters and conditions for workflow."""
from __future__ import absolute_import

from enum import Enum
from typing import List
import attr

from sagemaker.workflow.entities import Entity, DefaultEnumMeta, RequestType


DEFAULT_BACKOFF_RATE = 2.0
DEFAULT_INTERVAL_SECONDS = 1
MAX_ATTEMPTS_CAP = 20
MAX_EXPIRE_AFTER_MIN = 14400


class StepExceptionTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """Step ExceptionType enum."""

    SERVICE_FAULT = "Step.SERVICE_FAULT"
    THROTTLING = "Step.THROTTLING"


class SageMakerJobExceptionTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """SageMaker Job ExceptionType enum."""

    INTERNAL_ERROR = "SageMaker.JOB_INTERNAL_ERROR"
    CAPACITY_ERROR = "SageMaker.CAPACITY_ERROR"
    RESOURCE_LIMIT = "SageMaker.RESOURCE_LIMIT"


@attr.s
class RetryPolicy(Entity):
    """RetryPolicy base class

    Attributes:
        backoff_rate (float): The multiplier by which the retry interval increases
            during each attempt (default: 2.0)
        interval_seconds (int): An integer that represents the number of seconds before the
            first retry attempt (default: 1)
        max_attempts (int): A positive integer that represents the maximum
            number of retry attempts. (default: None)
        expire_after_mins (int): A positive integer that represents the maximum minute
            to expire any further retry attempt (default: None)
    """

    backoff_rate: float = attr.ib(default=DEFAULT_BACKOFF_RATE)
    interval_seconds: int = attr.ib(default=DEFAULT_INTERVAL_SECONDS)
    max_attempts: int = attr.ib(default=None)
    expire_after_mins: int = attr.ib(default=None)

    @backoff_rate.validator
    def validate_backoff_rate(self, _, value):
        """Validate the input back off rate type"""
        if value:
            assert value >= 0.0, "backoff_rate should be non-negative"

    @interval_seconds.validator
    def validate_interval_seconds(self, _, value):
        """Validate the input interval seconds"""
        if value:
            assert value >= 0.0, "interval_seconds rate should be non-negative"

    @max_attempts.validator
    def validate_max_attempts(self, _, value):
        """Validate the input max attempts"""
        if value:
            assert (
                MAX_ATTEMPTS_CAP >= value >= 1
            ), f"max_attempts must in range of (0, {MAX_ATTEMPTS_CAP}] attempts"

    @expire_after_mins.validator
    def validate_expire_after_mins(self, _, value):
        """Validate expire after mins"""
        if value:
            assert (
                MAX_EXPIRE_AFTER_MIN >= value >= 0
            ), f"expire_after_mins must in range of (0, {MAX_EXPIRE_AFTER_MIN}] minutes"

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        if (self.max_attempts is None) == self.expire_after_mins is None:
            raise ValueError("Only one of [max_attempts] and [expire_after_mins] can be given.")

        request = {
            "BackoffRate": self.backoff_rate,
            "IntervalSeconds": self.interval_seconds,
        }

        if self.max_attempts:
            request["MaxAttempts"] = self.max_attempts

        if self.expire_after_mins:
            request["ExpireAfterMin"] = self.expire_after_mins

        return request


class StepRetryPolicy(RetryPolicy):
    """RetryPolicy for a retryable step. The pipeline service will retry

        `sagemaker.workflow.retry.StepRetryExceptionTypeEnum.SERVICE_FAULT` and
        `sagemaker.workflow.retry.StepRetryExceptionTypeEnum.THROTTLING` regardless of
        pipeline step type by default. However, for step defined as retryable, you can override them
        by specifying a StepRetryPolicy.

    Attributes:
        exception_types (List[StepExceptionTypeEnum]): the exception types to match for this policy
        backoff_rate (float): The multiplier by which the retry interval increases
            during each attempt (default: 2.0)
        interval_seconds (int): An integer that represents the number of seconds before the
            first retry attempt (default: 1)
        max_attempts (int): A positive integer that represents the maximum
            number of retry attempts. (default: None)
        expire_after_mins (int): A positive integer that represents the maximum minute
            to expire any further retry attempt (default: None)
    """

    def __init__(
        self,
        exception_types: List[StepExceptionTypeEnum],
        backoff_rate: float = 2.0,
        interval_seconds: int = 1,
        max_attempts: int = None,
        expire_after_mins: int = None,
    ):
        super().__init__(backoff_rate, interval_seconds, max_attempts, expire_after_mins)
        for exception_type in exception_types:
            if not isinstance(exception_type, StepExceptionTypeEnum):
                raise ValueError(f"{exception_type} is not of StepExceptionTypeEnum.")
        self.exception_types = exception_types

    def to_request(self) -> RequestType:
        """Gets the request structure for retry policy."""
        request = super().to_request()
        request["ExceptionType"] = [e.value for e in self.exception_types]
        return request

    def __hash__(self):
        """Hash function for StepRetryPolicy types"""
        return hash(tuple(self.to_request()))


class SageMakerJobStepRetryPolicy(RetryPolicy):
    """RetryPolicy for exception thrown by SageMaker Job.

    Attributes:
        exception_types (List[SageMakerJobExceptionTypeEnum]):
            The SageMaker exception to match for this policy. The SageMaker exceptions
            captured here are the exceptions thrown by synchronously
            creating the job. For instance the resource limit exception.
        failure_reason_types (List[SageMakerJobExceptionTypeEnum]): the SageMaker
            failure reason types to match for this policy. The failure reason type
            is presented in FailureReason field of the Describe response, it indicates
            the runtime failure reason for a job.
        backoff_rate (float): The multiplier by which the retry interval increases
            during each attempt (default: 2.0)
        interval_seconds (int): An integer that represents the number of seconds before the
            first retry attempt (default: 1)
        max_attempts (int): A positive integer that represents the maximum
            number of retry attempts. (default: None)
        expire_after_mins (int): A positive integer that represents the maximum minute
            to expire any further retry attempt (default: None)
    """

    def __init__(
        self,
        exception_types: List[SageMakerJobExceptionTypeEnum] = None,
        failure_reason_types: List[SageMakerJobExceptionTypeEnum] = None,
        backoff_rate: float = 2.0,
        interval_seconds: int = 1,
        max_attempts: int = None,
        expire_after_mins: int = None,
    ):
        super().__init__(backoff_rate, interval_seconds, max_attempts, expire_after_mins)

        if not exception_types and not failure_reason_types:
            raise ValueError(
                "At least one of the [exception_types, failure_reason_types] needs to be given."
            )

        self.exception_type_list: List[SageMakerJobExceptionTypeEnum] = []
        if exception_types:
            self.exception_type_list += exception_types
        if failure_reason_types:
            self.exception_type_list += failure_reason_types

        for exception_type in self.exception_type_list:
            if not isinstance(exception_type, SageMakerJobExceptionTypeEnum):
                raise ValueError(f"{exception_type} is not of SageMakerJobExceptionTypeEnum.")

    def to_request(self) -> RequestType:
        """Gets the request structure for retry policy."""
        request = super().to_request()
        request["ExceptionType"] = [e.value for e in self.exception_type_list]
        return request

    def __hash__(self):
        """Hash function for SageMakerJobStepRetryPolicy types"""
        return hash(tuple(self.to_request()))
