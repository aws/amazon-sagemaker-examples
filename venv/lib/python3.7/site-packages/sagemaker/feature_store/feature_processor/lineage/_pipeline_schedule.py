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
"""Contains class to store the Pipeline Schedule"""
from __future__ import absolute_import
import attr


@attr.s
class PipelineSchedule:
    """A Schedule definition for FeatureProcessor Lineage.

    Attributes:
        schedule_name (str): Schedule Name.
        schedule_arn (str): The ARN of the Schedule.
        schedule_expression (str): The expression that defines when the schedule runs. It supports
            at expression, rate expression and cron expression. See https://docs.aws.amazon.com/
            scheduler/latest/APIReference/API_CreateSchedule.html#scheduler-CreateSchedule-request
            -ScheduleExpression for more details.
        pipeline_name (str): The SageMaker Pipeline name that will be scheduled.
        state (str): Specifies whether the schedule is enabled or disabled. Valid values are
            ENABLED and DISABLED. See https://docs.aws.amazon.com/scheduler/latest/APIReference/
            API_CreateSchedule.html#scheduler-CreateSchedule-request-State for more details.
            If not specified, it will default to DISABLED.
        start_date (Optional[datetime]): The date, in UTC, after which the schedule can begin
            invoking its target. Depending on the schedule’s recurrence expression, invocations
            might occur on, or after, the StartDate you specify.
    """

    schedule_name: str = attr.ib()
    schedule_arn: str = attr.ib()
    schedule_expression: str = attr.ib()
    pipeline_name: str = attr.ib()
    state: str = attr.ib()
    start_date: str = attr.ib()
