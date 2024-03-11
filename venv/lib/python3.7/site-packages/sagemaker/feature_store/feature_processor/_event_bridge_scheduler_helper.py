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
"""Contains classes for EventBridge Schedule management for a feature processor."""
from __future__ import absolute_import
import logging
from datetime import datetime
from typing import Dict, Optional, Any
import attr
from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.feature_store.feature_processor._constants import (
    EXECUTION_TIME_PIPELINE_PARAMETER,
    EVENT_BRIDGE_INVOCATION_TIME,
    NO_FLEXIBLE_TIME_WINDOW,
    RESOURCE_NOT_FOUND_EXCEPTION,
)

logger = logging.getLogger("sagemaker")


@attr.s
class EventBridgeSchedulerHelper:
    """Contains helper methods for scheduling events to EventBridge"""

    sagemaker_session: Session = attr.ib()
    event_bridge_scheduler_client = attr.ib()

    def upsert_schedule(
        self,
        schedule_name: str,
        pipeline_arn: str,
        schedule_expression: str,
        state: str,
        start_date: datetime,
        role: str,
    ) -> Dict:
        """Creates or updates a Schedule for the given pipeline_arn and schedule_expression.

        Args:
            schedule_name: The name of the schedule.
            pipeline_arn: The ARN of the sagemaker pipeline that needs to scheduled.
            schedule_expression: The schedule expression.
            state: Specifies whether the schedule is enabled or disabled. Can only
                be ENABLED or DISABLED.
            start_date: The date, in UTC, after which the schedule can begin invoking its target.
            role: The RoleArn used to execute the scheduled events.

        Returns:
            schedule_arn: The arn of the schedule.
        """
        pipeline_parameter = dict(
            PipelineParameterList=[
                dict(
                    Name=EXECUTION_TIME_PIPELINE_PARAMETER,
                    Value=EVENT_BRIDGE_INVOCATION_TIME,
                )
            ]
        )
        create_or_update_schedule_request_dict = dict(
            Name=schedule_name,
            ScheduleExpression=schedule_expression,
            FlexibleTimeWindow=NO_FLEXIBLE_TIME_WINDOW,
            Target=dict(
                Arn=pipeline_arn,
                SageMakerPipelineParameters=pipeline_parameter,
                RoleArn=role,
            ),
            State=state,
            StartDate=start_date,
        )
        try:
            return self.event_bridge_scheduler_client.update_schedule(
                **create_or_update_schedule_request_dict
            )
        except ClientError as e:
            if RESOURCE_NOT_FOUND_EXCEPTION == e.response["Error"]["Code"]:
                return self.event_bridge_scheduler_client.create_schedule(
                    **create_or_update_schedule_request_dict
                )
            raise e

    def delete_schedule(self, schedule_name: str) -> None:
        """Deletes an EventBridge Schedule of a given pipeline if there is one.

        Args:
            schedule_name: The name of the EventBridge Schedule.
        """
        logger.info("Deleting EventBridge Schedule for pipeline %s.", schedule_name)
        self.event_bridge_scheduler_client.delete_schedule(Name=schedule_name)

    def describe_schedule(self, schedule_name) -> Optional[Dict[str, Any]]:
        """Describe the EventBridge Schedule ARN corresponding to a sagemaker pipeline

        Args:
            schedule_name: The name of the EventBridge Schedule.
        Returns:
            Optional[Dict[str, str]] : Describe EventBridge Schedule response if exists.
        """
        try:
            event_bridge_scheduler_response = self.event_bridge_scheduler_client.get_schedule(
                Name=schedule_name
            )
            return event_bridge_scheduler_response
        except ClientError as e:
            if RESOURCE_NOT_FOUND_EXCEPTION == e.response["Error"]["Code"]:
                logger.info("No EventBridge Schedule found for pipeline %s.", schedule_name)
                return None
            raise e
