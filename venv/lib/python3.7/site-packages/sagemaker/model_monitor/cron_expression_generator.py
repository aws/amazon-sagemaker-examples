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
"""This module contains code related to the CronExpressionGenerator class.

Codes are used for generating cron expressions compatible with Amazon SageMaker Model
Monitoring Schedules.
"""
from __future__ import print_function, absolute_import


class CronExpressionGenerator(object):
    """Generates cron expression strings for the SageMaker Model Monitoring Schedule API."""

    @staticmethod
    def hourly():
        """Generates hourly cron expression that denotes that a job runs at the top of every hour.

        Returns:
            str: The cron expression format accepted by the Amazon SageMaker Model Monitoring
                Schedule API.

        """
        return "cron(0 * ? * * *)"

    @staticmethod
    def daily(hour=0):
        """Generates daily cron expression that denotes that a job runs at the top of every hour.

        Args:
            hour (int): The hour in HH24 format (UTC) to run the job at, on a daily schedule.
                Examples:
                    - 00
                    - 12
                    - 17
                    - 23

        Returns:
            str: The cron expression format accepted by the Amazon SageMaker Model Monitoring
                Schedule API.

        """
        return "cron(0 {} ? * * *)".format(hour)

    @staticmethod
    def daily_every_x_hours(hour_interval, starting_hour=0):
        """Generates "daily every x hours" cron expression.

        That denotes that a job runs every day at the specified hour, and then every x hours,
        as specified in hour_interval.

         Example:
             >>> daily_every_x_hours(hour_interval=2, starting_hour=0)
             This will run every 2 hours starting at midnight.

             >>> daily_every_x_hours(hour_interval=10, starting_hour=0)
             This will run at midnight, 10am, and 8pm every day.

        Args:
            hour_interval (int): The hour interval to run the job at.
            starting_hour (int): The hour at which to begin in HH24 format (UTC).

        Returns:
            str: The cron expression format accepted by the Amazon SageMaker Model Monitoring
                Schedule API.

        """
        return "cron(0 {}/{} ? * * *)".format(starting_hour, hour_interval)
