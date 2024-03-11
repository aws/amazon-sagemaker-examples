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
"""This module contains code related to the MonitoringAlerts."""
from __future__ import print_function, absolute_import

import attr


@attr.define
class ModelDashboardIndicatorAction(object):
    """An abstraction represents the monitoring dashboard indicator action summary.

    Attributes:
       enabled (bool): A boolean flag indicating if the model dashboard indicator is enabled or not.
    """

    enabled: bool = attr.ib()


@attr.define
class MonitoringAlertActions(object):
    """An abstraction represents the monitoring action.

    Attributes:
        model_dashboard_indicator (ModelDashboardIndicatorAction):
    """

    model_dashboard_indicator: ModelDashboardIndicatorAction = attr.ib()


@attr.define
class MonitoringAlertSummary(object):
    """An abstraction represents the monitoring alert summary.

    Attributes:
        alert_name (str): Monitoring alert name.
        creation_time (str): Creation time of the alert.
        last_modified_time (str): Last modified time of the alert.
        alert_status (str): Alert status, either InAlert or Ok.
        data_points_to_alert (int): Data points to evaluate to determine the alert status.
        evaluation_period (int): Period to evaluate the alert status.
        actions (MonitoringAlertActions): A list of actions to take when monitoring is InAlert
    """

    alert_name: str = attr.ib()
    creation_time: str = attr.ib()
    last_modified_time: str = attr.ib()
    alert_status: str = attr.ib()
    data_points_to_alert: int = attr.ib()
    evaluation_period: int = attr.ib()
    actions: MonitoringAlertActions = attr.ib()


@attr.define
class MonitoringAlertHistorySummary(object):
    """An abstraction represents the monitoring alert history summary.

    Attributes:
        alert_name (str): Monitoring alert name.
        creation_time (str): Creation time of the alert.
        alert_status (str): Alert status, either InAlert or Ok.
    """

    alert_name: str = attr.ib()
    creation_time: str = attr.ib()
    alert_status: str = attr.ib()
