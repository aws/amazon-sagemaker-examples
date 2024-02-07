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
"""Contains class that parse the input data start and end offset"""
from __future__ import absolute_import

import re
from typing import Optional, Tuple, Union
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from sagemaker.feature_store.feature_processor._constants import (
    EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT,
)

UNIT_RE = r"(\d+?)\s+([a-z]+?)s?"
VALID_UNITS = ["hour", "day", "week", "month", "year"]


class InputOffsetParser:
    """Contains methods to parse the input offset to different formats.

    Args:
        now (datetime):
            The point of time that the parser should calculate offset against.
    """

    def __init__(self, now: Union[datetime, str] = None) -> None:
        if now is None:
            self.now = datetime.now(timezone.utc)
        elif isinstance(now, datetime):
            self.now = now
        else:
            self.now = datetime.strptime(now, EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT)

    def get_iso_format_offset_date(self, offset: Optional[str]) -> str:
        """Get the iso format of target date based on offset diff.

        Args:
            offset (Optional[str]): Offset that is used for target date calcluation.

        Returns:
            str: ISO-8061 formatted string of the offset date.
        """
        if offset is None:
            return None

        offset_datetime = self.get_offset_datetime(offset)
        return offset_datetime.strftime(EXECUTION_TIME_PIPELINE_PARAMETER_FORMAT)

    def get_offset_datetime(self, offset: Optional[str]) -> datetime:
        """Get the datetime format of target date based on offset diff.

        Args:
            offset (Optional[str]): Offset that is used for target date calcluation.

        Returns:
            datetime: datetime instance of the offset date.
        """
        if offset is None:
            return None

        offset_td = InputOffsetParser.parse_offset_to_timedelta(offset)

        return self.now + offset_td

    def get_offset_date_year_month_day_hour(self, offset: Optional[str]) -> Tuple[str]:
        """Get the year, month, day and hour based on offset diff.

        Args:
            offset (Optional[str]): Offset that is used for target date calcluation.

        Returns:
            Tuple[str]: A tuple that consists of extracted year, month, day, hour from offset date.
        """
        if offset is None:
            return (None, None, None, None)

        offset_dt = self.get_offset_datetime(offset)
        return (
            offset_dt.strftime("%Y"),
            offset_dt.strftime("%m"),
            offset_dt.strftime("%d"),
            offset_dt.strftime("%H"),
        )

    @staticmethod
    def parse_offset_to_timedelta(offset: Optional[str]) -> relativedelta:
        """Parse the offset to time delta.

        Args:
            offset (Optional[str]): Offset that is used for target date calcluation.

        Raises:
            ValueError: If an offset is provided in a unrecognizable format.
            ValueError: If an invalid offset unit is provided.

        Returns:
            reletivedelta: Time delta representation of the time offset.
        """
        if offset is None:
            return None

        unit_match = re.fullmatch(UNIT_RE, offset)

        if not unit_match:
            raise ValueError(
                f"[{offset}] is not in a valid offset format. "
                "Please pass a valid offset e.g '1 day'."
            )

        multiple, unit = unit_match.groups()

        if unit not in VALID_UNITS:
            raise ValueError(f"[{unit}] is not a valid offset unit. Supported units: {VALID_UNITS}")

        shift_args = {f"{unit}s": -int(multiple)}

        return relativedelta(**shift_args)
