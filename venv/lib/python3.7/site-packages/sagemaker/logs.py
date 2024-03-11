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
"""Placeholder docstring"""
from __future__ import absolute_import

import collections
import functools
import os
import sys

##############################################################################
#
# Support for reading logs
#
##############################################################################


class ColorWrap(object):
    """A callable that will print text in a different color depending on the instance.

    Up to 6 if standard output is a terminal or a Jupyter notebook cell.
    """

    # For what color each number represents, see
    # https://misc.flogisoft.com/bash/tip_colors_and_formatting#colors
    _stream_colors = [34, 35, 32, 36, 33]

    def __init__(self, force=False):
        """Initialize the class.

        Args:
            force (bool): If True, render colorizes output no matter where the
                output is (default: False).
        """
        self.colorize = force or sys.stdout.isatty() or os.environ.get("JPY_PARENT_PID", None)

    def __call__(self, index, s):
        """Print the output, colorized or not, depending on the environment.

        Args:
            index (int): The instance number.
            s (str): The string to print.
        """
        if self.colorize:
            self._color_wrap(index, s)
        else:
            print(s)

    def _color_wrap(self, index, s):
        """Placeholder docstring"""
        print("\x1b[{}m{}\x1b[0m".format(self._stream_colors[index % len(self._stream_colors)], s))


def argmin(arr, f):
    """Return the index, i, in arr that minimizes f(arr[i])

    Args:
        arr:
        f:
    """
    m = None
    i = None
    for idx, item in enumerate(arr):
        if item is not None:
            if m is None or f(item) < m:
                m = f(item)
                i = idx
    return i


def some(arr):
    """Return True iff there is an element, a, of arr such that a is not None.

    Args:
        arr:
    """
    return functools.reduce(lambda x, y: x or (y is not None), arr, False)


# Position is a tuple that includes the last read timestamp and the number of items that were read
# at that time. This is used to figure out which event to start with on the next read.
Position = collections.namedtuple("Position", ["timestamp", "skip"])


def multi_stream_iter(client, log_group, streams, positions=None):
    """Iterate over the available events coming from a set of log streams.

    Log streams are in a single log group interleaving the events from each stream
    so they're yielded in timestamp order.

    Args:
        client (boto3 client): The boto client for logs.
        log_group (str): The name of the log group.
        streams (list of str): A list of the log stream names. The position of the stream in
        this list is the stream number.
        positions: (list of Positions): A list of pairs of (timestamp, skip) which represents
        the last record read from each stream.

    Yields:
        A tuple of (stream number, cloudwatch log event).
    """
    positions = positions or {s: Position(timestamp=0, skip=0) for s in streams}
    event_iters = [
        log_stream(client, log_group, s, positions[s].timestamp, positions[s].skip) for s in streams
    ]
    events = []
    for s in event_iters:
        if not s:
            events.append(None)
            continue
        try:
            events.append(next(s))
        except StopIteration:
            events.append(None)

    while some(events):
        i = argmin(events, lambda x: x["timestamp"] if x else 9999999999)
        yield (i, events[i])
        try:
            events[i] = next(event_iters[i])
        except StopIteration:
            events[i] = None


def log_stream(client, log_group, stream_name, start_time=0, skip=0):
    """A generator for log items in a single stream.

    This will yield all the items that are available at the current moment.

    Args:
        client (boto3.CloudWatchLogs.Client): The Boto client for CloudWatch logs.
        log_group (str): The name of the log group.
        stream_name (str): The name of the specific stream.
        start_time (int): The time stamp value to start reading the logs from (default: 0).
        skip (int): The number of log entries to skip at the start (default: 0). This is for
        when there are multiple entries at the same timestamp.

    Yields:
       dict: A CloudWatch log event with the following key-value pairs:
           'timestamp' (int): The time of the event.
           'message' (str): The log event data.
           'ingestionTime' (int): The time the event was ingested.
    """

    next_token = None

    event_count = 1
    while event_count > 0:
        if next_token is not None:
            token_arg = {"nextToken": next_token}
        else:
            token_arg = {}

        response = client.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            startTime=start_time,
            startFromHead=True,
            **token_arg
        )
        next_token = response["nextForwardToken"]
        events = response["events"]
        event_count = len(events)
        if event_count > skip:
            events = events[skip:]
            skip = 0
        else:
            skip = skip - event_count
            events = []
        for ev in events:
            yield ev
