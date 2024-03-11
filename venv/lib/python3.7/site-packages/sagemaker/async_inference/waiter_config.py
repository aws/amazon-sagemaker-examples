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
"""A class for WaiterConfig used in async inference

Use it when using async inference and wait for the result.
"""

from __future__ import absolute_import


class WaiterConfig(object):
    """Configuration object passed in when using async inference and wait for the result."""

    def __init__(
        self,
        max_attempts=60,
        delay=15,
    ):
        """Initialize a WaiterConfig object that provides parameters to control waiting behavior.

        Args:
            max_attempts (int): The maximum number of attempts to be made. If the max attempts is
            exceeded, Amazon SageMaker will raise ``PollingTimeoutError``. (Default: 60)
            delay (int): The amount of time in seconds to wait between attempts. (Default: 15)
        """

        self.max_attempts = max_attempts
        self.delay = delay

    def _to_request_dict(self):
        """Generates a dictionary using the parameters provided to the class."""
        waiter_dict = {
            "Delay": self.delay,
            "MaxAttempts": self.max_attempts,
        }

        return waiter_dict
