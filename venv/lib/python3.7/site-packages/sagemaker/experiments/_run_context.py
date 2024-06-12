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
"""Contains the SageMaker Experiment _RunContext class."""
from __future__ import absolute_import

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sagemaker.experiments import Run


class _RunContext:
    """A static context variable to keep track of the current Run object"""

    _context_run = None

    @classmethod
    def add_run_object(cls, run: "Run"):
        """Keep track of the current executing Run object

        by adding it to a class static variable.

        Args:
            run (Run): The current Run object to be tracked.
        """
        cls._context_run = run

    @classmethod
    def drop_current_run(cls) -> "Run":
        """Drop the Run object tracked in the global static variable

        as its execution finishes (its "with" block ends).

        Return:
            Run: the dropped Run object.
        """
        current_run = cls._context_run
        cls._context_run = None
        return current_run

    @classmethod
    def get_current_run(cls) -> "Run":
        """Return the current Run object without dropping it.

        Return:
            Run: the current Run object to be returned.
        """
        return cls._context_run
