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
"""Provides utilities for instantiating dependencies to boto-python objects."""
from __future__ import absolute_import

import random
import time
from sagemaker.session import Session


def suffix():
    """Generate a random string of length 4."""
    alph = "abcdefghijklmnopqrstuvwxyz"
    return "-".join([time.strftime("%Y-%m-%d-%H%M%S"), "".join(random.sample(alph, 4))])


def name(prefix):
    """Generate a new name with the specified prefix."""
    return "-".join([prefix, suffix()])


def default_session():
    """Create a default session."""
    return Session()
