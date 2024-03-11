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

import json

from sagemaker.workflow import is_pipeline_variable


class Hyperparameter(object):
    """An algorithm hyperparameter with optional validation.

    Implemented as a python descriptor object.
    """

    def __init__(self, name, validate=lambda _: True, validation_message="", data_type=str):
        """Args:

        name (str): The name of this hyperparameter validate
        (callable[object]->[bool]): A validation function or list of validation
        functions.

            Each function validates an object and returns False if the object
            value is invalid for this hyperparameter.

        validation_message (str): A usage guide to display on validation
        failure.

        Args:
            name:
            validate:
            validation_message:
            data_type:
        """
        self.validation = validate
        self.validation_message = validation_message
        self.name = name
        self.data_type = data_type
        try:
            iter(self.validation)
        except TypeError:
            self.validation = [self.validation]

    def validate(self, value):
        """Placeholder docstring"""
        if value is None:  # We allow assignment from None, but Nones are not sent to training.
            return

        for valid in self.validation:
            if not valid(value):
                error_message = "Invalid hyperparameter value {} for {}".format(value, self.name)
                if self.validation_message:
                    error_message = error_message + ". Expecting: " + self.validation_message
                raise ValueError(error_message)

    def __get__(self, obj, objtype):
        """Placeholder docstring"""
        if "_hyperparameters" not in dir(obj) or self.name not in obj._hyperparameters:
            raise AttributeError()
        return obj._hyperparameters[self.name]

    def __set__(self, obj, value):
        """Validate the supplied value and set this hyperparameter to value

        Args:
            obj:
            value:
        """
        value = None if value is None else self.data_type(value)
        self.validate(value)
        if "_hyperparameters" not in dir(obj):
            obj._hyperparameters = dict()
        obj._hyperparameters[self.name] = value

    def __delete__(self, obj):
        """Delete this hyperparameter

        Args:
            obj:
        """
        del obj._hyperparameters[self.name]

    @staticmethod
    def serialize_all(obj):
        """Return all non-None ``hyperparameter`` values on ``obj`` as a ``dict[str,str].``

        Args:
            obj:
        """
        if "_hyperparameters" not in dir(obj):
            return {}
        hps = {}
        for k, v in obj._hyperparameters.items():
            if v is not None:
                if isinstance(v, list):
                    v = json.dumps(v)
                elif is_pipeline_variable(v):
                    v = v.to_string()
                else:
                    v = str(v)
                hps[k] = v
        return hps
