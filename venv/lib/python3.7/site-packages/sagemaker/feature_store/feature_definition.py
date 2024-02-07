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
"""The Feature Definitions for FeatureStore.

A feature is a measurable property or characteristic that encapsulates an observed phenomenon.
In the Amazon SageMaker Feature Store API, a feature is an attribute of a record.
You can define a name and type for every feature stored in Feature Store. Name uniquely
identifies a feature within a feature group. Type identifies
the datatype for the values of the Feature.
"""
from __future__ import absolute_import

from enum import Enum
from typing import Dict, Any

import attr

from sagemaker.feature_store.inputs import Config


class FeatureTypeEnum(Enum):
    """Enum of feature types.

    The data type of a feature can be Fractional, Integral or String.
    """

    FRACTIONAL = "Fractional"
    INTEGRAL = "Integral"
    STRING = "String"


@attr.s
class FeatureDefinition(Config):
    """Feature definition.

    This instantiates a Feature Definition object where FeatureDefinition is a subclass of Config.

    Attributes:
        feature_name (str): The name of the feature
        feature_type (FeatureTypeEnum): The type of the feature
    """

    feature_name: str = attr.ib()
    feature_type: FeatureTypeEnum = attr.ib()

    def to_dict(self) -> Dict[str, Any]:
        """Construct a dictionary based on each attribute."""
        return Config.construct_dict(
            FeatureName=self.feature_name, FeatureType=self.feature_type.value
        )


class FractionalFeatureDefinition(FeatureDefinition):
    """Fractional feature definition.

    This class instantiates a FractionalFeatureDefinition object, a subclass of FeatureDefinition
    where the data type of the feature being defined is a Fractional.

    Attributes:
        feature_name (str): The name of the feature
        feature_type (FeatureTypeEnum): A `FeatureTypeEnum.FRACTIONAL` type
    """

    def __init__(self, feature_name: str):
        """Construct an instance of FractionalFeatureDefinition.

        Args:
            feature_name (str): the name of the feature.
        """
        super(FractionalFeatureDefinition, self).__init__(feature_name, FeatureTypeEnum.FRACTIONAL)


class IntegralFeatureDefinition(FeatureDefinition):
    """Fractional feature definition.

    This class instantiates a IntegralFeatureDefinition object, a subclass of FeatureDefinition
    where the data type of the feature being defined is a Integral.

    Attributes:
        feature_name (str): the name of the feature.
        feature_type (FeatureTypeEnum): a `FeatureTypeEnum.INTEGRAL` type.
    """

    def __init__(self, feature_name: str):
        """Construct an instance of IntegralFeatureDefinition.

        Args:
            feature_name (str): the name of the feature.
        """
        super(IntegralFeatureDefinition, self).__init__(feature_name, FeatureTypeEnum.INTEGRAL)


class StringFeatureDefinition(FeatureDefinition):
    """Fractional feature definition.

    This class instantiates a StringFeatureDefinition object, a subclass of FeatureDefinition
    where the data type of the feature being defined is a String.

    Attributes:
        feature_name (str): the name of the feature.
        feature_type (FeatureTypeEnum): a `FeatureTypeEnum.STRING` type.
    """

    def __init__(self, feature_name: str):
        """Construct an instance of StringFeatureDefinition.

        Args:
            feature_name (str): the name of the feature.
        """
        super(StringFeatureDefinition, self).__init__(feature_name, FeatureTypeEnum.STRING)
