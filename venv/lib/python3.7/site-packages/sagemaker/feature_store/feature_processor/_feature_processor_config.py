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
"""Contains data classes for the FeatureProcessor."""
from __future__ import absolute_import

from typing import Dict, List, Optional, Sequence, Union

import attr

from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
)
from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode


@attr.s(frozen=True)
class FeatureProcessorConfig:
    """Immutable data class containing the arguments for a FeatureProcessor.

    This class is used throughout sagemaker.feature_store.feature_processor module. Documentation
    for each field can be be found in the feature_processor decorator.

    Defaults are defined as literals in the feature_processor decorator's parameters for usability
    (i.e. literals in docs). Defaults, or any business logic, should not be added to this class.
    It only serves as an immutable data class.
    """

    inputs: Sequence[Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource]] = attr.ib()
    output: str = attr.ib()
    mode: FeatureProcessorMode = attr.ib()
    target_stores: Optional[List[str]] = attr.ib()
    parameters: Optional[Dict[str, Union[str, Dict]]] = attr.ib()
    enable_ingestion: bool = attr.ib()

    @staticmethod
    def create(
        inputs: Sequence[Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource]],
        output: str,
        mode: FeatureProcessorMode,
        target_stores: Optional[List[str]],
        parameters: Optional[Dict[str, Union[str, Dict]]],
        enable_ingestion: bool,
    ) -> "FeatureProcessorConfig":
        """Static initializer."""
        return FeatureProcessorConfig(
            inputs=inputs,
            output=output,
            mode=mode,
            target_stores=target_stores,
            parameters=parameters,
            enable_ingestion=enable_ingestion,
        )
