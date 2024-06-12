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
"""Feature Processor decorator for feature transformation functions."""
from __future__ import absolute_import

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from sagemaker.feature_store.feature_processor import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
)
from sagemaker.feature_store.feature_processor._enums import FeatureProcessorMode
from sagemaker.feature_store.feature_processor._factory import (
    UDFWrapperFactory,
    ValidatorFactory,
)
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)


def feature_processor(
    inputs: Sequence[Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource]],
    output: str,
    target_stores: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Union[str, Dict]]] = None,
    enable_ingestion: bool = True,
) -> Callable:
    """Decorator to facilitate feature engineering for Feature Groups.

    If the decorated function is executed without arguments then the decorated function's arguments
    are automatically loaded from the input data sources. Outputs are ingested to the output Feature
    Group. If arguments are provided to this function, then arguments are not automatically loaded
    (for testing).

    Decorated functions must conform to the expected signature. Parameters: one parameter of type
    pyspark.sql.DataFrame for each DataSource in 'inputs'; followed by the optional parameters with
    names nand types in [params: Dict[str, Any], spark: SparkSession]. Outputs: a single return
    value of type pyspark.sql.DataFrame. The function can have any name.

    Example:
        @feature_processor(
            inputs=[FeatureGroupDataSource("input-fg"), CSVDataSource("s3://bucket/prefix)],
            output='arn:aws:sagemaker:us-west-2:123456789012:feature-group/output-fg'
        )
        def transform(
            input_feature_group: DataFrame, input_csv: DataFrame, params: Dict[str, Any],
            spark: SparkSession
        ) -> DataFrame:
            return ...

    More concisely:
        @feature_processor(
            inputs=[FeatureGroupDataSource("input-fg"), CSVDataSource("s3://bucket/prefix)],
            output='arn:aws:sagemaker:us-west-2:123456789012:feature-group/output-fg'
        )
        def transform(input_feature_group, input_csv):
            return ...

    Args:
        inputs (Sequence[Union[FeatureGroupDataSource, CSVDataSource, ParquetDataSource]]): A list
            of data sources.
        output (str): A Feature Group ARN to write results of this function to.
        target_stores (Optional[list[str]], optional): A list containing at least one of
            'OnlineStore' or 'OfflineStore'. If unspecified, data will be ingested to the enabled
            stores of the output feature group. Defaults to None.
        parameters (Optional[Dict[str, Union[str, Dict]]], optional): Parameters to be provided to
            the decorated function, available as the 'params' argument. Useful for parameterized
            functions. The params argument also contains the set of system provided parameters
            under the key 'system'. E.g. 'scheduled_time': a timestamp representing the time that
            the execution was scheduled to execute at, if triggered by a Scheduler, otherwise, the
            current time.
        enable_ingestion (bool, optional): A boolean indicating whether the decorated function's
            return value is ingested to the 'output' Feature Group. This flag is useful during the
            development phase to ensure that data is not used until the function is ready. It also
            useful for users that want to manage their own data ingestion. Defaults to True.

    Raises:
        IngestionError: If any rows are not ingested successfully then a sample of the records,
            with failure reasons, is logged.

    Returns:
        Callable: The decorated function.
    """

    def decorator(udf: Callable[..., Any]) -> Callable:
        fp_config = FeatureProcessorConfig.create(
            inputs=inputs,
            output=output,
            mode=FeatureProcessorMode.PYSPARK,
            target_stores=target_stores,
            parameters=parameters,
            enable_ingestion=enable_ingestion,
        )

        validator_chain = ValidatorFactory.get_validation_chain(fp_config)
        udf_wrapper = UDFWrapperFactory.get_udf_wrapper(fp_config)

        validator_chain.validate(udf=udf, fp_config=fp_config)
        wrapped_function = udf_wrapper.wrap(udf=udf, fp_config=fp_config)

        wrapped_function.feature_processor_config = fp_config

        return wrapped_function

    return decorator
