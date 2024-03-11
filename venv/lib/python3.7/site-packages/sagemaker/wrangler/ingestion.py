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
# language governing permissions and limitations under the License.
"""Data wrangler helpers for data ingestion."""
from __future__ import absolute_import

from typing import Dict
from uuid import uuid4
from sagemaker.dataset_definition.inputs import (
    RedshiftDatasetDefinition,
    AthenaDatasetDefinition,
)


def generate_data_ingestion_flow_from_s3_input(
    input_name: str,
    s3_uri: str,
    s3_content_type: str = "csv",
    s3_has_header: bool = False,
    operator_version: str = "0.1",
    schema: Dict = None,
):
    """Generate the data ingestion only flow from s3 input

    Args:
        input_name (str): the name of the input to flow source node
        s3_uri (str): uri for the s3 input to flow source node
        s3_content_type (str): s3 input content type
        s3_has_header (bool): flag indicating the input has header or not
        operator_version: (str): the version of the operator
        schema: (typing.Dict): the schema for the data to be ingested
    Returns:
        dict (typing.Dict): A flow only conduct data ingestion with 1-1 mapping
        output_name (str): The output name used to configure
        `sagemaker.processing.FeatureStoreOutput`
    """
    source_node = {
        "node_id": str(uuid4()),
        "type": "SOURCE",
        "inputs": [],
        "outputs": [{"name": "default"}],
        "operator": f"sagemaker.s3_source_{operator_version}",
        "parameters": {
            "dataset_definition": {
                "datasetSourceType": "S3",
                "name": input_name,
                "s3ExecutionContext": {
                    "s3Uri": s3_uri,
                    "s3ContentType": s3_content_type,
                    "s3HasHeader": s3_has_header,
                },
            }
        },
    }

    output_node = _get_output_node(source_node["node_id"], operator_version, schema)

    flow = {
        "metadata": {"version": 1, "disable_limits": False},
        "nodes": [source_node, output_node],
    }

    return flow, f'{output_node["node_id"]}.default'


def generate_data_ingestion_flow_from_athena_dataset_definition(
    input_name: str,
    athena_dataset_definition: AthenaDatasetDefinition,
    operator_version: str = "0.1",
    schema: Dict = None,
):
    """Generate the data ingestion only flow from athena input

    Args:
        input_name (str): the name of the input to flow source node
        athena_dataset_definition (AthenaDatasetDefinition): athena input to flow source node
        operator_version: (str): the version of the operator
        schema: (typing.Dict): the schema for the data to be ingested
    Returns:
        dict (typing.Dict): A flow only conduct data ingestion with 1-1 mapping
        output_name (str): The output name used to configure
        `sagemaker.processing.FeatureStoreOutput`
    """
    source_node = {
        "node_id": str(uuid4()),
        "type": "SOURCE",
        "inputs": [],
        "outputs": [{"name": "default"}],
        "operator": f"sagemaker.athena_source_{operator_version}",
        "parameters": {
            "dataset_definition": {
                "datasetSourceType": "Athena",
                "name": input_name,
                "catalogName": athena_dataset_definition.catalog,
                "databaseName": athena_dataset_definition.database,
                "queryString": athena_dataset_definition.query_string,
                "s3OutputLocation": athena_dataset_definition.output_s3_uri,
                "outputFormat": athena_dataset_definition.output_format,
            }
        },
    }

    output_node = _get_output_node(source_node["node_id"], operator_version, schema)

    flow = {
        "metadata": {"version": 1, "disable_limits": False},
        "nodes": [source_node, output_node],
    }

    return flow, f'{output_node["node_id"]}.default'


def generate_data_ingestion_flow_from_redshift_dataset_definition(
    input_name: str,
    redshift_dataset_definition: RedshiftDatasetDefinition,
    operator_version: str = "0.1",
    schema: Dict = None,
):
    """Generate the data ingestion only flow from redshift input

    Args:
        input_name (str): the name of the input to flow source node
        redshift_dataset_definition (RedshiftDatasetDefinition): redshift input to flow source node
        operator_version: (str): the version of the operator
        schema: (typing.Dict): the schema for the data to be ingested
    Returns:
        dict (typing.Dict): A flow only conduct data ingestion with 1-1 mapping
        output_name (str): The output name used to configure
        `sagemaker.processing.FeatureStoreOutput`
    """
    source_node = {
        "node_id": str(uuid4()),
        "type": "SOURCE",
        "inputs": [],
        "outputs": [{"name": "default"}],
        "operator": f"sagemaker.redshift_source_{operator_version}",
        "parameters": {
            "dataset_definition": {
                "datasetSourceType": "Redshift",
                "name": input_name,
                "clusterIdentifier": redshift_dataset_definition.cluster_id,
                "database": redshift_dataset_definition.database,
                "dbUser": redshift_dataset_definition.db_user,
                "queryString": redshift_dataset_definition.query_string,
                "unloadIamRole": redshift_dataset_definition.cluster_role_arn,
                "s3OutputLocation": redshift_dataset_definition.output_s3_uri,
                "outputFormat": redshift_dataset_definition.output_format,
            }
        },
    }

    output_node = _get_output_node(source_node["node_id"], operator_version, schema)

    flow = {
        "metadata": {"version": 1, "disable_limits": False},
        "nodes": [source_node, output_node],
    }

    return flow, f'{output_node["node_id"]}.default'


def _get_output_node(source_node_id: str, operator_version: str, schema: Dict):
    """A helper function to generate output node, for internal use only

    Args:
        source_node_id (str): source node id
        operator_version: (str): the version of the operator
        schema: (typing.Dict): the schema for the data to be ingested
    Returns:
        dict (typing.Dict): output node
    """
    return {
        "node_id": str(uuid4()),
        "type": "TRANSFORM",
        "operator": f"sagemaker.spark.infer_and_cast_type_{operator_version}",
        "trained_parameters": {} if schema is None else schema,
        "parameters": {},
        "inputs": [{"name": "default", "node_id": source_node_id, "output_name": "default"}],
        "outputs": [{"name": "default"}],
    }
