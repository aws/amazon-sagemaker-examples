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
"""SageMaker lineage utility methods."""
from __future__ import absolute_import
from sagemaker.lineage import association


def _disassociate(source_arn=None, destination_arn=None, sagemaker_session=None):
    """Remove the association.

    Remove incoming association when source_arn is provided, remove outgoing association when
    destination_arn is provided.
    """
    association_summaries = association.Association.list(
        source_arn=source_arn,
        destination_arn=destination_arn,
        sagemaker_session=sagemaker_session,
    )

    for association_summary in association_summaries:

        curr_association = association.Association(
            sagemaker_session=sagemaker_session,
            source_arn=association_summary.source_arn,
            destination_arn=association_summary.destination_arn,
        )
        curr_association.delete()


def get_resource_name_from_arn(arn):
    """Extract the resource name from an ARN string.

    Args:
        arn (str): An ARN.

    Returns:
        str: The resource name.
    """
    return arn.split(":", 5)[5].split("/", 1)[1]
