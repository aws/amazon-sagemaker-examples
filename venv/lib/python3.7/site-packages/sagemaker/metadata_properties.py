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
"""This file contains code related to metadata properties."""
from __future__ import absolute_import

from typing import Optional, Union

from sagemaker.workflow.entities import PipelineVariable


class MetadataProperties(object):
    """Accepts metadata properties parameters for conversion to request dict."""

    def __init__(
        self,
        commit_id: Optional[Union[str, PipelineVariable]] = None,
        repository: Optional[Union[str, PipelineVariable]] = None,
        generated_by: Optional[Union[str, PipelineVariable]] = None,
        project_id: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Initialize a ``MetadataProperties`` instance and turn parameters into dict.

        # TODO: flesh out docstrings
        Args:
            commit_id (str or PipelineVariable):
            repository (str or PipelineVariable):
            generated_by (str or PipelineVariable):
            project_id (str or PipelineVariable):
        """
        self.commit_id = commit_id
        self.repository = repository
        self.generated_by = generated_by
        self.project_id = project_id

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        metadata_properties_request = dict()
        if self.commit_id:
            metadata_properties_request["CommitId"] = self.commit_id
        if self.repository:
            metadata_properties_request["Repository"] = self.repository
        if self.generated_by:
            metadata_properties_request["GeneratedBy"] = self.generated_by
        if self.project_id:
            metadata_properties_request["ProjectId"] = self.project_id
        return metadata_properties_request
