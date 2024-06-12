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
"""A member of ``CreateEndpointConfig`` that enables explainers."""

from __future__ import print_function, absolute_import
from typing import Optional
from sagemaker.explainer.clarify_explainer_config import ClarifyExplainerConfig


class ExplainerConfig(object):
    """A parameter to activate explainers."""

    def __init__(
        self,
        clarify_explainer_config: Optional[ClarifyExplainerConfig] = None,
    ):
        """Initializes a config object to activate explainer.

        Args:
            clarify_explainer_config (:class:`~sagemaker.explainer.explainer_config.ClarifyExplainerConfig`):
                Optional. A config contains parameters for the SageMaker Clarify explainer. (Default: None)
        """  # noqa E501  # pylint: disable=line-too-long
        self.clarify_explainer_config = clarify_explainer_config

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {}

        if self.clarify_explainer_config:
            request_dict[
                "ClarifyExplainerConfig"
            ] = self.clarify_explainer_config._to_request_dict()

        return request_dict
