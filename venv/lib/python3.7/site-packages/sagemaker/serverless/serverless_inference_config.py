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
"""This module contains code related to the ServerlessInferenceConfig class.

Codes are used for configuring serverless inference endpoint. Use it when deploying
the model to the endpoints.
"""
from __future__ import print_function, absolute_import
from typing import Optional


class ServerlessInferenceConfig(object):
    """Configuration object passed in when deploying models to Amazon SageMaker Endpoints.

    This object specifies configuration related to serverless endpoint. Use this configuration
    when trying to create serverless endpoint and make serverless inference
    """

    def __init__(
        self,
        memory_size_in_mb: int = 2048,
        max_concurrency: int = 5,
        provisioned_concurrency: Optional[int] = None,
    ):
        """Initialize a ServerlessInferenceConfig object for serverless inference configuration.

        Args:
            memory_size_in_mb (int): Optional. The memory size of your serverless endpoint.
                Valid values are in 1 GB increments: 1024 MB, 2048 MB, 3072 MB, 4096 MB,
                5120 MB, or 6144 MB. If no value is provided, Amazon SageMaker will choose
                the default value for you. (Default: 2048)
            max_concurrency (int): Optional. The maximum number of concurrent invocations
                your serverless endpoint can process. If no value is provided, Amazon
                SageMaker will choose the default value for you. (Default: 5)
            provisioned_concurrency (int): Optional. The provisioned concurrency of your
                serverless endpoint. If no value is provided, Amazon SageMaker will not
                apply provisioned concucrrency to your Serverless endpoint. (Default: None)
        """
        self.memory_size_in_mb = memory_size_in_mb
        self.max_concurrency = max_concurrency
        self.provisioned_concurrency = provisioned_concurrency

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {
            "MemorySizeInMB": self.memory_size_in_mb,
            "MaxConcurrency": self.max_concurrency,
        }

        if self.provisioned_concurrency is not None:
            request_dict["ProvisionedConcurrency"] = self.provisioned_concurrency

        return request_dict
