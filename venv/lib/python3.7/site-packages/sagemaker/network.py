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
"""This file contains code related to network configuration.

It also includes encryption, network isolation, and VPC configurations.
"""
from __future__ import absolute_import

from typing import Union, Optional, List

from sagemaker.workflow.entities import PipelineVariable


class NetworkConfig(object):
    """Accepts network configuration parameters for conversion to request dict.

    The `_to_request_dict` provides a method to turn the parameters into a dict.
    """

    def __init__(
        self,
        enable_network_isolation: Union[bool, PipelineVariable] = None,
        security_group_ids: Optional[List[Union[str, PipelineVariable]]] = None,
        subnets: Optional[List[Union[str, PipelineVariable]]] = None,
        encrypt_inter_container_traffic: Optional[Union[bool, PipelineVariable]] = None,
    ):
        """Initialize a ``NetworkConfig`` instance.

        NetworkConfig accepts network configuration parameters and provides a method to turn
        these parameters into a dictionary.

        Args:
            enable_network_isolation (bool or PipelineVariable): Boolean that determines
                whether to enable network isolation.
            security_group_ids (list[str] or list[PipelineVariable]): A list of strings representing
                security group IDs.
            subnets (list[str] or list[PipelineVariable]): A list of strings representing subnets.
            encrypt_inter_container_traffic (bool or PipelineVariable): Boolean that determines
                whether to encrypt inter-container traffic. Default value is None.
        """
        self.enable_network_isolation = enable_network_isolation
        self.security_group_ids = security_group_ids
        self.subnets = subnets
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        # Enable Network Isolation should default to False if it is not provided.
        enable_network_isolation = (
            False if self.enable_network_isolation is None else self.enable_network_isolation
        )
        network_config_request = {"EnableNetworkIsolation": enable_network_isolation}

        if self.encrypt_inter_container_traffic is not None:
            network_config_request[
                "EnableInterContainerTrafficEncryption"
            ] = self.encrypt_inter_container_traffic

        if self.security_group_ids is not None or self.subnets is not None:
            network_config_request["VpcConfig"] = {}

        if self.security_group_ids is not None:
            network_config_request["VpcConfig"]["SecurityGroupIds"] = self.security_group_ids

        if self.subnets is not None:
            network_config_request["VpcConfig"]["Subnets"] = self.subnets

        return network_config_request
