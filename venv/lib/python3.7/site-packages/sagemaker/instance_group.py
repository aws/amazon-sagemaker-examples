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
"""Defines the InstanceGroup class that configures a heterogeneous cluster."""
from __future__ import absolute_import


class InstanceGroup(object):
    """The class to create instance groups for a heterogeneous cluster."""

    def __init__(
        self,
        instance_group_name=None,
        instance_type=None,
        instance_count=None,
    ):
        """It initializes an ``InstanceGroup`` instance.

        You can create instance group object of the ``InstanceGroup`` class
        by specifying the instance group configuration arguments.

        For instructions on how to use InstanceGroup objects
        to configure a heterogeneous cluster
        through the SageMaker generic and framework estimator classes, see
        `Train Using a Heterogeneous Cluster
        <https://docs.aws.amazon.com/sagemaker/latest/dg/train-heterogeneous-cluster.html>`_
        in the *Amazon SageMaker developer guide*.

        Args:
            instance_group_name (str): The name of the instance group.
            instance_type (str): The instance type to use in the instance group.
            instance_count (int): The number of instances to use in the instance group.

                .. tip::

                    For more information about available values for the arguments,
                    see `InstanceGroup
                    <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_InstanceGroup.html>`_
                    API in the `Amazon SageMaker API reference`.

        """
        self.instance_group_name = instance_group_name
        self.instance_type = instance_type
        self.instance_count = instance_count

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        return {
            "InstanceGroupName": self.instance_group_name,
            "InstanceType": self.instance_type,
            "InstanceCount": self.instance_count,
        }
