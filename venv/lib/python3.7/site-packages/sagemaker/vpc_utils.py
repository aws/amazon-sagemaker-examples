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
"""Placeholder docstring"""
from __future__ import absolute_import

SUBNETS_KEY = "Subnets"
SECURITY_GROUP_IDS_KEY = "SecurityGroupIds"
VPC_CONFIG_KEY = "VpcConfig"

# A global constant value for methods which can optionally override VpcConfig
# Using the default implies that VpcConfig should be reused from an existing Estimator or
# Training Job
VPC_CONFIG_DEFAULT = "VPC_CONFIG_DEFAULT"


def to_dict(subnets, security_group_ids):
    """Prepares a VpcConfig dict containing keys 'Subnets' and 'SecurityGroupIds'.

    This is the dict format expected by SageMaker CreateTrainingJob and CreateModel APIs.
    See https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html

    Args:
        subnets (list): list of subnet IDs to use in VpcConfig
        security_group_ids (list): list of security group IDs to use in
            VpcConfig

    Returns:
        A VpcConfig dict containing keys 'Subnets' and 'SecurityGroupIds' If
        either or both parameters are None, returns None
    """
    if subnets is None or security_group_ids is None:
        return None
    return {SUBNETS_KEY: subnets, SECURITY_GROUP_IDS_KEY: security_group_ids}


def from_dict(vpc_config, do_sanitize=False):
    """Extracts subnets and security group ids as lists from a VpcConfig dict

    Args:
        vpc_config (dict): a VpcConfig dict containing 'Subnets' and
            'SecurityGroupIds'
        do_sanitize (bool): whether to sanitize the VpcConfig dict before
            extracting values

    Returns:
        Tuple of lists as (subnets, security_group_ids) If vpc_config parameter
        is None, returns (None, None)

    Raises:
        * ValueError if sanitize enabled and vpc_config is invalid

        * KeyError if sanitize disabled and vpc_config is missing key(s)
    """
    if do_sanitize:
        vpc_config = sanitize(vpc_config)
    if vpc_config is None:
        return None, None
    return vpc_config[SUBNETS_KEY], vpc_config[SECURITY_GROUP_IDS_KEY]


def sanitize(vpc_config):
    """Checks and removes unexpected keys from VpcConfig or raises error for violations.

    Checks that an instance of VpcConfig has the expected keys and values,
    removes unexpected keys, and raises ValueErrors if any expectations are
    violated.

    Args:
        vpc_config (dict): a VpcConfig dict containing 'Subnets' and
            'SecurityGroupIds'

    Returns:
        A valid VpcConfig dict containing only 'Subnets' and 'SecurityGroupIds'
        from the vpc_config parameter If vpc_config parameter is None, returns
        None

    Raises:
        ValueError if any expectations are violated:
            * vpc_config must be a non-empty dict
            * vpc_config must have key `Subnets` and the value must be a non-empty list
            * vpc_config must have key `SecurityGroupIds` and the value must be a non-empty list
    """
    if vpc_config is None:
        return vpc_config
    if not isinstance(vpc_config, dict):
        raise ValueError("vpc_config is not a dict: {}".format(vpc_config))
    if not vpc_config:
        raise ValueError("vpc_config is empty")

    subnets = vpc_config.get(SUBNETS_KEY)
    if subnets is None:
        raise ValueError("vpc_config is missing key: {}".format(SUBNETS_KEY))
    if not isinstance(subnets, list):
        raise ValueError("vpc_config value for {} is not a list: {}".format(SUBNETS_KEY, subnets))
    if not subnets:
        raise ValueError("vpc_config value for {} is empty".format(SUBNETS_KEY))

    security_group_ids = vpc_config.get(SECURITY_GROUP_IDS_KEY)
    if security_group_ids is None:
        raise ValueError("vpc_config is missing key: {}".format(SECURITY_GROUP_IDS_KEY))
    if not isinstance(security_group_ids, list):
        raise ValueError(
            "vpc_config value for {} is not a list: {}".format(
                SECURITY_GROUP_IDS_KEY, security_group_ids
            )
        )
    if not security_group_ids:
        raise ValueError("vpc_config value for {} is empty".format(SECURITY_GROUP_IDS_KEY))

    return to_dict(subnets, security_group_ids)
