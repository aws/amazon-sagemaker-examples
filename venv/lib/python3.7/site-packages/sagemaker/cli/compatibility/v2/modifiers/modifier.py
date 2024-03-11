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
"""Abstract class for modifying AST nodes."""
from __future__ import absolute_import

from abc import abstractmethod


class Modifier(object):
    """Abstract class to check if an AST node needs modification, potentially modify the node."""

    def check_and_modify_node(self, node):
        """Check an AST node, and modify, replace, or remove it if applicable."""
        if self.node_should_be_modified(node):
            node = self.modify_node(node)
        return node

    @abstractmethod
    def node_should_be_modified(self, node):
        """Check if an AST node should be modified."""

    @abstractmethod
    def modify_node(self, node):
        """Modify an AST node."""
