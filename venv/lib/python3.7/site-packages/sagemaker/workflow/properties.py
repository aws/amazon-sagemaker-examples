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
"""The properties definitions for workflow."""
from __future__ import absolute_import

from abc import ABCMeta
from typing import Dict, Union, List

import attr

import botocore.loaders

from sagemaker.workflow.entities import Expression, PipelineVariable


class PropertiesMeta(ABCMeta):
    """Load an internal shapes attribute from the botocore service model

    for sagemaker and emr service.
    """

    _shapes_map = dict()
    _primitive_types = {"string", "boolean", "integer", "float"}

    def __new__(mcs, *args, **kwargs):
        """Loads up the shapes from the botocore service model."""
        if len(mcs._shapes_map.keys()) == 0:
            loader = botocore.loaders.Loader()

            sagemaker_model = loader.load_service_model("sagemaker", "service-2")
            emr_model = loader.load_service_model("emr", "service-2")
            mcs._shapes_map["sagemaker"] = sagemaker_model["shapes"]
            mcs._shapes_map["emr"] = emr_model["shapes"]

        return super().__new__(mcs, *args, **kwargs)


class Properties(PipelineVariable, metaclass=PropertiesMeta):
    """Properties for use in workflow expressions."""

    def __init__(
        self,
        step_name: str,
        path: str = None,
        shape_name: str = None,
        shape_names: List[str] = None,
        service_name: str = "sagemaker",
    ):
        """Create a Properties instance representing the given shape.

        Args:
            step_name (str): The name of the Step this Property belongs to.
            path (str): The relative path of this Property value.
            shape_name (str): The botocore service model shape name.
            shape_names (str): A List of the botocore service model shape name.
        """
        self.step_name = step_name
        self.path = path

        shape_names = [] if shape_names is None else shape_names
        self._shape_names = shape_names if shape_name is None else [shape_name] + shape_names

        shapes = Properties._shapes_map.get(service_name, {})

        for name in self._shape_names:
            shape = shapes.get(name, {})
            shape_type = shape.get("type")
            if shape_type in Properties._primitive_types:
                self.__str__ = name
            elif shape_type == "structure":
                members = shape["members"]
                for key, info in members.items():
                    if shapes.get(info["shape"], {}).get("type") == "list":
                        self.__dict__[key] = PropertiesList(
                            step_name=step_name,
                            path=".".join(filter(None, (path, key))),
                            shape_name=info["shape"],
                            service_name=service_name,
                        )
                    elif shapes.get(info["shape"], {}).get("type") == "map":
                        self.__dict__[key] = PropertiesMap(
                            step_name=step_name,
                            path=".".join(filter(None, (path, key))),
                            shape_name=info["shape"],
                            service_name=service_name,
                        )
                    else:
                        self.__dict__[key] = Properties(
                            step_name=step_name,
                            path=".".join(filter(None, (path, key))),
                            shape_name=info["shape"],
                            service_name=service_name,
                        )

    @property
    def expr(self):
        """The 'Get' expression dict for a `Properties`."""
        prefix = f"Steps.{self.step_name}"
        full_path = prefix if self.path is None else f"{prefix}.{self.path}"
        return {"Get": full_path}

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        return [self.step_name]


class PropertiesList(Properties):
    """PropertiesList for use in workflow expressions."""

    def __init__(
        self, step_name: str, path: str, shape_name: str = None, service_name: str = "sagemaker"
    ):
        """Create a PropertiesList instance representing the given shape.

        Args:
            step_name (str): The name of the Step this Property belongs to.
            path (str): The relative path of this Property value.
            shape_name (str): The botocore service model shape name.
            service_name (str): The botocore service name.
        """
        super(PropertiesList, self).__init__(step_name, path, shape_name)
        self.shape_name = shape_name
        self.service_name = service_name
        self._items: Dict[Union[int, str], Properties] = dict()

    def __getitem__(self, item: Union[int, str]):
        """Populate the indexing item with a Property, for both lists and dictionaries.

        Args:
            item (Union[int, str]): The index of the item in sequence.
        """
        if item not in self._items.keys():
            shape = Properties._shapes_map.get(self.service_name, {}).get(self.shape_name)
            member = shape["member"]["shape"]
            if isinstance(item, str):
                property_item = Properties(self.step_name, f"{self.path}['{item}']", member)
            else:
                property_item = Properties(self.step_name, f"{self.path}[{item}]", member)
            self._items[item] = property_item

        return self._items.get(item)


class PropertiesMap(Properties):
    """PropertiesMap for use in workflow expressions."""

    def __init__(
        self, step_name: str, path: str, shape_name: str = None, service_name: str = "sagemaker"
    ):
        """Create a PropertiesMap instance representing the given shape.

        Args:
            step_name (str): The name of the Step this Property belongs to.
            path (str): The relative path of this Property value.
            shape_name (str): The botocore service model shape name.
            service_name (str): The botocore service name.
        """
        super(PropertiesMap, self).__init__(step_name, path, shape_name)
        self.shape_name = shape_name
        self.service_name = service_name
        self._items: Dict[Union[int, str], Properties] = dict()

    def __getitem__(self, item: Union[int, str]):
        """Populate the indexing item with a Property, for both lists and dictionaries.

        Args:
            item (Union[int, str]): The index of the item in sequence.
        """
        if item not in self._items.keys():
            shape = Properties._shapes_map.get(self.service_name, {}).get(self.shape_name)
            member = shape["value"]["shape"]
            if isinstance(item, str):
                property_item = Properties(self.step_name, f"{self.path}['{item}']", member)
            else:
                property_item = Properties(self.step_name, f"{self.path}[{item}]", member)
            self._items[item] = property_item

        return self._items.get(item)


@attr.s
class PropertyFile(Expression):
    """Provides a property file struct.

    Attributes:
        name (str): The name of the property file for reference with `JsonGet` functions.
        output_name (str): The name of the processing job output channel.
        path (str): The path to the file at the output channel location.
    """

    name: str = attr.ib()
    output_name: str = attr.ib()
    path: str = attr.ib()

    @property
    def expr(self) -> Dict[str, str]:
        """The expression dict for a `PropertyFile`."""
        return {
            "PropertyFileName": self.name,
            "OutputName": self.output_name,
            "FilePath": self.path,
        }
