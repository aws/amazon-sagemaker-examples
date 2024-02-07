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
"""Helper functions for model card."""
from __future__ import absolute_import

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List
import inspect
from enum import Enum
import json
import copy
import hashlib
import collections
from botocore.exceptions import ClientError
from boto3.session import Session


logger = logging.getLogger(__name__)


class _JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for model card"""

    def default(self, o):
        """Add additional condition to customize the encoding strategy for model card object.

        Args:
            o (Any): object.
        """
        if hasattr(o, "_to_request_dict"):
            # custom object with _to_request_dict method
            result = o._to_request_dict()
        else:
            # fall back to default encoding strategy
            result = json.JSONEncoder.default(self, o)

        return result


class _DefaultToRequestDict(object):
    """Provide a default behavior for to_request_dict method."""

    def _clean_descriptor_name(self, name: str):
        """Update the attribute name to be the same as schema.

        Args:
            name (str): attribute name.
        """
        if name.startswith("_") and hasattr(self, name[1:]):
            name = name[1:]

        return name

    def _skip_encoding(self, attr: str):
        """Skip encoding if the attribute is an instance of _SkipEncodingDecoding descriptor"""
        if attr in self.__class__.__dict__:
            return isinstance(self.__class__.__dict__[attr], _SkipEncodingDecoding)

        return False

    def _to_request_dict(self):
        """Implement this method in a subclass to return a custom request_dict."""
        request_data = {}
        for attr, value in self.__dict__.items():
            if value is not None:
                name = self._clean_descriptor_name(attr)
                if not self._skip_encoding(name):
                    request_data[name] = value

        return request_data


class _DefaultFromDict(object):
    """Provide a default behavior for from_dict method."""

    @classmethod
    def _from_dict(cls, raw_data: dict):
        """Implement this method in a subclass to custom load JSON object.

        Args:
            raw_data (dict): model card json data ready to be encode.
        """
        args = copy.deepcopy(raw_data)

        return cls(**args)


class _DescriptorBase(ABC):
    """Base class for model card descriptor source: https://docs.python.org/3/howto/descriptor.html#complete-practical-example."""  # noqa E501  # pylint: disable=c0301

    def __set_name__(self, owner: type, name: str):
        """Set descriptor private name.

        Args:
            owner (type): attribute Parent class.
            name (str): attribute name.
        """
        self.private_name = "_" + name  # pylint: disable=W0201

    def __get__(self, obj, objtype: type = None):
        """Get the object attribute value during obj.attribute.

        Args:
            objtype (type): attribute Parent class (default: None).
        """
        return getattr(obj, self.private_name)

    def __set__(self, obj: object, value: Any):
        """Set the object attribute value during obj.attribute = value.

        Args:
            obj (object): Attribute object.
            value (Any): Value assigned to the attribute.
        """
        self.validate(value)
        if self.require_decode(value):
            # custom decode method
            value = self.decode(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        """Custom validators need to inherit from _DescriptorBase and must supply a validate method to test various restrictions as needed.

        Args:
            value (Any): value to be validated.
        """  # noqa E501  # pylint: disable=c0301
        pass  # pylint: disable=W0107

    @abstractmethod
    def require_decode(self, value: dict):
        """Decide if the object requires to run decode method. Implement this method in a subclass to return a custom request_dict.

        Args:
            value (dict): raw data to be decoded.
        """  # noqa E501  # pylint: disable=c0301
        pass  # pylint: disable=W0107

    @abstractmethod
    def decode(self, value: dict):
        """Decode routine of the JSON object.

        Args:
            value (dict): raw data to be decoded.
        """
        pass  # pylint: disable=W0107


class _SkipEncodingDecoding(_DescriptorBase):
    """Object that skip the encoding/decoding in model card attributes."""

    def __init__(self, value_type: Any):
        """Initialize an SkipEncodingDecoding descriptor.

        Args:
            value_type (Any): Value type of the attribute.
        """
        self.value_type = value_type

    def validate(self, value: Any):
        """Check if value type is valid.

        Args:
            value (Any): value type depends on self.value_type

        Raises:
            ValueError: value is not a self.value_type.
        """
        if value is not None and not isinstance(value, self.value_type):
            raise ValueError(f"Please assign a {self.value_type} to {self.private_name[1:]}")

    def require_decode(self, value: Any):
        """No decoding is required."""
        return False

    def decode(self, value: Any):
        """No decoding is required. Required placeholder for abstractmethod"""
        pass  # pylint: disable=W0107


class _OneOf(_DescriptorBase):
    """Verifies that a value is one of a restricted set of options"""

    def __init__(self, enumerator: Enum):
        """Initialize a OneOf descriptor.

        Args:
            options (Enum): options to be verified.
        """
        self.options = set(enumerator)
        self.enumerator = enumerator
        self.enumerator_reverse = {i.value: i.name for i in self.enumerator}

    def validate(self, value):
        """Check if value is valid for enumerator.

        Args:
            value (str or Enum): enumerator value.

        Raises:
            ValueError: value is not in the enumerator.
        """
        if value is not None and value not in self.options:
            if not isinstance(value, Enum):
                expect = sorted([i.value for i in self.options])
            else:
                expect = sorted([str(i) for i in self.options])

            raise ValueError(f"Expected {str(value)} to be one of {expect}")

    def require_decode(self, value: Union[Enum, str]):
        """Check if the value requires decoding.

        Args:
            value (Enum or str): raw data to be decoded to Enum.
        """
        if isinstance(value, (Enum, type(None))):
            return False
        return True

    def decode(self, value: str):
        """Decode the value to an enumerator.

        Args:
            value (str): raw data to be decoded to Enum.
        """
        return getattr(self.enumerator, self.enumerator_reverse[value])


class _IsList(_DescriptorBase):
    """List object."""

    def __init__(self, item_type: object, max_size: Optional[int] = None):
        """Initialize an IsList descriptor.

        Args:
            item_type (object): Item class in the list.
            max_size (int, optional): max size of the list. Defaults to None, i.e. no size limit for the list.
        """  # noqa E501  # pylint: disable=c0301
        self.item_type = item_type
        self.max_size = max_size

    def validate(self, value: List):
        """Check if value is valid for _MaxSizeList.

        Args:
            value (List)

        Raises:
            ValueError: value is not a list.
        """
        if value is not None and not isinstance(value, list):
            raise ValueError(f"Please assign a list to {self.private_name[1:]}")

    def require_decode(self, value: List):  # pylint: disable=w0613
        """Check if the value requires decoding.

        Args:
            value (List): raw data to be decoded to _MaxSizeList.
        """
        # required to convert the list to _MaxSizeList
        return True

    def decode(self, value: List):
        """Decode the value to a _MaxSizeList.

        Args:
            value (List): raw data to be decoded to _MaxSizeList.
        """
        # default value for _IsList attribute is []
        array = []
        if value is not None:
            for item in value:
                if isinstance(value[0], self.item_type):
                    array.append(item)
                else:
                    array.append(self.item_type._from_dict(item))

        return _MaxSizeArray(self.max_size, self.item_type, array)


class _IsModelCardObject(_DescriptorBase):
    """Model Card object class."""

    def __init__(self, custom_class: object):
        """Initialize a model card object descriptor.

        Args:
            custom_class (object): model card object class.
        """
        self.custom_class = custom_class

    def validate(self, value: Union[dict, object]):
        """Validate if data is value for model card object.

        Args:
            value (dict or object)

        Raises:
            ValueError: value is not a dict or custom_class.
        """
        from_dict = isinstance(value, dict)
        if value is not None and not from_dict and not isinstance(value, self.custom_class):
            raise ValueError(
                f"Expected {type(value)} instance to be of class {self.custom_class.__name__}."
            )  # noqa E501  # pylint: disable=c0301

    def require_decode(self, value: Union[dict, object]):
        """Check if value requires decoding.

        Args:
            value (dict or object): raw data to be decoded to custom_class object.
        """
        if isinstance(value, (self.custom_class, type(None))):
            res = False
        else:
            res = True
        return res

    def decode(self, value: dict):
        """Decode the value to a custom class object.

        Args:
            value (dict): raw data to be decoded to custom_class object.

        Raises:
            TypeError: Attributes in the data don't match the class definition.
        """
        try:
            return self.custom_class._from_dict(value)
        except TypeError as e:
            raise TypeError(f"class {self.custom_class} {str(e)}")


class _MaxSizeArray(collections.abc.MutableSequence):  # pylint: disable=too-many-ancestors
    """Array with maximum size and items of the same type."""

    def __init__(self, max_size: int, item_type: Any, array: List = None):
        """Initialize a Max Size Array.

        Args:
            max_size (int): array max size.
            item_type (Any): array item type.
            array (List, optional): initial array items (default: None).
        """
        super().__init__()
        if max_size is None:
            max_size = float("inf")
        if max_size < 0:
            raise ValueError("Max size has to be positive integer")
        self._max_size = max_size
        if not inspect.isclass(item_type):
            raise ValueError("Item type has to be a class")
        self._item_type = item_type
        self.list = []
        self._initialize_data(array)

    def __len__(self):
        """Return len(self)."""
        return len(self.list)

    def __getitem__(self, index):
        """Get Self[index]

        Args:
            index (int): List index.
        """
        return self.list[index]

    def __delitem__(self, index):
        """Delete self[index].

        Args:
            index (int): List index.
        """
        del self.list[index]

    def __setitem__(self, index: int, value: Any):
        """Set self[key] to value.

        Args:
            index (int): List index.
            value (Any): List element.
        """
        self.check(value)
        self.list[index] = value

    def __str__(self):
        """Return str(self)."""
        return str([i for i in self.list])  # pylint: disable=unnecessary-comprehension

    def __repr__(self):
        """Return repr(self)."""
        return self.__str__()

    def __eq__(self, other: Any):
        """Return self==value.

        Args:
            other (Any): The other object used for comparison.
        """
        return self.list == other

    def _initialize_data(self, array: List):
        """Initialize the max size list from a list.

        Args:
            array (list): initial list data.

        Raises:
            ValueError: size of the array is larger than max_size
        """
        if array:
            if len(array) > self._max_size:
                raise ValueError(
                    f"Data size {len(array)} exceed the maximum size of {self._max_size}"
                )
            for item in array:
                self.append(item)

    def check(self, value: Any):
        """Check if the item is valid.

        Args:
            value (Any): item in the max size list.

        Raises:
            ValueError: Item's type is not the same as item_type.
            ValueError: Result list size is larger than max_size.
        """

        if not isinstance(value, self._item_type):
            raise TypeError(
                f"Provided item type is {type(value)} and Expected the item type is {self._item_type}"  # noqa E501  # pylint: disable=c0301
            )
        if len(self.list) >= self._max_size:
            raise ValueError(f"Exceed the maximum size of {self._max_size}")

    def insert(self, index, value):
        """Insert item into the list. (implicitly overwrite append method)"""
        self.check(value)
        self.list.insert(index, value)

    def _to_request_dict(self):
        """Create data for request body"""
        return list(self)

    def to_map(self, key_attribute: str):
        """Generate a map from the list items for fast look up

        Args:
            key_attribute (str): the attribute in the item where the map key is coming from
        """
        map_ = {}
        for item in self.list:
            key = getattr(item, key_attribute, None)
            if key:
                map_[key] = item

        return map_


def _hash_content_str(content: str):
    """Create hash for content string

    Args:
        content (str): content string from json.dumps.
    """
    dhash = hashlib.md5()
    encoded = json.dumps(json.loads(content), sort_keys=True, default=str).encode()
    dhash.update(encoded)

    return dhash.hexdigest()


def _read_s3_json(session: Session, bucket: str, key: str):
    """Read json file from S3 bucket.

    Args:
        session (Session): boto3 session.
        bucket (str): S3 bucket name.
        key (str): S3 key of the json file.
    """
    client = session.client("s3")
    try:
        data = client.get_object(Bucket=bucket, Key=key)
    except ClientError as ex:
        if ex.response["Error"]["Code"] == "NoSchKey":  # pylint: disable=r1705
            logger.warning("Metric file %s does not exist in %s.", key, bucket)
            return {}
        else:
            raise

    result = {}
    if data["ContentType"] == "application/json" or data["ContentType"] == "binary/octet-stream":
        result = json.loads(data["Body"].read().decode("utf-8"))
    else:
        logger.warning(
            "Invalid file type %s. application/json or binary/octet-stream is expected.",
            data["ContentType"],
        )

    return result
