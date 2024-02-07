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
"""Implements base methods for serializing data for an inference endpoint."""
from __future__ import absolute_import

import abc
from collections.abc import Iterable
import csv
import io
import json
import numpy as np
from pandas import DataFrame
from six import with_metaclass

from sagemaker.utils import DeferredError

try:
    import scipy.sparse
except ImportError as e:
    scipy = DeferredError(e)


class BaseSerializer(abc.ABC):
    """Abstract base class for creation of new serializers.

    Provides a skeleton for customization requiring the overriding of the method
    serialize and the class attribute CONTENT_TYPE.
    """

    @abc.abstractmethod
    def serialize(self, data):
        """Serialize data into the media type specified by CONTENT_TYPE.

        Args:
            data (object): Data to be serialized.

        Returns:
            object: Serialized data used for a request.
        """

    @property
    @abc.abstractmethod
    def CONTENT_TYPE(self):
        """The MIME type of the data sent to the inference endpoint."""


class SimpleBaseSerializer(with_metaclass(abc.ABCMeta, BaseSerializer)):
    """Abstract base class for creation of new serializers.

    This class extends the API of :class:~`sagemaker.serializers.BaseSerializer` with more
    user-friendly options for setting the Content-Type header, in situations where it can be
    provided at init and freely updated.
    """

    def __init__(self, content_type="application/json"):
        """Initialize a ``SimpleBaseSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
            request data (default: "application/json").
        """
        super(SimpleBaseSerializer, self).__init__()
        if not isinstance(content_type, str):
            raise ValueError(
                "content_type must be a string specifying the MIME type of the data sent in "
                "requests: e.g. 'application/json', 'text/csv', etc. Got %s" % content_type
            )
        self.content_type = content_type

    @property
    def CONTENT_TYPE(self):
        """The data MIME type set in the Content-Type header on prediction endpoint requests."""
        return self.content_type


class CSVSerializer(SimpleBaseSerializer):
    """Serialize data of various formats to a CSV-formatted string."""

    def __init__(self, content_type="text/csv"):
        """Initialize a ``CSVSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "text/csv").
        """
        super(CSVSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data of various formats to a CSV-formatted string.

        Args:
            data (object): Data to be serialized. Can be a NumPy array, list,
                file, Pandas DataFrame, or buffer.

        Returns:
            str: The data serialized as a CSV-formatted string.
        """
        if hasattr(data, "read"):
            return data.read()

        if isinstance(data, DataFrame):
            return data.to_csv(header=False, index=False)

        is_mutable_sequence_like = self._is_sequence_like(data) and hasattr(data, "__setitem__")
        has_multiple_rows = len(data) > 0 and self._is_sequence_like(data[0])

        if is_mutable_sequence_like and has_multiple_rows:
            return "\n".join([self._serialize_row(row) for row in data])

        return self._serialize_row(data)

    def _serialize_row(self, data):
        """Serialize data as a CSV-formatted row.

        Args:
            data (object): Data to be serialized in a row.

        Returns:
            str: The data serialized as a CSV-formatted row.
        """
        if isinstance(data, str):
            return data

        if isinstance(data, np.ndarray):
            data = np.ndarray.flatten(data)

        if hasattr(data, "__len__"):
            if len(data) == 0:
                raise ValueError("Cannot serialize empty array")
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer, delimiter=",")
            csv_writer.writerow(data)
            return csv_buffer.getvalue().rstrip("\r\n")

        raise ValueError("Unable to handle input format: %s" % type(data))

    def _is_sequence_like(self, data):
        """Returns true if obj is iterable and subscriptable."""
        return hasattr(data, "__iter__") and hasattr(data, "__getitem__")


class NumpySerializer(SimpleBaseSerializer):
    """Serialize data to a buffer using the .npy format."""

    def __init__(self, dtype=None, content_type="application/x-npy"):
        """Initialize a ``NumpySerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/x-npy").
            dtype (str): The dtype of the data.
        """
        super(NumpySerializer, self).__init__(content_type=content_type)
        self.dtype = dtype

    def serialize(self, data):
        """Serialize data to a buffer using the .npy format.

        Args:
            data (object): Data to be serialized. Can be a NumPy array, list,
                file, or buffer.

        Returns:
            io.BytesIO: A buffer containing data serialzied in the .npy format.
        """
        if isinstance(data, np.ndarray):
            if data.size == 0:
                raise ValueError("Cannot serialize empty array.")
            return self._serialize_array(data)

        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Cannot serialize empty array.")
            return self._serialize_array(np.array(data, self.dtype))

        # files and buffers. Assumed to hold npy-formatted data.
        if hasattr(data, "read"):
            return data.read()

        return self._serialize_array(np.array(data))

    def _serialize_array(self, array):
        """Saves a NumPy array in a buffer.

        Args:
            array (numpy.ndarray): The array to serialize.

        Returns:
            io.BytesIO: A buffer containing the serialized array.
        """
        buffer = io.BytesIO()
        np.save(buffer, array)
        return buffer.getvalue()


class JSONSerializer(SimpleBaseSerializer):
    """Serialize data to a JSON formatted string."""

    def serialize(self, data):
        """Serialize data of various formats to a JSON formatted string.

        Args:
            data (object): Data to be serialized.

        Returns:
            str: The data serialized as a JSON string.
        """
        if isinstance(data, dict):
            return json.dumps(
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in data.items()
                }
            )

        if hasattr(data, "read"):
            return data.read()

        if isinstance(data, np.ndarray):
            return json.dumps(data.tolist())

        return json.dumps(data)


class IdentitySerializer(SimpleBaseSerializer):
    """Serialize data by returning data without modification.

    This serializer may be useful if, for example, you're sending raw bytes such as from an image
    file's .read() method.
    """

    def __init__(self, content_type="application/octet-stream"):
        """Initialize an ``IdentitySerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/octet-stream").
        """
        super(IdentitySerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Return data without modification.

        Args:
            data (object): Data to be serialized.

        Returns:
            object: The unmodified data.
        """
        return data


class JSONLinesSerializer(SimpleBaseSerializer):
    """Serialize data to a JSON Lines formatted string."""

    def __init__(self, content_type="application/jsonlines"):
        """Initialize a ``JSONLinesSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/jsonlines").
        """
        super(JSONLinesSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data of various formats to a JSON Lines formatted string.

        Args:
            data (object): Data to be serialized. The data can be a string,
                iterable of JSON serializable objects, or a file-like object.

        Returns:
            str: The data serialized as a string containing newline-separated
                JSON values.
        """
        if isinstance(data, str):
            return data

        if hasattr(data, "read"):
            return data.read()

        if isinstance(data, Iterable):
            return "\n".join(json.dumps(element) for element in data)

        raise ValueError("Object of type %s is not JSON Lines serializable." % type(data))


class SparseMatrixSerializer(SimpleBaseSerializer):
    """Serialize a sparse matrix to a buffer using the .npz format."""

    def __init__(self, content_type="application/x-npz"):
        """Initialize a ``SparseMatrixSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/x-npz").
        """
        super(SparseMatrixSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize a sparse matrix to a buffer using the .npz format.

        Sparse matrices can be in the ``csc``, ``csr``, ``bsr``, ``dia`` or
        ``coo`` formats.

        Args:
            data (scipy.sparse.spmatrix): The sparse matrix to serialize.

        Returns:
            io.BytesIO: A buffer containing the serialized sparse matrix.
        """
        buffer = io.BytesIO()
        scipy.sparse.save_npz(buffer, data)
        return buffer.getvalue()


class LibSVMSerializer(SimpleBaseSerializer):
    """Serialize data of various formats to a LibSVM-formatted string.

    The data must already be in LIBSVM file format:
    <label> <index1>:<value1> <index2>:<value2> ...

    It is suitable for sparse datasets since it does not store zero-valued
    features.
    """

    def __init__(self, content_type="text/libsvm"):
        """Initialize a ``LibSVMSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "text/libsvm").
        """
        super(LibSVMSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data of various formats to a LibSVM-formatted string.

        Args:
            data (object): Data to be serialized. Can be a string or a
                file-like object.

        Returns:
            str: The data serialized as a LibSVM-formatted string.
        """
        if isinstance(data, str):
            return data

        if hasattr(data, "read"):
            return data.read()

        raise ValueError("Unable to handle input format: %s" % type(data))


class DataSerializer(SimpleBaseSerializer):
    """Serialize data in any file by extracting raw bytes from the file."""

    def __init__(self, content_type="file-path/raw-bytes"):
        """Initialize a ``DataSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "file-path/raw-bytes").
        """
        super(DataSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize file data to a raw bytes.

        Args:
            data (object): Data to be serialized. The data can be a string
                representing file-path or the raw bytes from a file.
        Returns:
            raw-bytes: The data serialized as raw-bytes from the input.
        """
        if isinstance(data, str):
            try:
                with open(data, "rb") as data_file:
                    data_file_info = data_file.read()
                    return data_file_info
            except Exception as e:
                raise ValueError(f"Could not open/read file: {data}. {e}")
        if isinstance(data, bytes):
            return data

        raise ValueError(f"Object of type {type(data)} is not Data serializable.")
