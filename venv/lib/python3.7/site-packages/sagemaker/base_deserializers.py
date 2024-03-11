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
"""Implements base methods for deserializing data returned from an inference endpoint."""
from __future__ import absolute_import

import csv

import abc
import codecs
import io
import json

import numpy as np
from six import with_metaclass

from sagemaker.utils import DeferredError

try:
    import pandas
except ImportError as e:
    pandas = DeferredError(e)


class BaseDeserializer(abc.ABC):
    """Abstract base class for creation of new deserializers.

    Provides a skeleton for customization requiring the overriding of the method
    deserialize and the class attribute ACCEPT.
    """

    @abc.abstractmethod
    def deserialize(self, stream, content_type):
        """Deserialize data received from an inference endpoint.

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            object: The data deserialized into an object.
        """

    @property
    @abc.abstractmethod
    def ACCEPT(self):
        """The content types that are expected from the inference endpoint."""


class SimpleBaseDeserializer(with_metaclass(abc.ABCMeta, BaseDeserializer)):
    """Abstract base class for creation of new deserializers.

    This class extends the API of :class:~`sagemaker.deserializers.BaseDeserializer` with more
    user-friendly options for setting the ACCEPT content type header, in situations where it can be
    provided at init and freely updated.
    """

    def __init__(self, accept="*/*"):
        """Initialize a ``SimpleBaseDeserializer`` instance.

        Args:
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: "*/*").
        """
        super(SimpleBaseDeserializer, self).__init__()
        self.accept = accept

    @property
    def ACCEPT(self):
        """The tuple of possible content types that are expected from the inference endpoint."""
        if isinstance(self.accept, str):
            return (self.accept,)
        return self.accept


class StringDeserializer(SimpleBaseDeserializer):
    """Deserialize data from an inference endpoint into a decoded string."""

    def __init__(self, encoding="UTF-8", accept="application/json"):
        """Initialize a ``StringDeserializer`` instance.

        Args:
            encoding (str): The string encoding to use (default: UTF-8).
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: "application/json").
        """
        super(StringDeserializer, self).__init__(accept=accept)
        self.encoding = encoding

    def deserialize(self, stream, content_type):
        """Deserialize data from an inference endpoint into a decoded string.

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            str: The data deserialized into a decoded string.
        """
        try:
            return stream.read().decode(self.encoding)
        finally:
            stream.close()


class BytesDeserializer(SimpleBaseDeserializer):
    """Deserialize a stream of bytes into a bytes object."""

    def deserialize(self, stream, content_type):
        """Read a stream of bytes returned from an inference endpoint.

        Args:
            stream (botocore.response.StreamingBody): A stream of bytes.
            content_type (str): The MIME type of the data.

        Returns:
            bytes: The bytes object read from the stream.
        """
        try:
            return stream.read()
        finally:
            stream.close()


class CSVDeserializer(SimpleBaseDeserializer):
    """Deserialize a stream of bytes into a list of lists.

    Consider using :class:~`sagemaker.deserializers.NumpyDeserializer` or
    :class:~`sagemaker.deserializers.PandasDeserializer` instead, if you'd like to convert text/csv
    responses directly into other data types.
    """

    def __init__(self, encoding="utf-8", accept="text/csv"):
        """Initialize a ``CSVDeserializer`` instance.

        Args:
            encoding (str): The string encoding to use (default: "utf-8").
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: "text/csv").
        """
        super(CSVDeserializer, self).__init__(accept=accept)
        self.encoding = encoding

    def deserialize(self, stream, content_type):
        """Deserialize data from an inference endpoint into a list of lists.

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            list: The data deserialized into a list of lists representing the
                contents of a CSV file.
        """
        try:
            decoded_string = stream.read().decode(self.encoding)
            return list(csv.reader(decoded_string.splitlines()))
        finally:
            stream.close()


class StreamDeserializer(SimpleBaseDeserializer):
    """Directly return the data and content-type received from an inference endpoint.

    It is the user's responsibility to close the data stream once they're done
    reading it.
    """

    def deserialize(self, stream, content_type):
        """Returns a stream of the response body and the MIME type of the data.

        Args:
            stream (botocore.response.StreamingBody): A stream of bytes.
            content_type (str): The MIME type of the data.

        Returns:
            tuple: A two-tuple containing the stream and content-type.
        """
        return stream, content_type


class NumpyDeserializer(SimpleBaseDeserializer):
    """Deserialize a stream of data in .npy, .npz or UTF-8 CSV/JSON format to a numpy array.

    Note that when using application/x-npz archive format, the result will usually be a
    dictionary-like object containing multiple arrays (as per ``numpy.load()``) - instead of a
    single array.
    """

    def __init__(self, dtype=None, accept="application/x-npy", allow_pickle=True):
        """Initialize a ``NumpyDeserializer`` instance.

        Args:
            dtype (str): The dtype of the data (default: None).
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: "application/x-npy").
            allow_pickle (bool): Allow loading pickled object arrays (default: True).
        """
        super(NumpyDeserializer, self).__init__(accept=accept)
        self.dtype = dtype
        self.allow_pickle = allow_pickle

    def deserialize(self, stream, content_type):
        """Deserialize data from an inference endpoint into a NumPy array.

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            numpy.ndarray: The data deserialized into a NumPy array.
        """
        try:
            if content_type == "text/csv":
                return np.genfromtxt(
                    codecs.getreader("utf-8")(stream), delimiter=",", dtype=self.dtype
                )
            if content_type == "application/json":
                return np.array(json.load(codecs.getreader("utf-8")(stream)), dtype=self.dtype)
            if content_type == "application/x-npy":
                return np.load(io.BytesIO(stream.read()), allow_pickle=self.allow_pickle)
            if content_type == "application/x-npz":
                try:
                    return np.load(io.BytesIO(stream.read()), allow_pickle=self.allow_pickle)
                finally:
                    stream.close()
        finally:
            stream.close()

        raise ValueError("%s cannot read content type %s." % (__class__.__name__, content_type))


class JSONDeserializer(SimpleBaseDeserializer):
    """Deserialize JSON data from an inference endpoint into a Python object."""

    def __init__(self, accept="application/json"):
        """Initialize a ``JSONDeserializer`` instance.

        Args:
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: "application/json").
        """
        super(JSONDeserializer, self).__init__(accept=accept)

    def deserialize(self, stream, content_type):
        """Deserialize JSON data from an inference endpoint into a Python object.

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            object: The JSON-formatted data deserialized into a Python object.
        """
        try:
            return json.load(codecs.getreader("utf-8")(stream))
        finally:
            stream.close()


class PandasDeserializer(SimpleBaseDeserializer):
    """Deserialize CSV or JSON data from an inference endpoint into a pandas dataframe."""

    def __init__(self, accept=("text/csv", "application/json")):
        """Initialize a ``PandasDeserializer`` instance.

        Args:
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: ("text/csv","application/json")).
        """
        super(PandasDeserializer, self).__init__(accept=accept)

    def deserialize(self, stream, content_type):
        """Deserialize CSV or JSON data from an inference endpoint into a pandas dataframe.

        If the data is JSON, the data should be formatted in the 'columns' orient.
        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            pandas.DataFrame: The data deserialized into a pandas DataFrame.
        """
        if content_type == "text/csv":
            return pandas.read_csv(stream)

        if content_type == "application/json":
            return pandas.read_json(stream)

        raise ValueError("%s cannot read content type %s." % (__class__.__name__, content_type))


class JSONLinesDeserializer(SimpleBaseDeserializer):
    """Deserialize JSON lines data from an inference endpoint."""

    def __init__(self, accept="application/jsonlines"):
        """Initialize a ``JSONLinesDeserializer`` instance.

        Args:
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: ("text/csv","application/json")).
        """
        super(JSONLinesDeserializer, self).__init__(accept=accept)

    def deserialize(self, stream, content_type):
        """Deserialize JSON lines data from an inference endpoint.

        See https://docs.python.org/3/library/json.html#py-to-json-table to
        understand how JSON values are converted to Python objects.

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            list: A list of JSON serializable objects.
        """
        try:
            body = stream.read().decode("utf-8")
            lines = body.rstrip().split("\n")
            return [json.loads(line) for line in lines]
        finally:
            stream.close()
