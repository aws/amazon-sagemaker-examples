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

import os
import platform
import sys
import tempfile
from abc import ABCMeta
from abc import abstractmethod
from six import with_metaclass

from six.moves.urllib.parse import urlparse

import sagemaker.amazon.common
import sagemaker.local.utils
import sagemaker.utils


def get_data_source_instance(data_source, sagemaker_session):
    """Return an Instance of :class:`sagemaker.local.data.DataSource`.

    The instance can handle the provided data_source URI.

    data_source can be either file:// or s3://

    Args:
        data_source (str): a valid URI that points to a data source.
        sagemaker_session (:class:`sagemaker.session.Session`): a SageMaker Session to
            interact with S3 if required.

    Returns:
        sagemaker.local.data.DataSource: an Instance of a Data Source

    Raises:
        ValueError: If parsed_uri scheme is neither `file` nor `s3` , raise an
            error.
    """
    parsed_uri = urlparse(data_source)
    if parsed_uri.scheme == "file":
        return LocalFileDataSource(parsed_uri.netloc + parsed_uri.path)
    if parsed_uri.scheme == "s3":
        return S3DataSource(parsed_uri.netloc, parsed_uri.path, sagemaker_session)
    raise ValueError(
        "data_source must be either file or s3. parsed_uri.scheme: {}".format(parsed_uri.scheme)
    )


def get_splitter_instance(split_type):
    """Return an Instance of :class:`sagemaker.local.data.Splitter`.

    The instance returned is according to the specified `split_type`.

    Args:
        split_type (str): either 'Line' or 'RecordIO'. Can be left as None to
            signal no data split will happen.

    Returns
        :class:`sagemaker.local.data.Splitter`: an Instance of a Splitter
    """
    if split_type is None:
        return NoneSplitter()
    if split_type == "Line":
        return LineSplitter()
    if split_type == "RecordIO":
        return RecordIOSplitter()
    raise ValueError("Invalid Split Type: %s" % split_type)


def get_batch_strategy_instance(strategy, splitter):
    """Return an Instance of :class:`sagemaker.local.data.BatchStrategy` according to `strategy`

    Args:
        strategy (str): Either 'SingleRecord' or 'MultiRecord'
        splitter (:class:`sagemaker.local.data.Splitter): splitter to get the data from.

    Returns
        :class:`sagemaker.local.data.BatchStrategy`: an Instance of a BatchStrategy
    """
    if strategy == "SingleRecord":
        return SingleRecordStrategy(splitter)
    if strategy == "MultiRecord":
        return MultiRecordStrategy(splitter)
    raise ValueError('Invalid Batch Strategy: %s - Valid Strategies: "SingleRecord", "MultiRecord"')


class DataSource(with_metaclass(ABCMeta, object)):
    """Placeholder docstring"""

    @abstractmethod
    def get_file_list(self):
        """Retrieve the list of absolute paths to all the files in this data source.

        Returns:
            List[str]: List of absolute paths.
        """

    @abstractmethod
    def get_root_dir(self):
        """Retrieve the absolute path to the root directory of this data source.

        Returns:
            str: absolute path to the root directory of this data source.
        """


class LocalFileDataSource(DataSource):
    """Represents a data source within the local filesystem."""

    def __init__(self, root_path):
        super(LocalFileDataSource, self).__init__()

        self.root_path = os.path.abspath(root_path)
        if not os.path.exists(self.root_path):
            raise RuntimeError("Invalid data source: %s does not exist." % self.root_path)

    def get_file_list(self):
        """Retrieve the list of absolute paths to all the files in this data source.

        Returns:
            List[str] List of absolute paths.
        """
        if os.path.isdir(self.root_path):
            return [
                os.path.join(self.root_path, f)
                for f in os.listdir(self.root_path)
                if os.path.isfile(os.path.join(self.root_path, f))
            ]
        return [self.root_path]

    def get_root_dir(self):
        """Retrieve the absolute path to the root directory of this data source.

        Returns:
            str: absolute path to the root directory of this data source.
        """
        if os.path.isdir(self.root_path):
            return self.root_path
        return os.path.dirname(self.root_path)


class S3DataSource(DataSource):
    """Defines a data source given by a bucket and S3 prefix.

    The contents will be downloaded and then processed as local data.
    """

    def __init__(self, bucket, prefix, sagemaker_session):
        """Create an S3DataSource instance.

        Args:
            bucket (str): S3 bucket name
            prefix (str): S3 prefix path to the data
            sagemaker_session (:class:`sagemaker.session.Session`): a sagemaker_session with the
            desired settings
                to talk to S3
        """
        super(S3DataSource, self).__init__()

        # Create a temporary dir to store the S3 contents
        root_dir = sagemaker.utils.get_config_value(
            "local.container_root", sagemaker_session.config
        )
        if root_dir:
            root_dir = os.path.abspath(root_dir)

        working_dir = tempfile.mkdtemp(dir=root_dir)
        # Docker cannot mount Mac OS /var folder properly see
        # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
        # Only apply this workaround if the user didn't provide an alternate storage root dir.
        if root_dir is None and platform.system() == "Darwin":
            working_dir = "/private{}".format(working_dir)

        sagemaker.utils.download_folder(bucket, prefix, working_dir, sagemaker_session)
        self.files = LocalFileDataSource(working_dir)

    def get_file_list(self):
        """Retrieve the list of absolute paths to all the files in this data source.

        Returns:
            List[str]: List of absolute paths.
        """
        return self.files.get_file_list()

    def get_root_dir(self):
        """Retrieve the absolute path to the root directory of this data source.

        Returns:
            str: absolute path to the root directory of this data source.
        """
        return self.files.get_root_dir()


class Splitter(with_metaclass(ABCMeta, object)):
    """Placeholder docstring"""

    @abstractmethod
    def split(self, file):
        """Split a file into records using a specific strategy

        Args:
            file (str): path to the file to split

        Returns:
            generator for the individual records that were split from the file
        """


class NoneSplitter(Splitter):
    """Does not split records, essentially reads the whole file."""

    # non-utf8 characters.
    _textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})

    def split(self, filename):
        """Split a file into records using a specific strategy.

        For this NoneSplitter there is no actual split happening and the file
        is returned as a whole.

        Args:
            filename (str): path to the file to split

        Returns: generator for the individual records that were split from
            the file
        """
        with open(filename, "rb") as f:
            buf = f.read()
            if not self._is_binary(buf):
                buf = buf.decode()
            yield buf

    def _is_binary(self, buf):
        """Check whether `buf` contains binary data.

        Returns True if `buf` contains any non-utf-8 characters.

        Args:
            buf (bytes): data to inspect

        Returns:
            True if data is binary, otherwise False
        """
        return bool(buf.translate(None, self._textchars))


class LineSplitter(Splitter):
    """Split records by new line."""

    def split(self, file):
        """Split a file into records using a specific strategy

        This LineSplitter splits the file on each line break.

        Args:
            file (str): path to the file to split

        Returns: generator for the individual records that were split from
        the file
        """
        with open(file, "r") as f:
            for line in f:
                yield line


class RecordIOSplitter(Splitter):
    """Split using Amazon Recordio.

    Not useful for string content.
    """

    def split(self, file):
        """Split a file into records using a specific strategy

        This RecordIOSplitter splits the data into individual RecordIO
        records.

        Args:
            file (str): path to the file to split

        Returns: generator for the individual records that were split from
        the file
        """
        with open(file, "rb") as f:
            for record in sagemaker.amazon.common.read_recordio(f):
                yield record


class BatchStrategy(with_metaclass(ABCMeta, object)):
    """Placeholder docstring"""

    def __init__(self, splitter):
        """Create a Batch Strategy Instance

        Args:
            splitter (sagemaker.local.data.Splitter): A Splitter to pre-process
                the data before batching.
        """
        self.splitter = splitter

    @abstractmethod
    def pad(self, file, size):
        """Group together as many records as possible to fit in the specified size.

        Args:
            file (str): file path to read the records from.
            size (int): maximum size in MB that each group of records will be
                fitted to. passing 0 means unlimited size.

        Returns:
            generator of records
        """


class MultiRecordStrategy(BatchStrategy):
    """Feed multiple records at a time for batch inference.

    Will group up as many records as possible within the payload specified.
    """

    def pad(self, file, size=6):
        """Group together as many records as possible to fit in the specified size.

        Args:
            file (str): file path to read the records from.
            size (int): maximum size in MB that each group of records will be
                fitted to. passing 0 means unlimited size.

        Returns:
            generator of records
        """
        buffer = ""
        for element in self.splitter.split(file):
            if _payload_size_within_limit(buffer + element, size):
                buffer += element
            else:
                tmp = buffer
                buffer = element
                yield tmp
        if _validate_payload_size(buffer, size):
            yield buffer


class SingleRecordStrategy(BatchStrategy):
    """Feed a single record at a time for batch inference.

    If a single record does not fit within the payload specified it will
    throw a RuntimeError.
    """

    def pad(self, file, size=6):
        """Group together as many records as possible to fit in the specified size.

        This SingleRecordStrategy will not group any record and will return
        them one by one as long as they are within the maximum size.

        Args:
            file (str): file path to read the records from.
            size (int): maximum size in MB that each group of records will be
                fitted to. passing 0 means unlimited size.

        Returns:
            generator of records
        """
        for element in self.splitter.split(file):
            if _validate_payload_size(element, size):
                yield element


def _payload_size_within_limit(payload, size):
    """Placeholder docstring."""
    size_in_bytes = size * 1024 * 1024
    if size == 0:
        return True
    return sys.getsizeof(payload) < size_in_bytes


def _validate_payload_size(payload, size):
    """Check if a payload is within the size in MB threshold.

    Raise an exception if the payload is beyond the size in MB threshold.

    Args:
        payload: data that will be checked
        size (int): max size in MB

    Returns:
        bool: True if within bounds. if size=0 it will always return True

    Raises:
        RuntimeError: If the payload is larger a runtime error is thrown.
    """

    if _payload_size_within_limit(payload, size):
        return True
    raise RuntimeError("Record is larger than %sMB. Please increase your max_payload" % size)
