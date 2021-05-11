# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
import tarfile
from urllib.parse import urlparse

import boto3


def retrieve_output_from_s3(s3_url, output_dir):
    """
    Downloads output artifacts from s3 and extracts them into the given directory.

    Args:
        s3_url: S3 URL to the output artifacts
        output_dir: directory to write artifacts to
    """
    o = urlparse(s3_url)
    s3 = boto3.resource("s3")
    output_data_path = os.path.join(output_dir)
    output_file_name = os.path.join(output_data_path, "output.tar.gz")
    try:
        os.makedirs(output_data_path)
    except FileExistsError:
        pass
    s3.Bucket(o.netloc).download_file(o.path.lstrip("/"), output_file_name)
    tar = tarfile.open(output_file_name)
    tar.extractall(output_data_path)
    tar.close()
