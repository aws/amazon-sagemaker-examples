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
"""A tool to upgrade SageMaker Python SDK code to be compatible with version 2.0 and later."""
from __future__ import absolute_import

import argparse
import os

from sagemaker.cli.compatibility.v2 import files

_EXT_TO_UPDATER_CLS = {".py": files.PyFileUpdater, ".ipynb": files.JupyterNotebookFileUpdater}


def _update_file(input_file, output_file):
    """Updates a file to be compatible with version 2.0 and later of the SageMaker Python SDK.

    It also writes the updated source to the output file.

    Args:
        input_file (str): The path to the file to be updated.
        output_file (str): The output file destination.

    Raises:
        ValueError: If the input and output filename extensions don't match,
            or if the file extensions are neither ".py" nor ".ipynb".
    """
    input_file_ext = os.path.splitext(input_file)[1]
    output_file_ext = os.path.splitext(output_file)[1]

    if input_file_ext != output_file_ext:
        raise ValueError(
            "Mismatched file extensions: input: {}, output: {}".format(
                input_file_ext, output_file_ext
            )
        )

    if input_file_ext not in _EXT_TO_UPDATER_CLS:
        raise ValueError("Unrecognized file extension: {}".format(input_file_ext))

    updater_cls = _EXT_TO_UPDATER_CLS[input_file_ext]
    updater_cls(input_path=input_file, output_path=output_file).update()


def _parse_args():
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="A tool to convert files to be compatible with "
        "version 2.0 and later of the SageMaker Python SDK. "
        "Simple usage: sagemaker-upgrade-v2 --in-file foo.py --out-file bar.py"
    )
    parser.add_argument(
        "--in-file",
        help="If converting a single file, the file to convert. The file's extension "
        "must be either '.py' or '.ipynb'.",
    )
    parser.add_argument(
        "--out-file",
        help="If converting a single file, the output file destination. The file's extension "
        "must be either '.py' or '.ipynb'. If needed, directories in the output path are created. "
        "If the output file already exists, it is overwritten.",
    )

    return parser.parse_args()


def main():
    """Parses the CLI arguments and executes the file update."""
    args = _parse_args()
    _update_file(args.in_file, args.out_file)
