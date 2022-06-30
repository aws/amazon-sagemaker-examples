# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""A CLI to get pipeline definitions from pipeline modules."""
from __future__ import absolute_import

import argparse
import sys

from pipelines._utils import get_pipeline_driver


def main():  # pragma: no cover
    """The main harness that gets the pipeline definition JSON.

    Prints the json to stdout or saves to file.
    """
    parser = argparse.ArgumentParser("Gets the pipeline definition for the pipeline script.")

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import.",
    )
    parser.add_argument(
        "-f",
        "--file-name",
        dest="file_name",
        type=str,
        default=None,
        help="The file to output the pipeline definition json to.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    args = parser.parse_args()

    if args.module_name is None:
        parser.print_help()
        sys.exit(2)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        content = pipeline.definition()
        if args.file_name:
            with open(args.file_name, "w") as f:
                f.write(content)
        else:
            print(content)
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
