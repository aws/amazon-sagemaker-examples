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
"""A Python script to upgrade framework versions and regions"""
from __future__ import absolute_import

import argparse
import json
import os

from sagemaker.image_uris import config_for_framework

IMAGE_URI_CONFIG_DIR = os.path.join("..", "image_uri_config")


def get_latest_values(existing_content, scope=None):
    """Get the latest "registries", "py_versions" and "repository" values

    Args:
        existing_content (dict): Dictionary of complete framework image information.
        scope (str): Type of the image, required if the target is DLC
            framework (Default: None).
    """
    if scope in existing_content:
        existing_content = existing_content[scope]
    else:
        if "versions" not in existing_content:
            raise ValueError(
                "Invalid image scope: {}. Valid options: {}.".format(
                    scope, ", ".join(existing_content.key())
                )
            )

    latest_version = list(existing_content["versions"].keys())[-1]
    registries = existing_content["versions"][latest_version].get("registries", None)
    py_versions = existing_content["versions"][latest_version].get("py_versions", None)
    repository = existing_content["versions"][latest_version].get("repository", None)

    return registries, py_versions, repository


def _write_dict_to_json(filename, existing_content):
    """Write a Python dictionary to a json file.

    Args:
        filename (str): Name of the target json file.
        existing_content (dict): Dictionary to be written to the json file.
    """
    with open(filename, "w") as f:
        json.dump(existing_content, f, sort_keys=True, indent=4)


def add_dlc_framework_version(
    existing_content,
    short_version,
    full_version,
    scope,
    processors,
    py_versions,
    registries,
    repository,
):
    """Update DLC framework image uri json file with new version information.

    Args:
        existing_content (dict): Existing framework image uri information read from
            "<framework>.json" file.
        framework (str): Framework name (e.g. tensorflow, pytorch, mxnet)
        short_version (str): Abbreviated framework version (e.g. 1.0, 1.5)
        full_version (str): Complete framework version (e.g. 1.0.0, 1.5.2)
        scope (str): Framework image type, it could be "training", "inference"
            or "eia"
        processors (list): Supported processors (e.g. ["cpu", "gpu"])
        py_versions (list): Supported Python versions (e.g. ["py3", "py37"])
        registries (dict): Framework image's region to account mapping.
        repository (str): Framework image's ECR repository.
    """
    for processor in processors:
        if processor not in existing_content[scope]["processors"]:
            existing_content[scope]["processors"].append(processor)
    existing_content[scope]["version_aliases"][short_version] = full_version

    new_version = {
        "registries": registries,
        "repository": repository,
    }
    if py_versions:
        new_version["py_versions"] = py_versions
    existing_content[scope]["versions"][full_version] = new_version


def add_algo_version(
    existing_content,
    processors,
    scopes,
    full_version,
    py_versions,
    registries,
    repository,
    tag_prefix,
):
    """Update Algorithm image uri json file with new version information.

    Args:
        existing_content (dict): Existing algorithm image uri information read from
            "<algorithm>.json" file.
        processors (list): Supported processors (e.g. ["cpu", "gpu"])
        scopes (list): Framework image type, it could be "training", "inference
        full_version (str): Complete framework version (e.g. 1.0.0, 1.5.2)
        py_versions (list): Supported Python versions (e.g. ["py3", "py37"])
        registries (dict): Algorithm image's region to account mapping.
        repository (str): Algorithm's corresponding repository name.
        tag_prefix (str): Algorithm image's tag prefix.
    """
    for processor in processors:
        if processor not in existing_content["processors"]:
            existing_content["processors"].append(processor)
    for scope in scopes:
        if scope not in existing_content["scope"]:
            existing_content["scope"].append(scope)

    new_version = {
        "registries": registries,
        "repository": repository,
    }
    if py_versions:
        new_version["py_versions"] = py_versions
    if tag_prefix:
        new_version["tag_prefix"] = tag_prefix
    existing_content["versions"][full_version] = new_version


def add_region(existing_content, region, account):
    """Add region account to framework/algorithm registries.

    Args:
        existing_content (dict): Existing framework/algorithm image uri information read from
            json file.
        region (str): New region to be added to framework/algorithm registries (e.g. us-west-2).
        account (str): Region registry account number.
    """
    if "scope" not in existing_content:
        for scope in existing_content:
            for version in existing_content[scope]["versions"]:
                existing_content[scope]["versions"][version]["registries"][region] = account
    else:
        for version in existing_content["versions"]:
            existing_content["versions"][version]["registries"][region] = account


def add_version(
    existing_content,
    short_version,
    full_version,
    scope,
    processors,
    py_versions,
    tag_prefix,
):
    """Read, update and write framework image uri.

    Read framework image uri information from json file to a dictionary, update it with new
    framework version information, then write the dictionary back to json file.

    Args:
        existing_content (dict): Existing framework/algorithm image uri information read from
            json file.
        short_version (str): Abbreviated framework version (e.g. 1.0, 1.5).
        full_version (str): Complete framework version (e.g. 1.0.0, 1.5.2).
        scope (str): Framework image type, it could be "training", "inference".
        processors (str): Supported processors (e.g. "cpu,gpu").
        py_versions (str): Supported Python versions (e.g. "py3,py37").
        tag_prefix (str): Algorithm image's tag prefix.
    """
    if py_versions:
        py_versions = py_versions.split(",")
    processors = processors.split(",")
    latest_registries, latest_py_versions, latest_repository = get_latest_values(
        existing_content, scope
    )
    if not py_versions:
        py_versions = latest_py_versions

    if scope in existing_content:
        add_dlc_framework_version(
            existing_content,
            short_version,
            full_version,
            scope,
            processors,
            py_versions,
            latest_registries,
            latest_repository,
        )
    else:
        scopes = scope.split(",")
        add_algo_version(
            existing_content,
            processors,
            scopes,
            full_version,
            py_versions,
            latest_registries,
            latest_repository,
            tag_prefix,
        )


def main():
    """Parse command line arguments, call corresponding methods."""
    parser = argparse.ArgumentParser(description="Framework upgrade tool.")
    parser.add_argument(
        "--framework", required=True, help="Name of the framework (e.g. tensorflow, mxnet, etc.)"
    )
    parser.add_argument("--short-version", help="Abbreviated framework version (e.g. 2.0)")
    parser.add_argument("--full-version", help="Full framework version (e.g. 2.0.1)")
    parser.add_argument("--processors", help="Suppoted processors (e.g. cpu, gpu)")
    parser.add_argument("--py-versions", help="Supported Python versions (e.g. py3,py37)")
    parser.add_argument("--scope", help="Scope for the Algorithm image (e.g. inference, training)")
    parser.add_argument(
        "--tag-prefix", help="Tag prefix of the Algorithm image (e.g. ray-0.8.5-torch)"
    )
    parser.add_argument("--region", help="New region to be added (e.g. us-west-2)")
    parser.add_argument("--account", help="Registry account of new region")

    args = parser.parse_args()

    content = config_for_framework(args.framework)

    if args.region or args.account:
        if args.region and not args.account or args.account and not args.region:
            raise ValueError("--region and --account must be used together to expand region.")
        add_region(content, args.region, args.account)
    else:
        add_version(
            content,
            args.short_version,
            args.full_version,
            args.scope,
            args.processors,
            args.py_versions,
            args.tag_prefix,
        )

    file = os.path.join(IMAGE_URI_CONFIG_DIR, "{}.json".format(args.framework))
    _write_dict_to_json(file, content)


if __name__ == "__main__":
    main()
