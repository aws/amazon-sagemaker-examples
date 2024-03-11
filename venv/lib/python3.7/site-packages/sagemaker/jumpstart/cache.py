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
"""This module defines the JumpStartModelsCache class."""
from __future__ import absolute_import
import datetime
from difflib import get_close_matches
import os
from typing import List, Optional, Tuple, Union
import json
import boto3
import botocore
from packaging.version import Version
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from sagemaker.jumpstart.constants import (
    ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE,
    ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE,
    JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY,
    JUMPSTART_DEFAULT_REGION_NAME,
    MODEL_ID_LIST_WEB_URL,
)
from sagemaker.jumpstart.parameters import (
    JUMPSTART_DEFAULT_MAX_S3_CACHE_ITEMS,
    JUMPSTART_DEFAULT_MAX_SEMANTIC_VERSION_CACHE_ITEMS,
    JUMPSTART_DEFAULT_S3_CACHE_EXPIRATION_HORIZON,
    JUMPSTART_DEFAULT_SEMANTIC_VERSION_CACHE_EXPIRATION_HORIZON,
)
from sagemaker.jumpstart.types import (
    JumpStartCachedS3ContentKey,
    JumpStartCachedS3ContentValue,
    JumpStartModelHeader,
    JumpStartModelSpecs,
    JumpStartS3FileType,
    JumpStartVersionedModelId,
)
from sagemaker.jumpstart import utils
from sagemaker.utilities.cache import LRUCache


class JumpStartModelsCache:
    """Class that implements a cache for JumpStart models manifests and specs.

    The manifest and specs associated with JumpStart models provide the information necessary
    for launching JumpStart models from the SageMaker SDK.
    """

    # fmt: off
    def __init__(
        self,
        region: str = JUMPSTART_DEFAULT_REGION_NAME,
        max_s3_cache_items: int = JUMPSTART_DEFAULT_MAX_S3_CACHE_ITEMS,
        s3_cache_expiration_horizon: datetime.timedelta =
        JUMPSTART_DEFAULT_S3_CACHE_EXPIRATION_HORIZON,
        max_semantic_version_cache_items: int =
        JUMPSTART_DEFAULT_MAX_SEMANTIC_VERSION_CACHE_ITEMS,
        semantic_version_cache_expiration_horizon: datetime.timedelta =
        JUMPSTART_DEFAULT_SEMANTIC_VERSION_CACHE_EXPIRATION_HORIZON,
        manifest_file_s3_key: str =
        JUMPSTART_DEFAULT_MANIFEST_FILE_S3_KEY,
        s3_bucket_name: Optional[str] = None,
        s3_client_config: Optional[botocore.config.Config] = None,
        s3_client: Optional[boto3.client] = None,
    ) -> None:  # fmt: on
        """Initialize a ``JumpStartModelsCache`` instance.

        Args:
            region (str): AWS region to associate with cache. Default: region associated
                with boto3 session.
            max_s3_cache_items (int): Maximum number of items to store in s3 cache.
                Default: 20.
            s3_cache_expiration_horizon (datetime.timedelta): Maximum time to hold
                items in s3 cache before invalidation. Default: 6 hours.
            max_semantic_version_cache_items (int): Maximum number of items to store in
                semantic version cache. Default: 20.
            semantic_version_cache_expiration_horizon (datetime.timedelta):
                Maximum time to hold items in semantic version cache before invalidation.
                Default: 6 hours.
            manifest_file_s3_key (str): The key in S3 corresponding to the sdk metadata manifest.
            s3_bucket_name (Optional[str]): S3 bucket to associate with cache.
                Default: JumpStart-hosted content bucket for region.
            s3_client_config (Optional[botocore.config.Config]): s3 client config to use for cache.
                Default: None (no config).
            s3_client (Optional[boto3.client]): s3 client to use. Default: None.
        """

        self._region = region
        self._s3_cache = LRUCache[JumpStartCachedS3ContentKey, JumpStartCachedS3ContentValue](
            max_cache_items=max_s3_cache_items,
            expiration_horizon=s3_cache_expiration_horizon,
            retrieval_function=self._retrieval_function,
        )
        self._model_id_semantic_version_manifest_key_cache = LRUCache[
            JumpStartVersionedModelId, JumpStartVersionedModelId
        ](
            max_cache_items=max_semantic_version_cache_items,
            expiration_horizon=semantic_version_cache_expiration_horizon,
            retrieval_function=self._get_manifest_key_from_model_id_semantic_version,
        )
        self._manifest_file_s3_key = manifest_file_s3_key
        self.s3_bucket_name = (
            utils.get_jumpstart_content_bucket(self._region)
            if s3_bucket_name is None
            else s3_bucket_name
        )
        self._s3_client = s3_client or (
            boto3.client("s3", region_name=self._region, config=s3_client_config)
            if s3_client_config
            else boto3.client("s3", region_name=self._region)
        )

    def set_region(self, region: str) -> None:
        """Set region for cache. Clears cache after new region is set."""
        if region != self._region:
            self._region = region
            self.clear()

    def get_region(self) -> str:
        """Return region for cache."""
        return self._region

    def set_manifest_file_s3_key(self, key: str) -> None:
        """Set manifest file s3 key. Clears cache after new key is set."""
        if key != self._manifest_file_s3_key:
            self._manifest_file_s3_key = key
            self.clear()

    def get_manifest_file_s3_key(self) -> str:
        """Return manifest file s3 key for cache."""
        return self._manifest_file_s3_key

    def set_s3_bucket_name(self, s3_bucket_name: str) -> None:
        """Set s3 bucket used for cache."""
        if s3_bucket_name != self.s3_bucket_name:
            self.s3_bucket_name = s3_bucket_name
            self.clear()

    def get_bucket(self) -> str:
        """Return bucket used for cache."""
        return self.s3_bucket_name

    def _get_manifest_key_from_model_id_semantic_version(
        self,
        key: JumpStartVersionedModelId,
        value: Optional[JumpStartVersionedModelId],  # pylint: disable=W0613
    ) -> JumpStartVersionedModelId:
        """Return model ID and version in manifest that matches semantic version/id.

        Uses ``packaging.version`` to perform version comparison. The highest model version
        matching the semantic version is used, which is compatible with the SageMaker
        version.

        Args:
            key (JumpStartVersionedModelId): Key for which to fetch versioned model ID.
            value (Optional[JumpStartVersionedModelId]): Unused variable for current value of
                old cached model ID/version.

        Raises:
            KeyError: If the semantic version is not found in the manifest, or is found but
                the SageMaker version needs to be upgraded in order for the model to be used.
        """

        model_id, version = key.model_id, key.version

        manifest = self._s3_cache.get(
            JumpStartCachedS3ContentKey(JumpStartS3FileType.MANIFEST, self._manifest_file_s3_key)
        ).formatted_content

        sm_version = utils.get_sagemaker_version()

        versions_compatible_with_sagemaker = [
            Version(header.version)
            for header in manifest.values()  # type: ignore
            if header.model_id == model_id and Version(header.min_version) <= Version(sm_version)
        ]

        sm_compatible_model_version = self._select_version(
            version, versions_compatible_with_sagemaker
        )

        if sm_compatible_model_version is not None:
            return JumpStartVersionedModelId(model_id, sm_compatible_model_version)

        versions_incompatible_with_sagemaker = [
            Version(header.version) for header in manifest.values()  # type: ignore
            if header.model_id == model_id
        ]
        sm_incompatible_model_version = self._select_version(
            version, versions_incompatible_with_sagemaker
        )

        if sm_incompatible_model_version is not None:
            model_version_to_use_incompatible_with_sagemaker = sm_incompatible_model_version
            sm_version_to_use_list = [
                header.min_version
                for header in manifest.values()  # type: ignore
                if header.model_id == model_id
                and header.version == model_version_to_use_incompatible_with_sagemaker
            ]
            if len(sm_version_to_use_list) != 1:
                # ``manifest`` dict should already enforce this
                raise RuntimeError("Found more than one incompatible SageMaker version to use.")
            sm_version_to_use = sm_version_to_use_list[0]

            error_msg = (
                f"Unable to find model manifest for '{model_id}' with version '{version}' "
                f"compatible with your SageMaker version ('{sm_version}'). "
                f"Consider upgrading your SageMaker library to at least version "
                f"'{sm_version_to_use}' so you can use version "
                f"'{model_version_to_use_incompatible_with_sagemaker}' of '{model_id}'."
            )
            raise KeyError(error_msg)

        error_msg = f"Unable to find model manifest for '{model_id}' with version '{version}'. "
        error_msg += (
            f"Visit {MODEL_ID_LIST_WEB_URL} for updated list of models. "
        )

        other_model_id_version = self._select_version(
            "*", versions_incompatible_with_sagemaker
        )  # all versions here are incompatible with sagemaker
        if other_model_id_version is not None:
            error_msg += (
                f"Consider using model ID '{model_id}' with version "
                f"'{other_model_id_version}'."
            )

        else:
            possible_model_ids = [header.model_id for header in manifest.values()]  # type: ignore
            closest_model_id = get_close_matches(model_id, possible_model_ids, n=1, cutoff=0)[0]
            error_msg += f"Did you mean to use model ID '{closest_model_id}'?"

        raise KeyError(error_msg)

    def _get_json_file_and_etag_from_s3(self, key: str) -> Tuple[Union[dict, list], str]:
        """Returns json file from s3, along with its etag."""
        response = self._s3_client.get_object(Bucket=self.s3_bucket_name, Key=key)
        return json.loads(response["Body"].read().decode("utf-8")), response["ETag"]

    def _is_local_metadata_mode(self) -> bool:
        """Returns True if the cache should use local metadata mode, based off env variables."""
        return (ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE in os.environ
                and os.path.isdir(os.environ[ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE])
                and ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE in os.environ
                and os.path.isdir(os.environ[ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE]))

    def _get_json_file(
        self,
        key: str,
        filetype: JumpStartS3FileType
    ) -> Tuple[Union[dict, list], Optional[str]]:
        """Returns json file either from s3 or local file system.

        Returns etag along with json object for s3, or just the json
        object and None when reading from the local file system.
        """
        if self._is_local_metadata_mode():
            file_content, etag = self._get_json_file_from_local_override(key, filetype), None
        else:
            file_content, etag = self._get_json_file_and_etag_from_s3(key)
        return file_content, etag

    def _get_json_md5_hash(self, key: str):
        """Retrieves md5 object hash for s3 objects, using `s3.head_object`.

        Raises:
            ValueError: if the cache should use local metadata mode.
        """
        if self._is_local_metadata_mode():
            raise ValueError("Cannot get md5 hash of local file.")
        return self._s3_client.head_object(Bucket=self.s3_bucket_name, Key=key)["ETag"]

    def _get_json_file_from_local_override(
        self,
        key: str,
        filetype: JumpStartS3FileType
    ) -> Union[dict, list]:
        """Reads json file from local filesystem and returns data."""
        if filetype == JumpStartS3FileType.MANIFEST:
            metadata_local_root = (
                os.environ[ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE]
            )
        elif filetype == JumpStartS3FileType.SPECS:
            metadata_local_root = os.environ[ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE]
        else:
            raise ValueError(f"Unsupported file type for local override: {filetype}")
        file_path = os.path.join(metadata_local_root, key)
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def _retrieval_function(
        self,
        key: JumpStartCachedS3ContentKey,
        value: Optional[JumpStartCachedS3ContentValue],
    ) -> JumpStartCachedS3ContentValue:
        """Return s3 content given a file type and s3_key in ``JumpStartCachedS3ContentKey``.

        If a manifest file is being fetched, we only download the object if the md5 hash in
        ``head_object`` does not match the current md5 hash for the stored value. This prevents
        unnecessarily downloading the full manifest when it hasn't changed.

        Args:
            key (JumpStartCachedS3ContentKey): key for which to fetch s3 content.
            value (Optional[JumpStartVersionedModelId]): Current value of old cached
                s3 content. This is used for the manifest file, so that it is only
                downloaded when its content changes.
        """

        file_type, s3_key = key.file_type, key.s3_key

        if file_type == JumpStartS3FileType.MANIFEST:
            if value is not None and not self._is_local_metadata_mode():
                etag = self._get_json_md5_hash(s3_key)
                if etag == value.md5_hash:
                    return value
            formatted_body, etag = self._get_json_file(s3_key, file_type)
            return JumpStartCachedS3ContentValue(
                formatted_content=utils.get_formatted_manifest(formatted_body),
                md5_hash=etag,
            )
        if file_type == JumpStartS3FileType.SPECS:
            formatted_body, _ = self._get_json_file(s3_key, file_type)
            model_specs = JumpStartModelSpecs(formatted_body)
            utils.emit_logs_based_on_model_specs(model_specs, self.get_region())
            return JumpStartCachedS3ContentValue(
                formatted_content=model_specs
            )
        raise ValueError(
            f"Bad value for key '{key}': must be in {[JumpStartS3FileType.MANIFEST, JumpStartS3FileType.SPECS]}"
        )

    def get_manifest(self) -> List[JumpStartModelHeader]:
        """Return entire JumpStart models manifest."""

        manifest_dict = self._s3_cache.get(
            JumpStartCachedS3ContentKey(JumpStartS3FileType.MANIFEST, self._manifest_file_s3_key)
        ).formatted_content
        manifest = list(manifest_dict.values())  # type: ignore
        return manifest

    def get_header(self, model_id: str, semantic_version_str: str) -> JumpStartModelHeader:
        """Return header for a given JumpStart model ID and semantic version.

        Args:
            model_id (str): model ID for which to get a header.
            semantic_version_str (str): The semantic version for which to get a
                header.
        """

        return self._get_header_impl(model_id, semantic_version_str=semantic_version_str)

    def _select_version(
        self,
        semantic_version_str: str,
        available_versions: List[Version],
    ) -> Optional[str]:
        """Perform semantic version search on available versions.

        Args:
            semantic_version_str (str): the semantic version for which to filter
                available versions.
            available_versions (List[Version]): list of available versions.
        """
        if semantic_version_str == "*":
            if len(available_versions) == 0:
                return None
            return str(max(available_versions))

        try:
            spec = SpecifierSet(f"=={semantic_version_str}")
        except InvalidSpecifier:
            raise KeyError(f"Bad semantic version: {semantic_version_str}")
        available_versions_filtered = list(spec.filter(available_versions))
        return (
            str(max(available_versions_filtered)) if available_versions_filtered != [] else None
        )

    def _get_header_impl(
        self,
        model_id: str,
        semantic_version_str: str,
        attempt: int = 0,
    ) -> JumpStartModelHeader:
        """Lower-level function to return header.

        Allows a single retry if the cache is old.

        Args:
            model_id (str): model ID for which to get a header.
            semantic_version_str (str): The semantic version for which to get a
                header.
            attempt (int): attempt number at retrieving a header.
        """

        versioned_model_id = self._model_id_semantic_version_manifest_key_cache.get(
            JumpStartVersionedModelId(model_id, semantic_version_str)
        )
        manifest = self._s3_cache.get(
            JumpStartCachedS3ContentKey(JumpStartS3FileType.MANIFEST, self._manifest_file_s3_key)
        ).formatted_content
        try:
            header = manifest[versioned_model_id]  # type: ignore
            return header
        except KeyError:
            if attempt > 0:
                raise
            self.clear()
            return self._get_header_impl(model_id, semantic_version_str, attempt + 1)

    def get_specs(self, model_id: str, semantic_version_str: str) -> JumpStartModelSpecs:
        """Return specs for a given JumpStart model ID and semantic version.

        Args:
            model_id (str): model ID for which to get specs.
            semantic_version_str (str): The semantic version for which to get
                specs.
        """

        header = self.get_header(model_id, semantic_version_str)
        spec_key = header.spec_key
        specs = self._s3_cache.get(
            JumpStartCachedS3ContentKey(JumpStartS3FileType.SPECS, spec_key)
        ).formatted_content
        return specs  # type: ignore

    def clear(self) -> None:
        """Clears the model ID/version and s3 cache."""
        self._s3_cache.clear()
        self._model_id_semantic_version_manifest_key_cache.clear()
