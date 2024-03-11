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
"""Contains the helper classes for SageMaker Experiment."""
from __future__ import absolute_import

import json
import logging
import os

import botocore

from sagemaker import s3
from sagemaker.experiments._utils import is_already_exist_error

logger = logging.getLogger(__name__)


_DEFAULT_ARTIFACT_PREFIX = "trial-component-artifacts"
_DEFAULT_ARTIFACT_TYPE = "Tracker"


class _ArtifactUploader(object):
    """Artifact uploader"""

    def __init__(
        self,
        trial_component_name,
        sagemaker_session,
        artifact_bucket=None,
        artifact_prefix=_DEFAULT_ARTIFACT_PREFIX,
    ):
        """Initialize a `_ArtifactUploader` instance.

        Args:
            trial_component_name (str): The name of the trial component,
                which is used to generate the S3 path to upload the artifact to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed.
            artifact_bucket (str): The S3 bucket to upload the artifact to.
                If not specified, the default bucket defined in `sagemaker_session`
                will be used.
            artifact_prefix (str): The S3 key prefix used to generate the S3 path
                to upload the artifact to (default: "trial-component-artifacts").
        """
        self.sagemaker_session = sagemaker_session
        self.trial_component_name = trial_component_name
        self.artifact_bucket = artifact_bucket
        self.artifact_prefix = artifact_prefix
        self._s3_client = self.sagemaker_session.boto_session.client("s3")

    def upload_artifact(self, file_path):
        """Upload an artifact file to S3.

        Args:
            file_path (str): the file path of the artifact

        Returns:
            (str, str): The s3 URI of the uploaded file and the etag of the file.

        Raises:
            ValueError: If file does not exist.
        """
        file_path = os.path.expanduser(file_path)
        if not os.path.isfile(file_path):
            raise ValueError(
                "{} does not exist or is not a file. Please supply a file path.".format(file_path)
            )

        # If self.artifact_bucket is falsy, it will be set to sagemaker_session.default_bucket.
        # In that case, and if sagemaker_session.default_bucket_prefix exists, self.artifact_prefix
        # needs to be updated too (because not updating self.artifact_prefix would result in
        # different behavior the 1st time this method is called vs the 2nd).
        self.artifact_bucket, self.artifact_prefix = s3.determine_bucket_and_prefix(
            bucket=self.artifact_bucket,
            key_prefix=self.artifact_prefix,
            sagemaker_session=self.sagemaker_session,
        )

        artifact_name = os.path.basename(file_path)
        artifact_s3_key = "{}/{}/{}".format(
            self.artifact_prefix, self.trial_component_name, artifact_name
        )
        self._s3_client.upload_file(file_path, self.artifact_bucket, artifact_s3_key)
        etag = self._try_get_etag(artifact_s3_key)
        return "s3://{}/{}".format(self.artifact_bucket, artifact_s3_key), etag

    def upload_object_artifact(self, artifact_name, artifact_object, file_extension=None):
        """Upload an artifact object to S3.

        Args:
            artifact_name (str): the name of the artifact.
            artifact_object (obj): the object of the artifact
            file_extension (str): Optional file extension.

        Returns:
            str: The s3 URI of the uploaded file and the version of the file.
        """

        # If self.artifact_bucket is falsy, it will be set to sagemaker_session.default_bucket.
        # In that case, and if sagemaker_session.default_bucket_prefix exists, self.artifact_prefix
        # needs to be updated too (because not updating self.artifact_prefix would result in
        # different behavior the 1st time this method is called vs the 2nd).
        self.artifact_bucket, self.artifact_prefix = s3.determine_bucket_and_prefix(
            bucket=self.artifact_bucket,
            key_prefix=self.artifact_prefix,
            sagemaker_session=self.sagemaker_session,
        )

        if file_extension:
            artifact_name = (
                artifact_name + ("" if file_extension.startswith(".") else ".") + file_extension
            )
        artifact_s3_key = "{}/{}/{}".format(
            self.artifact_prefix, self.trial_component_name, artifact_name
        )
        self._s3_client.put_object(
            Body=json.dumps(artifact_object), Bucket=self.artifact_bucket, Key=artifact_s3_key
        )
        etag = self._try_get_etag(artifact_s3_key)
        return "s3://{}/{}".format(self.artifact_bucket, artifact_s3_key), etag

    def _try_get_etag(self, key):
        """Get ETag of given key and return None if not allowed

        Args:
            key (str): The S3 object key.

        Returns:
            str: The S3 object ETag if it allows, otherwise return None.
        """
        try:
            response = self._s3_client.head_object(Bucket=self.artifact_bucket, Key=key)
            return response["ETag"]
        except botocore.exceptions.ClientError as error:
            # requires read permissions
            logger.warning("Failed to get ETag of %s due to %s", key, error)
        return None


class _LineageArtifactManager(object):
    """A helper class to manage Lineage Artifacts"""

    def __init__(
        self,
        name,
        source_uri,
        etag,
        source_arn=None,
        dest_arn=None,
        artifact_type=_DEFAULT_ARTIFACT_TYPE,
    ):
        """Initialize a `_LineageArtifactManager` instance.

        Args:
            name (str): The name of the Lineage artifact to be created.
            source_uri (str): The source URI used to create the Lineage artifact.
            etag (str): The S3 Etag used to create the Lineage artifact.
            source_arn (str): The source ARN of a trail component to associate
                this Lineage artifact with (default: None).
            dest_arn (str): The destination ARN of a trial component to associate
                this Lineage artifact with (default: None).
            artifact_type (str): The type of the Lineage artifact (default: "Tracker").
        """
        self.name = name
        self.source_uri = source_uri
        self.etag = etag
        self.source_arn = source_arn
        self.dest_arn = dest_arn
        self.artifact_arn = None
        self.artifact_type = artifact_type

    def create_artifact(self, sagemaker_session):
        """Create the artifact by calling `CreateArtifact` API

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed.
        """
        source_ids = []
        if self.etag:
            source_ids.append({"SourceIdType": "S3ETag", "Value": self.etag})

        try:
            response = sagemaker_session.sagemaker_client.create_artifact(
                ArtifactName=self.name,
                ArtifactType=self.artifact_type,
                Source={"SourceUri": self.source_uri, "SourceTypes": source_ids},
            )
            self.artifact_arn = response["ArtifactArn"]
        except botocore.exceptions.ClientError as err:
            err_info = err.response["Error"]
            if not is_already_exist_error(err_info):
                raise
            logger.warning(
                "Skip creating the artifact since it already exists: %s", err_info["Message"]
            )

    def add_association(self, sagemaker_session):
        """Associate the artifact with a source/destination ARN (e.g. trial component arn)

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed.
        """
        source_arn = self.source_arn if self.source_arn else self.artifact_arn
        dest_arn = self.dest_arn if self.dest_arn else self.artifact_arn
        # if the trial component (job) is the source then it produced the artifact,
        # otherwise the artifact contributed to the trial component (job)
        association_edge_type = "Produced" if self.source_arn else "ContributedTo"
        try:
            sagemaker_session.sagemaker_client.add_association(
                SourceArn=source_arn, DestinationArn=dest_arn, AssociationType=association_edge_type
            )
        except botocore.exceptions.ClientError as err:
            err_info = err.response["Error"]
            if not is_already_exist_error(err_info):
                raise
            logger.warning(
                "Skip associating since the association already exists: %s", err_info["Message"]
            )


class _LineageArtifactTracker(object):
    """Lineage Artifact Tracker"""

    def __init__(self, trial_component_arn, sagemaker_session):
        """Initialize a `_LineageArtifactTracker` instance.

        Args:
            trial_component_arn (str): The ARN of the trial component to be
                associated with the input/output artifacts.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed.
        """
        self.trial_component_arn = trial_component_arn
        self.sagemaker_session = sagemaker_session
        self.artifacts = []

    def add_input_artifact(self, name, source_uri, etag, artifact_type):
        """Add a Lineage input artifact locally

        Args:
            name (str): The name of the Lineage input artifact to be added.
            source_uri (str): The source URI used to create the Lineage input artifact.
            etag (str): The S3 Etag used to create the Lineage input artifact.
            artifact_type (str): The type of the Lineage input artifact.
        """
        artifact = _LineageArtifactManager(
            name, source_uri, etag, dest_arn=self.trial_component_arn, artifact_type=artifact_type
        )
        self.artifacts.append(artifact)

    def add_output_artifact(self, name, source_uri, etag, artifact_type):
        """Add a Lineage output artifact locally

        Args:
            name (str): The name of the Lineage output artifact to be added.
            source_uri (str): The source URI used to create the Lineage output artifact.
            etag (str): The S3 Etag used to create the Lineage output artifact.
            artifact_type (str): The type of the Lineage output artifact.
        """
        artifact = _LineageArtifactManager(
            name, source_uri, etag, source_arn=self.trial_component_arn, artifact_type=artifact_type
        )
        self.artifacts.append(artifact)

    def save(self):
        """Persist any artifact data saved locally"""
        for artifact in self.artifacts:
            artifact.create_artifact(self.sagemaker_session)
            artifact.add_association(self.sagemaker_session)
