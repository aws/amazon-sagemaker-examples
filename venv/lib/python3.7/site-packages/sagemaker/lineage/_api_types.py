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
"""Contains API objects for SageMaker Lineage."""
from __future__ import absolute_import

from sagemaker.apiutils import _base_types


class ArtifactSource(_base_types.ApiObject):
    """ArtifactSource.

    Attributes:
        source_uri (str): The URI of the source.
        source_types(list[ArtifactSourceType]): List of source types
    """

    source_uri = None
    source_types = None

    def __init__(self, source_uri=None, source_types=None, **kwargs):
        """Initialize ArtifactSource.

        Args:
            source_uri (str): Source S3 URI of the artifact.
            source_types (array): Array of artifact source types.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ArtifactSource, self).__init__(
            source_uri=source_uri, source_types=source_types, **kwargs
        )


class ArtifactSourceType(_base_types.ApiObject):
    """ArtifactSourceType.

    Attributes:
        source_id_type (str): The source id type of artifact source.
        value(str): The value of source
    """

    source_id_type = None
    value = None

    def __init__(self, source_id_type=None, value=None, **kwargs):
        """Initialize ArtifactSourceType.

        Args:
            source_id_type (str): The type of the source id.
            value (str): The source id.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ArtifactSourceType, self).__init__(
            source_id_type=source_id_type, value=value, **kwargs
        )


class ActionSource(_base_types.ApiObject):
    """ActionSource.

    Attributes:
        source_uri (str): The URI of the source.
        source_type (str):  The type of the source URI.
    """

    source_uri = None
    source_type = None

    def __init__(self, source_uri=None, source_type=None, **kwargs):
        """Initialize ActionSource.

        Args:
            source_uri (str): The URI of the source.
            source_type (str): The type of the source.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ActionSource, self).__init__(source_uri=source_uri, source_type=source_type, **kwargs)


class ContextSource(_base_types.ApiObject):
    """ContextSource.

    Attributes:
        source_uri (str): The URI of the source.
        source_type (str): The type of the source.
    """

    source_uri = None
    source_type = None

    def __init__(self, source_uri=None, source_type=None, **kwargs):
        """Initialize ContextSource.

        Args:
           source_uri (str): The URI of the source.
           source_type (str): The type of the source.
           **kwargs: Arbitrary keyword arguments.
        """
        super(ContextSource, self).__init__(
            source_uri=source_uri, source_type=source_type, **kwargs
        )


class ArtifactSummary(_base_types.ApiObject):
    """Summary model of an Artifact.

    Attributes:
        artifact_arn (str): ARN of artifact.
        artifact_name (str): Name of artifact.
        source (obj): Source of artifact.
        artifact_type (str): Type of artifact.
        creation_time (datetime): Creation time.
        last_modified_time (datetime): Date last modified.
    """

    _custom_boto_types = {
        "source": (ArtifactSource, False),
    }
    artifact_arn = None
    artifact_name = None
    source = None
    artifact_type = None
    creation_time = None
    last_modified_time = None


class ActionSummary(_base_types.ApiObject):
    """Summary model of an action.

    Attributes:
        action_arn (str): ARN of action.
        action_name (str): Name of action.
        source (obj): Source of action.
        action_type (str): Type of action.
        status (str): The status of the action.
        creation_time (datetime): Creation time.
        last_modified_time (datetime): Date last modified.
    """

    _custom_boto_types = {
        "source": (ActionSource, False),
    }
    action_arn = None
    action_name = None
    source = None
    action_type = None
    status = None
    creation_time = None
    last_modified_time = None


class ContextSummary(_base_types.ApiObject):
    """Summary model of an context.

    Attributes:
        context_arn (str): ARN of context.
        context_name (str): Name of context.
        source (obj): Source of context.
        context_type (str): Type of context.
        creation_time (datetime): Creation time.
        last_modified_time (datetime): Date last modified.
    """

    _custom_boto_types = {
        "source": (ContextSource, False),
    }
    context_arn = None
    context_name = None
    source = None
    context_type = None
    creation_time = None
    last_modified_time = None


class UserContext(_base_types.ApiObject):
    """Summary model of a user context.

    Attributes:
        user_profile_arn (str): User profile ARN.
        user_profile_name (str): User profile name.
        domain_id (str): DomainId.
    """

    user_profile_arn = None
    user_profile_name = None
    domain_id = None

    def __init__(self, user_profile_arn=None, user_profile_name=None, domain_id=None, **kwargs):
        """Initialize UserContext.

        Args:
            user_profile_arn (str): User profile ARN.
            user_profile_name (str): User profile name.
            domain_id (str): DomainId.
            **kwargs: Arbitrary keyword arguments.
        """
        super(UserContext, self).__init__(
            user_profile_arn=user_profile_arn,
            user_profile_name=user_profile_name,
            domain_id=domain_id,
            **kwargs
        )


class AssociationSummary(_base_types.ApiObject):
    """Summary model of an association.

    Attributes:
        source_arn (str): ARN of source entity.
        source_name (str): Name of the source entity.
        destination_arn (str): ARN of the destination entity.
        destination_name (str): Name of the destination entity.
        source_type (obj): Type of the source entity.
        destination_type (str): Type of destination entity.
        association_type (str): The type of the association.
        creation_time (datetime): Creation time.
        created_by (obj): Context on creator.
    """

    _custom_boto_types = {
        "created_by": (UserContext, False),
    }
    source_arn = None
    source_name = None
    destination_arn = None
    destination_name = None
    source_type = None
    destination_type = None
    association_type = None
    creation_time = None
