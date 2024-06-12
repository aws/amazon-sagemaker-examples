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
"""This module contains code to create and manage SageMaker ``Artifact``."""
from __future__ import absolute_import

import logging
import math

from datetime import datetime
from typing import Iterator, Union, Any, Optional, List

from sagemaker.apiutils import _base_types, _utils
from sagemaker.lineage import _api_types
from sagemaker.lineage._api_types import ArtifactSource, ArtifactSummary
from sagemaker.lineage.query import (
    LineageQuery,
    LineageFilter,
    LineageSourceEnum,
    LineageEntityEnum,
    LineageQueryDirectionEnum,
)
from sagemaker.lineage._utils import _disassociate, get_resource_name_from_arn
from sagemaker.lineage.association import Association
from sagemaker.utils import get_module

LOGGER = logging.getLogger("sagemaker")


class Artifact(_base_types.Record):
    """An Amazon SageMaker artifact, which is part of a SageMaker lineage.

    Examples:
        .. code-block:: python

            from sagemaker.lineage import artifact

            my_artifact = artifact.Artifact.create(
                artifact_name='MyArtifact',
                artifact_type='S3File',
                source_uri='s3://...')

            my_artifact.properties["added"] = "property"
            my_artifact.save()

            for artfct in artifact.Artifact.list():
                print(artfct)

            my_artifact.delete()

    Attributes:
        artifact_arn (str): The ARN of the artifact.
        artifact_name (str): The name of the artifact.
        artifact_type (str): The type of the artifact.
        source (obj): The source of the artifact with a URI and types.
        properties (dict): Dictionary of properties.
        tags (List[dict[str, str]]): A list of tags to associate with the artifact.
        creation_time (datetime): When the artifact was created.
        created_by (obj): Contextual info on which account created the artifact.
        last_modified_time (datetime): When the artifact was last modified.
        last_modified_by (obj): Contextual info on which account created the artifact.
    """

    artifact_arn: str = None
    artifact_name: str = None
    artifact_type: str = None
    source: ArtifactSource = None
    properties: dict = None
    tags: list = None
    creation_time: datetime = None
    created_by: str = None
    last_modified_time: datetime = None
    last_modified_by: str = None

    _boto_create_method: str = "create_artifact"
    _boto_load_method: str = "describe_artifact"
    _boto_update_method: str = "update_artifact"
    _boto_delete_method: str = "delete_artifact"

    _boto_update_members = [
        "artifact_arn",
        "artifact_name",
        "properties",
        "properties_to_remove",
    ]

    _boto_delete_members = ["artifact_arn"]

    _custom_boto_types = {"source": (_api_types.ArtifactSource, False)}

    def save(self) -> "Artifact":
        """Save the state of this Artifact to SageMaker.

        Note that this method must be run from a SageMaker context such as Studio or a training job
        due to restrictions on the CreateArtifact API.

        Returns:
            Artifact: A SageMaker `Artifact` object.
        """
        return self._invoke_api(self._boto_update_method, self._boto_update_members)

    def delete(self, disassociate: bool = False):
        """Delete the artifact object.

        Args:
            disassociate (bool): When set to true, disassociate incoming and outgoing association.
        """
        if disassociate:
            _disassociate(source_arn=self.artifact_arn, sagemaker_session=self.sagemaker_session)
            _disassociate(
                destination_arn=self.artifact_arn,
                sagemaker_session=self.sagemaker_session,
            )
        self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    @classmethod
    def load(cls, artifact_arn: str, sagemaker_session=None) -> "Artifact":
        """Load an existing artifact and return an ``Artifact`` object representing it.

        Args:
            artifact_arn (str): ARN of the artifact
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Artifact: A SageMaker ``Artifact`` object
        """
        artifact = cls._construct(
            cls._boto_load_method,
            artifact_arn=artifact_arn,
            sagemaker_session=sagemaker_session,
        )
        return artifact

    def downstream_trials(self, sagemaker_session=None) -> list:
        """Use the lineage API to retrieve all downstream trials that use this artifact.

        Args:
            sagemaker_session (obj): Sagemaker Session to use. If not provided a default session
                will be created.

        Returns:
            [Trial]: A list of SageMaker `Trial` objects.
        """
        # don't specify destination type because for Trial Components it could be one of
        # SageMaker[TrainingJob|ProcessingJob|TransformJob|ExperimentTrialComponent]
        outgoing_associations: Iterator = Association.list(
            source_arn=self.artifact_arn, sagemaker_session=sagemaker_session
        )
        trial_component_arns: list = list(map(lambda x: x.destination_arn, outgoing_associations))

        return self._get_trial_from_trial_component(trial_component_arns)

    def downstream_trials_v2(self) -> list:
        """Use a lineage query to retrieve all downstream trials that use this artifact.

        Returns:
            [Trial]: A list of SageMaker `Trial` objects.
        """
        return self._trials(direction=LineageQueryDirectionEnum.DESCENDANTS)

    def upstream_trials(self) -> List:
        """Use the lineage query to retrieve all upstream trials that use this artifact.

        Returns:
            [Trial]: A list of SageMaker `Trial` objects.
        """
        return self._trials(direction=LineageQueryDirectionEnum.ASCENDANTS)

    def _trials(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.BOTH
    ) -> List:
        """Use the lineage query to retrieve all trials that use this artifact.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            [Trial]: A list of SageMaker `Trial` objects.
        """
        query_filter = LineageFilter(entities=[LineageEntityEnum.TRIAL_COMPONENT])
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.artifact_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        trial_component_arns: list = list(map(lambda x: x.arn, query_result.vertices))
        return self._get_trial_from_trial_component(trial_component_arns)

    def _get_trial_from_trial_component(self, trial_component_arns: list) -> List:
        """Retrieve all upstream trial runs which that use the trial component arns.

        Args:
            trial_component_arns (list): list of trial component arns

        Returns:
            [Trial]: A list of SageMaker `Trial` objects.
        """
        if not trial_component_arns:
            # no outgoing associations for this artifact
            return []

        get_module("smexperiments")
        from smexperiments import trial_component, search_expression

        max_search_by_arn: int = 60
        num_search_batches = math.ceil(len(trial_component_arns) % max_search_by_arn)
        trial_components: list = []

        sagemaker_session = self.sagemaker_session or _utils.default_session()
        sagemaker_client = sagemaker_session.sagemaker_client

        for i in range(num_search_batches):
            start: int = i * max_search_by_arn
            end: int = start + max_search_by_arn
            arn_batch: list = trial_component_arns[start:end]
            se: Any = self._get_search_expression(arn_batch, search_expression)
            search_result: Any = trial_component.TrialComponent.search(
                search_expression=se, sagemaker_boto_client=sagemaker_client
            )

            trial_components: list = trial_components + list(search_result)

        trials: set = set()

        for tc in list(trial_components):
            for parent in tc.parents:
                trials.add(parent["TrialName"])

        return list(trials)

    def _get_search_expression(self, arns: list, search_expression: object) -> object:
        """Convert a set of arns to a search expression.

        Args:
            arns (list): Trial Component arns to search for.
            search_expression (obj): smexperiments.search_expression

        Returns:
            search_expression (obj): Arns converted to a Trial Component search expression.
        """
        max_arn_per_filter: int = 3
        num_filters: Union[float, int] = math.ceil(len(arns) / max_arn_per_filter)
        filters: list = []

        for i in range(num_filters):
            start: int = i * max_arn_per_filter
            end: int = i + max_arn_per_filter
            batch_arns: list = arns[start:end]
            search_filter = search_expression.Filter(
                name="TrialComponentArn",
                operator=search_expression.Operator.EQUALS,
                value=",".join(batch_arns),
            )

            filters.append(search_filter)

        search_expression = search_expression.SearchExpression(
            filters=filters,
            boolean_operator=search_expression.BooleanOperator.OR,
        )
        return search_expression

    def set_tag(self, tag=None):
        """Add a tag to the object.

        Args:
            tag (obj): Key value pair to set tag.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.artifact_arn, tags=[tag])

    def set_tags(self, tags=None):
        """Add tags to the object.

        Args:
            tags ([{key:value}]): list of key value pairs.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.artifact_arn, tags=tags)

    @classmethod
    def create(
        cls,
        artifact_name: Optional[str] = None,
        source_uri: Optional[str] = None,
        source_types: Optional[list] = None,
        artifact_type: Optional[str] = None,
        properties: Optional[dict] = None,
        tags: Optional[dict] = None,
        sagemaker_session=None,
    ) -> "Artifact":
        """Create an artifact and return an ``Artifact`` object representing it.

        Args:
            artifact_name (str, optional): Name of the artifact
            source_uri (str, optional): Source URI of the artifact
            source_types (list, optional): Source types
            artifact_type (str, optional): Type of the artifact
            properties (dict, optional): key/value properties
            tags (dict, optional): AWS tags for the artifact
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Artifact: A SageMaker ``Artifact`` object.
        """
        return super(Artifact, cls)._construct(
            cls._boto_create_method,
            artifact_name=artifact_name,
            source=_api_types.ArtifactSource(source_uri=source_uri, source_types=source_types),
            artifact_type=artifact_type,
            properties=properties,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def list(
        cls,
        source_uri: Optional[str] = None,
        artifact_type: Optional[str] = None,
        created_before: Optional[datetime] = None,
        created_after: Optional[datetime] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
        sagemaker_session=None,
    ) -> Iterator[ArtifactSummary]:
        """Return a list of artifact summaries.

        Args:
            source_uri (str, optional): A source URI.
            artifact_type (str, optional): An artifact type.
            created_before (datetime.datetime, optional): Return artifacts created before this
                instant.
            created_after (datetime.datetime, optional): Return artifacts created after this
                instant.
            sort_by (str, optional): Which property to sort results by.
                One of 'SourceArn', 'CreatedBefore','CreatedAfter'
            sort_order (str, optional): One of 'Ascending', or 'Descending'.
            max_results (int, optional): maximum number of artifacts to retrieve
            next_token (str, optional): token for next page of results
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            collections.Iterator[ArtifactSummary]: An iterator
                over ``ArtifactSummary`` objects.
        """
        return super(Artifact, cls)._list(
            "list_artifacts",
            _api_types.ArtifactSummary.from_boto,
            "ArtifactSummaries",
            source_uri=source_uri,
            artifact_type=artifact_type,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            max_results=max_results,
            next_token=next_token,
            sagemaker_session=sagemaker_session,
        )

    def s3_uri_artifacts(self, s3_uri: str) -> dict:
        """Retrieve a list of artifacts that use provided s3 uri.

        Args:
            s3_uri (str): A S3 URI.

        Returns:
            A list of ``Artifacts``
        """
        return self.sagemaker_session.sagemaker_client.list_artifacts(SourceUri=s3_uri)


class ModelArtifact(Artifact):
    """A SageMaker lineage artifact representing a model.

    Common model specific lineage traversals to discover how the model is connected
    to other entities.
    """

    from sagemaker.lineage.context import Context

    def endpoints(self) -> list:
        """Get association summaries for endpoints deployed with this model.

        Returns:
            [AssociationSummary]: A list of associations representing the endpoints using the model.
        """
        endpoint_development_actions: Iterator = Association.list(
            source_arn=self.artifact_arn,
            destination_type="Action",
            sagemaker_session=self.sagemaker_session,
        )

        endpoint_context_list: list = [
            endpoint_context_associations
            for endpoint_development_action in endpoint_development_actions
            for endpoint_context_associations in Association.list(
                source_arn=endpoint_development_action.destination_arn,
                destination_type="Context",
                sagemaker_session=self.sagemaker_session,
            )
        ]
        return endpoint_context_list

    def endpoint_contexts(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.DESCENDANTS
    ) -> List[Context]:
        """Get contexts representing endpoints from the models's lineage.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Contexts: Contexts representing an endpoint.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.CONTEXT], sources=[LineageSourceEnum.ENDPOINT]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.artifact_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )

        endpoint_contexts = []
        for vertex in query_result.vertices:
            endpoint_contexts.append(vertex.to_lineage_object())
        return endpoint_contexts

    def dataset_artifacts(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> List[Artifact]:
        """Get artifacts representing datasets from the model's lineage.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Artifacts: Artifacts representing a dataset.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.ARTIFACT], sources=[LineageSourceEnum.DATASET]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.artifact_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )

        dataset_artifacts = []
        for vertex in query_result.vertices:
            dataset_artifacts.append(vertex.to_lineage_object())
        return dataset_artifacts

    def training_job_arns(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> List[str]:
        """Get ARNs for all training jobs that appear in the model's lineage.

        Returns:
            list of str: Training job ARNs.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.TRIAL_COMPONENT], sources=[LineageSourceEnum.TRAINING_JOB]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.artifact_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )

        training_job_arns = []
        for vertex in query_result.vertices:
            trial_component_name = get_resource_name_from_arn(vertex.arn)
            trial_component = self.sagemaker_session.sagemaker_client.describe_trial_component(
                TrialComponentName=trial_component_name
            )
            training_job_arns.append(trial_component["Source"]["SourceArn"])
        return training_job_arns

    def pipeline_execution_arn(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> str:
        """Get the ARN for the pipeline execution associated with this model (if any).

        Returns:
            str: A pipeline execution ARN.
        """
        training_job_arns = self.training_job_arns(direction=direction)
        for training_job_arn in training_job_arns:
            tags = self.sagemaker_session.sagemaker_client.list_tags(ResourceArn=training_job_arn)[
                "Tags"
            ]
            for tag in tags:
                if tag["Key"] == "sagemaker:pipeline-execution-arn":
                    return tag["Value"]

        return None


class DatasetArtifact(Artifact):
    """A SageMaker Lineage artifact representing a dataset.

    Encapsulates common dataset specific lineage traversals to discover how the dataset is
    connect to related entities.
    """

    from sagemaker.lineage.context import Context

    def trained_models(self) -> List[Association]:
        """Given a dataset artifact, get associated trained models.

        Returns:
            list(Association): List of Contexts representing model artifacts.
        """
        trial_components: Iterator = Association.list(
            source_arn=self.artifact_arn, sagemaker_session=self.sagemaker_session
        )
        result: list = []
        for trial_component in trial_components:
            if "experiment-trial-component" in trial_component.destination_arn:
                models = Association.list(
                    source_arn=trial_component.destination_arn,
                    destination_type="Context",
                    sagemaker_session=self.sagemaker_session,
                )
                result.extend(models)

        return result

    def endpoint_contexts(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.DESCENDANTS
    ) -> List[Context]:
        """Get contexts representing endpoints from the dataset's lineage.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Contexts: Contexts representing an endpoint.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.CONTEXT], sources=[LineageSourceEnum.ENDPOINT]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.artifact_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )

        endpoint_contexts = []
        for vertex in query_result.vertices:
            endpoint_contexts.append(vertex.to_lineage_object())
        return endpoint_contexts

    def upstream_datasets(self) -> List[Artifact]:
        """Use the lineage query to retrieve upstream artifacts that use this dataset artifact.

        Returns:
            list of Artifacts: Artifacts representing an dataset.
        """
        return self._datasets(direction=LineageQueryDirectionEnum.ASCENDANTS)

    def downstream_datasets(self) -> List[Artifact]:
        """Use the lineage query to retrieve downstream artifacts that use this dataset.

        Returns:
            list of Artifacts: Artifacts representing an dataset.
        """
        return self._datasets(direction=LineageQueryDirectionEnum.DESCENDANTS)

    def _datasets(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.BOTH
    ) -> List[Artifact]:
        """Use the lineage query to retrieve all artifacts that use this dataset.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Artifacts: Artifacts representing an dataset.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.ARTIFACT], sources=[LineageSourceEnum.DATASET]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.artifact_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]


class ImageArtifact(Artifact):
    """A SageMaker lineage artifact representing an image.

    Common model specific lineage traversals to discover how the image is connected
    to other entities.
    """

    def datasets(self, direction: LineageQueryDirectionEnum) -> List[Artifact]:
        """Use the lineage query to retrieve datasets that use this image artifact.

        Args:
            direction (LineageQueryDirectionEnum): The query direction.

        Returns:
            list of Artifacts: Artifacts representing a dataset.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.ARTIFACT], sources=[LineageSourceEnum.DATASET]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.artifact_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]
