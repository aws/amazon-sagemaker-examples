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
"""This module contains code to create and manage SageMaker ``Context``."""
from __future__ import absolute_import

from datetime import datetime
from typing import Iterator, Optional, List

from sagemaker.apiutils import _base_types
from sagemaker.lineage import (
    _api_types,
    _utils,
    association,
)
from sagemaker.lineage._api_types import ContextSummary
from sagemaker.lineage.query import (
    LineageQuery,
    LineageFilter,
    LineageSourceEnum,
    LineageEntityEnum,
    LineageQueryDirectionEnum,
)
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.action import Action
from sagemaker.lineage.lineage_trial_component import LineageTrialComponent


class Context(_base_types.Record):
    """An Amazon SageMaker context, which is part of a SageMaker lineage.

    Attributes:
        context_arn (str): The ARN of the context.
        context_name (str): The name of the context.
        context_type (str): The type of the context.
        description (str): A description of the context.
        source (obj): The source of the context with a URI and type.
        properties (dict): Dictionary of properties.
        tags (List[dict[str, str]]): A list of tags to associate with the context.
        creation_time (datetime): When the context was created.
        created_by (obj): Contextual info on which account created the context.
        last_modified_time (datetime): When the context was last modified.
        last_modified_by (obj): Contextual info on which account created the context.
    """

    context_arn: str = None
    context_name: str = None
    context_type: str = None
    properties: dict = None
    tags: list = None
    creation_time: datetime = None
    created_by: str = None
    last_modified_time: datetime = None
    last_modified_by: str = None

    _boto_load_method: str = "describe_context"
    _boto_create_method: str = "create_context"
    _boto_update_method: str = "update_context"
    _boto_delete_method: str = "delete_context"

    _custom_boto_types = {
        "source": (_api_types.ContextSource, False),
    }

    _boto_update_members = [
        "context_name",
        "description",
        "properties",
        "properties_to_remove",
    ]
    _boto_delete_members = ["context_name"]

    def save(self) -> "Context":
        """Save the state of this Context to SageMaker.

        Returns:
            obj: boto API response.
        """
        return self._invoke_api(self._boto_update_method, self._boto_update_members)

    def delete(self, disassociate: bool = False):
        """Delete the context object.

        Args:
            disassociate (bool): When set to true, disassociate incoming and outgoing association.

        Returns:
            obj: boto API response.
        """
        if disassociate:
            _utils._disassociate(
                source_arn=self.context_arn, sagemaker_session=self.sagemaker_session
            )
            _utils._disassociate(
                destination_arn=self.context_arn,
                sagemaker_session=self.sagemaker_session,
            )
        return self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    def set_tag(self, tag=None):
        """Add a tag to the object.

        Args:
            tag (obj): Key value pair to set tag.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.context_arn, tags=[tag])

    def set_tags(self, tags=None):
        """Add tags to the object.

        Args:
            tags ([{key:value}]): list of key value pairs.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.context_arn, tags=tags)

    @classmethod
    def load(cls, context_name: str, sagemaker_session=None) -> "Context":
        """Load an existing context and return an ``Context`` object representing it.

        Examples:
            .. code-block:: python

                from sagemaker.lineage import context

                my_context = context.Context.create(
                    context_name='MyContext',
                    context_type='Endpoint',
                    source_uri='arn:aws:...')

                my_context.properties["added"] = "property"
                my_context.save()

                for ctx in context.Context.list():
                    print(ctx)

                my_context.delete()

            Args:
                context_name (str): Name of the context
                sagemaker_session (sagemaker.session.Session): Session object which
                    manages interactions with Amazon SageMaker APIs and any other
                    AWS services needed. If not specified, one is created using the
                    default AWS configuration chain.

            Returns:
                Context: A SageMaker ``Context`` object
        """
        context = cls._construct(
            cls._boto_load_method,
            context_name=context_name,
            sagemaker_session=sagemaker_session,
        )
        return context

    @classmethod
    def create(
        cls,
        context_name: str = None,
        source_uri: str = None,
        source_type: str = None,
        context_type: str = None,
        description: str = None,
        properties: dict = None,
        tags: dict = None,
        sagemaker_session=None,
    ) -> "Context":
        """Create a context and return a ``Context`` object representing it.

        Args:
            context_name (str): The name of the context.
            source_uri (str): The source URI of the context.
            source_type (str): The type of the source.
            context_type (str): The type of the context.
            description (str): Description of the context.
            properties (dict): Metadata associated with the context.
            tags (dict): Tags to add to the context.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Context: A SageMaker ``Context`` object.
        """
        return super(Context, cls)._construct(
            cls._boto_create_method,
            context_name=context_name,
            source=_api_types.ContextSource(source_uri=source_uri, source_type=source_type),
            context_type=context_type,
            description=description,
            properties=properties,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def list(
        cls,
        source_uri: Optional[str] = None,
        context_type: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
        sagemaker_session=None,
    ) -> Iterator[ContextSummary]:
        """Return a list of context summaries.

        Args:
            source_uri (str, optional): A source URI.
            context_type (str, optional): An context type.
            created_before (datetime.datetime, optional): Return contexts created before this
                instant.
            created_after (datetime.datetime, optional): Return contexts created after this instant.
            sort_by (str, optional): Which property to sort results by.
                One of 'SourceArn', 'CreatedBefore', 'CreatedAfter'
            sort_order (str, optional): One of 'Ascending', or 'Descending'.
            max_results (int, optional): maximum number of contexts to retrieve
            next_token (str, optional): token for next page of results
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            collections.Iterator[ContextSummary]: An iterator
                over ``ContextSummary`` objects.
        """
        return super(Context, cls)._list(
            "list_contexts",
            _api_types.ContextSummary.from_boto,
            "ContextSummaries",
            source_uri=source_uri,
            context_type=context_type,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            max_results=max_results,
            next_token=next_token,
            sagemaker_session=sagemaker_session,
        )

    def actions(self, direction: LineageQueryDirectionEnum) -> List[Action]:
        """Use the lineage query to retrieve actions that use this context.

        Args:
            direction (LineageQueryDirectionEnum): The query direction.

        Returns:
            list of Actions: Actions.
        """
        query_filter = LineageFilter(entities=[LineageEntityEnum.ACTION])
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.context_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]


class EndpointContext(Context):
    """An Amazon SageMaker endpoint context, which is part of a SageMaker lineage."""

    def models(self) -> List[association.Association]:
        """Use Lineage API to get all models deployed by this endpoint.

        Returns:
            list of Associations: Associations that destination represents an endpoint's model.
        """
        endpoint_actions: Iterator = association.Association.list(
            sagemaker_session=self.sagemaker_session,
            source_arn=self.context_arn,
            destination_type="ModelDeployment",
        )

        model_list: list = [
            model
            for endpoint_action in endpoint_actions
            for model in association.Association.list(
                source_arn=endpoint_action.destination_arn,
                destination_type="Model",
                sagemaker_session=self.sagemaker_session,
            )
        ]
        return model_list

    def models_v2(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.DESCENDANTS
    ) -> List[Artifact]:
        """Use the lineage query to retrieve downstream model artifacts that use this endpoint.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Artifacts: Artifacts representing a model.
        """
        # Firstly query out the model_deployment vertices
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.ACTION], sources=[LineageSourceEnum.MODEL_DEPLOYMENT]
        )
        model_deployment_query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.context_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        if not model_deployment_query_result:
            return []

        model_deployment_vertices: [] = model_deployment_query_result.vertices

        # Secondary query model based on model deployment
        model_vertices = []
        for vertex in model_deployment_vertices:
            query_result = LineageQuery(self.sagemaker_session).query(
                start_arns=[vertex.arn],
                query_filter=LineageFilter(
                    entities=[LineageEntityEnum.ARTIFACT], sources=[LineageSourceEnum.MODEL]
                ),
                direction=LineageQueryDirectionEnum.DESCENDANTS,
                include_edges=False,
            )
            model_vertices.extend(query_result.vertices)

        if not model_vertices:
            return []

        model_artifacts = []
        for vertex in model_vertices:
            lineage_object = vertex.to_lineage_object()
            model_artifacts.append(lineage_object)

        return model_artifacts

    def dataset_artifacts(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> List[Artifact]:
        """Use the lineage query to retrieve datasets that use this endpoint.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of Artifacts: Artifacts representing a dataset.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.ARTIFACT], sources=[LineageSourceEnum.DATASET]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.context_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )

        return [vertex.to_lineage_object() for vertex in query_result.vertices]

    def training_job_arns(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> List[str]:
        """Get ARNs for all training jobs that appear in the endpoint's lineage.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of str: Training job ARNs.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.TRIAL_COMPONENT], sources=[LineageSourceEnum.TRAINING_JOB]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.context_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )

        training_job_arns = []
        for vertex in query_result.vertices:
            trial_component_name = _utils.get_resource_name_from_arn(vertex.arn)
            trial_component = self.sagemaker_session.sagemaker_client.describe_trial_component(
                TrialComponentName=trial_component_name
            )
            training_job_arns.append(trial_component["Source"]["SourceArn"])
        return training_job_arns

    def processing_jobs(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> List[LineageTrialComponent]:
        """Use the lineage query to retrieve processing jobs that use this endpoint.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of LineageTrialComponent: Lineage trial component that represent Processing jobs.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.TRIAL_COMPONENT], sources=[LineageSourceEnum.PROCESSING_JOB]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.context_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]

    def transform_jobs(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> List[LineageTrialComponent]:
        """Use the lineage query to retrieve transform jobs that use this endpoint.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of LineageTrialComponent: Lineage trial component that represent Transform jobs.
        """
        query_filter = LineageFilter(
            entities=[LineageEntityEnum.TRIAL_COMPONENT], sources=[LineageSourceEnum.TRANSFORM_JOB]
        )
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.context_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]

    def trial_components(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> List[LineageTrialComponent]:
        """Use the lineage query to retrieve trial components that use this endpoint.

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

        Returns:
            list of LineageTrialComponent: Lineage trial component.
        """
        query_filter = LineageFilter(entities=[LineageEntityEnum.TRIAL_COMPONENT])
        query_result = LineageQuery(self.sagemaker_session).query(
            start_arns=[self.context_arn],
            query_filter=query_filter,
            direction=direction,
            include_edges=False,
        )
        return [vertex.to_lineage_object() for vertex in query_result.vertices]

    def pipeline_execution_arn(
        self, direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.ASCENDANTS
    ) -> str:
        """Get the ARN for the pipeline execution associated with this endpoint (if any).

        Args:
            direction (LineageQueryDirectionEnum, optional): The query direction.

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


class ModelPackageGroup(Context):
    """An Amazon SageMaker model package group context, which is part of a SageMaker lineage."""

    def pipeline_execution_arn(self) -> str:
        """Get the ARN for the pipeline execution associated with this model package group (if any).

        Returns:
            str: A pipeline execution ARN.
        """
        return self.properties.get("PipelineExecutionArn")
