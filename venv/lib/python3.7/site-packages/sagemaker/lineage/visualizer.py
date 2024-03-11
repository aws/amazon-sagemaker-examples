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
"""This module contains functionality to display lineage data."""
from __future__ import absolute_import

import logging

from typing import Optional, Any, Iterator

import pandas as pd
from pandas import DataFrame

from sagemaker.lineage._api_types import AssociationSummary
from sagemaker.lineage.association import Association


class LineageTableVisualizer(object):
    """Creates a dataframe containing the lineage assoociations of a SageMaker object."""

    def __init__(self, sagemaker_session):
        """Init for LineageTableVisualizer.

        Args:
            sagemaker_session (obj): The sagemaker session used for API requests.
        """
        self._session = sagemaker_session

    def show(
        self,
        trial_component_name: Optional[str] = None,
        training_job_name: Optional[str] = None,
        processing_job_name: Optional[str] = None,
        pipeline_execution_step: Optional[object] = None,
        model_package_arn: Optional[str] = None,
        endpoint_arn: Optional[str] = None,
        artifact_arn: Optional[str] = None,
        context_arn: Optional[str] = None,
        actions_arn: Optional[str] = None,
    ) -> DataFrame:
        """Generate a dataframe containing all incoming and outgoing lineage entities.

          Examples:
          .. code-block:: python

              viz = LineageTableVisualizer(sagemaker_session)
              df = viz.show(training_job_name=training_job_name)
              # in a notebook
              display(df.to_html())

        Args:
            trial_component_name (str, optional): Name of  a trial component. Defaults to None.
            training_job_name (str, optional): Name of a training job. Defaults to None.
            processing_job_name (str, optional): Name of a processing job. Defaults to None.
            pipeline_execution_step (obj, optional): Pipeline execution step. Defaults to None.
            model_package_arn (str, optional): Model package arn. Defaults to None.
            endpoint_arn (str, optional): Endpoint arn. Defaults to None.
            artifact_arn (str, optional): Artifact arn. Defaults to None.
            context_arn (str, optional): Context arn. Defaults to None.
            actions_arn (str, optional): Action arn. Defaults to None.

        Returns:
            DataFrame: Pandas dataframe containing lineage associations.
        """
        start_arn: str = None

        if trial_component_name:
            start_arn = self._get_start_arn_from_trial_component_name(trial_component_name)
        elif training_job_name:
            trial_component_name = training_job_name + "-aws-training-job"
            start_arn = self._get_start_arn_from_trial_component_name(trial_component_name)
        elif processing_job_name:
            trial_component_name = processing_job_name + "-aws-processing-job"
            start_arn = self._get_start_arn_from_trial_component_name(trial_component_name)
        elif pipeline_execution_step:
            start_arn = self._get_start_arn_from_pipeline_execution_step(pipeline_execution_step)
        elif model_package_arn:
            start_arn = self._get_start_arn_from_model_package_arn(model_package_arn)
        elif endpoint_arn:
            start_arn = self._get_start_arn_from_endpoint_arn(endpoint_arn)
        elif artifact_arn:
            start_arn = artifact_arn
        elif context_arn:
            start_arn = context_arn
        elif actions_arn:
            start_arn = actions_arn

        return self._get_associations_dataframe(start_arn)

    def _get_start_arn_from_pipeline_execution_step(self, pipeline_execution_step: object) -> str:
        """Given a pipeline exection step retrieve the arn of the lineage entity that represents it.

        Args:
            pipeline_execution_step (obj): Pipeline execution step.

        Returns:
            str: The arn of the lineage entity
        """
        start_arn: str = None

        if not pipeline_execution_step["Metadata"]:
            return None

        metadata: Any = pipeline_execution_step["Metadata"]
        jobs: list = ["TrainingJob", "ProcessingJob", "TransformJob"]

        for job in jobs:
            if job in metadata and metadata[job]:
                job_arn = metadata[job]["Arn"]
                start_arn = self._get_start_arn_from_job_arn(job_arn)
                break

        if "RegisterModel" in metadata:
            start_arn = self._get_start_arn_from_model_package_arn(metadata["RegisterModel"]["Arn"])

        return start_arn

    def _get_start_arn_from_job_arn(self, job_arn: str) -> str:
        """Given a job arn return the lineage entity.

        Args:
            job_arn (str): Arn of a training, processing, or transform job.

        Returns:
          str: The arn of the job's lineage entity.
        """
        start_arn: str = None
        response: Any = self._session.sagemaker_client.list_trial_components(SourceArn=job_arn)
        trial_components: Any = response["TrialComponentSummaries"]
        if trial_components:
            start_arn = trial_components[0]["TrialComponentArn"]
        else:
            logging.warning("No trial components found for %s", job_arn)
        return start_arn

    def _get_associations_dataframe(self, arn: str) -> DataFrame:
        """Create a data frame containing lineage association information.

        Args:
            arn (str): The arn of the lineage entity of interest.

        Returns:
            DataFrame: A dataframe with association information.
        """
        if arn is None:
            # no associations
            return None

        upstream_associations: Iterator[AssociationSummary] = self._get_associations(dest_arn=arn)
        downstream_associations: Iterator[AssociationSummary] = self._get_associations(src_arn=arn)
        inputs: list = list(map(self._convert_input_association_to_df_row, upstream_associations))
        outputs: list = list(
            map(self._convert_output_association_to_df_row, downstream_associations)
        )
        df: DataFrame = pd.DataFrame(
            inputs + outputs,
            columns=[
                "Name/Source",
                "Direction",
                "Type",
                "Association Type",
                "Lineage Type",
            ],
        )
        return df

    def _get_start_arn_from_trial_component_name(self, tc_name: str) -> str:
        """Given a trial component name retrieve a start arn.

        Args:
            tc_name (str): Name of the trial compoonent.

        Returns:
          str: The arn of the trial component.
        """
        response: Any = self._session.sagemaker_client.describe_trial_component(
            TrialComponentName=tc_name
        )
        tc_arn: str = response["TrialComponentArn"]
        return tc_arn

    def _get_start_arn_from_model_package_arn(self, model_package_arn: str) -> str:
        """Given a model package arn retrieve the arn lineage entity.

        Args:
            model_package_arn (str): The arn of a model package.

        Returns:
            str: The arn of the lineage entity that represents the model package.
        """
        response: Any = self._session.sagemaker_client.list_artifacts(SourceUri=model_package_arn)
        artifacts: Any = response["ArtifactSummaries"]
        artifact_arn: str = None
        if artifacts:
            artifact_arn = artifacts[0]["ArtifactArn"]
        else:
            logging.debug("No artifacts found for %s.", model_package_arn)
        return artifact_arn

    def _get_start_arn_from_endpoint_arn(self, endpoint_arn: str) -> str:
        """Given an endpoint arn retrieve the arn of the lineage entity.

        Args:
            endpoint_arn (str): The arn of an endpoint

        Returns:
            str: The arn of the lineage entity that represents the model package.
        """
        response: Any = self._session.sagemaker_client.list_contexts(SourceUri=endpoint_arn)
        contexts: Any = response["ContextSummaries"]
        context_arn: str = None
        if contexts:
            context_arn = contexts[0]["ContextArn"]
        else:
            logging.debug("No contexts found for %s.", endpoint_arn)
        return context_arn

    def _get_associations(
        self, src_arn: Optional[str] = None, dest_arn: Optional[str] = None
    ) -> Iterator[AssociationSummary]:
        """Given an arn retrieve all associated lineage entities.

        The arn must be one of: experiment, trial, trial component, artifact, action, or context.

        Args:
            src_arn (str, optional): The arn of the source. Defaults to None.
            dest_arn (str, optional): The arn of the destination. Defaults to None.

        Returns:
            array: An array of associations that are either incoming or outgoing from the lineage
            entity of interest.
        """
        if src_arn:
            associations: Iterator[AssociationSummary] = Association.list(
                source_arn=src_arn, sagemaker_session=self._session
            )
        else:
            associations: Iterator[AssociationSummary] = Association.list(
                destination_arn=dest_arn, sagemaker_session=self._session
            )
        return associations

    def _convert_input_association_to_df_row(self, association) -> list:
        """Convert an input association to a data frame row.

        Args:
            association (obj): ``Association``

        Returns:
            array: Array of column values for the association data frame.
        """
        return self._convert_association_to_df_row(
            association.source_arn,
            association.source_name,
            "Input",
            association.source_type,
            association.association_type,
        )

    def _convert_output_association_to_df_row(self, association) -> list:
        """Convert an output association to a data frame row.

        Args:
            association (obj): ``Association``

        Returns:
            array: Array of column values for the association data frame.
        """
        return self._convert_association_to_df_row(
            association.destination_arn,
            association.destination_name,
            "Output",
            association.destination_type,
            association.association_type,
        )

    def _convert_association_to_df_row(
        self,
        arn: str,
        name: str,
        direction: str,
        src_dest_type: str,
        association_type: type,
    ) -> list:
        """Convert association data into a data frame row.

        Args:
            arn (str): The arn of the associated entity.
            name (str): The name of the associated entity.
            direction (str): The direction the association is with the entity of interest. Values
            are 'Input' or 'Output'.
            src_dest_type (str): The type of the entity that is associated with the entity of
            interest.
            association_type ([type]): The type of the association.

        Returns:
            [type]: [description]
        """
        arn_name = arn.split(":")[5]
        entity_type = arn_name.split("/")[0]
        name = self._get_friendly_name(name, arn, entity_type)
        return [name, direction, src_dest_type, association_type, entity_type]

    def _get_friendly_name(self, name: str, arn: str, entity_type: str) -> str:
        """Get a human readable name from the association.

        Args:
            name (str): The name of the associated entity
            arn (str): The arn of the associated entity
            entity_type (str): The type of the associated entity (artifact, action, etc...)

        Returns:
            str: The name for the association that will be displayed in the data frame.
        """
        if name:
            return name

        if entity_type == "artifact":
            artifact = self._session.sagemaker_client.describe_artifact(ArtifactArn=arn)
            uri = artifact["Source"]["SourceUri"]

            # shorten the uri if the length is more than 40,
            # e.g s3://flintstone-end-to-end-tests-gamma-us-west-2-069083975568/results/
            # canary-auto-1608761252626/preprocessed-data/tuning_data/train.txt
            # become s3://.../preprocessed-data/tuning_data/train.txt
            if len(uri) > 48:
                name = uri[:5] + "..." + uri[len(uri) - 40 :]

            # if not then use the full uri
            if not name:
                name = uri

        # if still don't have name derive from arn
        if not name:
            name = arn.split(":")[5].split("/")[1]

        return name
