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
"""The step definitions for workflow."""
from __future__ import absolute_import

from typing import Any, Dict, List, Union, Optional

from sagemaker.workflow.entities import (
    RequestType,
)
from sagemaker.workflow.properties import (
    Properties,
)
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.steps import Step, StepTypeEnum, CacheConfig


class EMRStepConfig:
    """Config for a Hadoop Jar step."""

    def __init__(
        self, jar, args: List[str] = None, main_class: str = None, properties: List[dict] = None
    ):
        """Create a definition for input data used by an EMR cluster(job flow) step.

        See AWS documentation for more information about the `StepConfig
        <https://docs.aws.amazon.com/emr/latest/APIReference/API_StepConfig.html>`_ API parameters.

        Args:
            args(List[str]):
                A list of command line arguments passed to
                the JAR file's main function when executed.
            jar(str): A path to a JAR file run during the step.
            main_class(str): The name of the main class in the specified Java file.
            properties(List(dict)): A list of key-value pairs that are set when the step runs.
        """
        self.jar = jar
        self.args = args
        self.main_class = main_class
        self.properties = properties

    def to_request(self) -> RequestType:
        """Convert EMRStepConfig object to request dict."""
        config = {"HadoopJarStep": {"Jar": self.jar}}
        if self.args is not None:
            config["HadoopJarStep"]["Args"] = self.args
        if self.main_class is not None:
            config["HadoopJarStep"]["MainClass"] = self.main_class
        if self.properties is not None:
            config["HadoopJarStep"]["Properties"] = self.properties

        return config


INSTANCES = "Instances"
INSTANCEGROUPS = "InstanceGroups"
INSTANCEFLEETS = "InstanceFleets"
ERR_STR_WITH_NAME_AUTO_TERMINATION_OR_STEPS = (
    "In EMRStep {step_name}, cluster_config "
    "should not contain any of the Name, "
    "AutoTerminationPolicy and/or Steps."
)

ERR_STR_WITHOUT_INSTANCE = "In EMRStep {step_name}, cluster_config must contain " + INSTANCES + "."

ERR_STR_WITH_KEEPJOBFLOW_OR_TERMINATIONPROTECTED = (
    "In EMRStep {step_name}, " + INSTANCES + " should not contain "
    "KeepJobFlowAliveWhenNoSteps or "
    "TerminationProtected."
)

ERR_STR_BOTH_OR_NONE_INSTANCEGROUPS_OR_INSTANCEFLEETS = (
    "In EMRStep {step_name}, "
    + INSTANCES
    + " should contain either "
    + INSTANCEGROUPS
    + " or "
    + INSTANCEFLEETS
    + "."
)

ERR_STR_WITH_BOTH_CLUSTER_ID_AND_CLUSTER_CFG = (
    "EMRStep {step_name} can not have both cluster_id"
    "or cluster_config."
    "To use EMRStep with "
    "cluster_config, cluster_id "
    "must be explicitly set to None."
)

ERR_STR_WITH_EXEC_ROLE_ARN_AND_WITHOUT_CLUSTER_ID = (
    "EMRStep {step_name} cannot have execution_role_arn"
    "without cluster_id."
    "To use EMRStep with "
    "execution_role_arn, cluster_id "
    "must not be None."
)

ERR_STR_WITHOUT_CLUSTER_ID_AND_CLUSTER_CFG = (
    "EMRStep {step_name} must have either cluster_id or cluster_config"
)


class EMRStep(Step):
    """EMR step for workflow."""

    def _validate_cluster_config(self, cluster_config, step_name):
        """Validates user provided cluster_config.

        Args:
            cluster_config(Union[Dict[str, Any], List[Dict[str, Any]]]):
                user provided cluster configuration.
            step_name: The name of the EMR step.
        """

        if (
            "Name" in cluster_config
            or "AutoTerminationPolicy" in cluster_config
            or "Steps" in cluster_config
        ):
            raise ValueError(
                ERR_STR_WITH_NAME_AUTO_TERMINATION_OR_STEPS.format(step_name=step_name)
            )

        if INSTANCES not in cluster_config:
            raise ValueError(ERR_STR_WITHOUT_INSTANCE.format(step_name=step_name))

        if (
            "KeepJobFlowAliveWhenNoSteps" in cluster_config[INSTANCES]
            or "TerminationProtected" in cluster_config[INSTANCES]
        ):
            raise ValueError(
                ERR_STR_WITH_KEEPJOBFLOW_OR_TERMINATIONPROTECTED.format(step_name=step_name)
            )

        if (
            INSTANCEGROUPS in cluster_config[INSTANCES]
            and INSTANCEFLEETS in cluster_config[INSTANCES]
        ) or (
            INSTANCEGROUPS not in cluster_config[INSTANCES]
            and INSTANCEFLEETS not in cluster_config[INSTANCES]
        ):
            raise ValueError(
                ERR_STR_BOTH_OR_NONE_INSTANCEGROUPS_OR_INSTANCEFLEETS.format(step_name=step_name)
            )

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        cluster_id: str,
        step_config: EMRStepConfig,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        cache_config: CacheConfig = None,
        cluster_config: Dict[str, Any] = None,
        execution_role_arn: str = None,
    ):
        """Constructs an `EMRStep`.

        Args:
            name(str): The name of the EMR step.
            display_name(str): The display name of the EMR step.
            description(str): The description of the EMR step.
            cluster_id(str): The ID of the running EMR cluster.
            step_config(EMRStepConfig): One StepConfig to be executed by the job flow.
            depends_on (List[Union[str, Step, StepCollection]]): A list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that this `EMRStep`
                depends on.
            cache_config(CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            cluster_config(Dict[str, Any]): The recipe of the
                EMR cluster, passed as a dictionary.
                The elements are defined in the request syntax for `RunJobFlow`.
                However, the following elements are not recognized as part of the cluster
                configuration and you should not include them in the dictionary:

                * ``cluster_config[Name]``
                * ``cluster_config[Steps]``
                * ``cluster_config[AutoTerminationPolicy]``
                * ``cluster_config[Instances][KeepJobFlowAliveWhenNoSteps]``
                * ``cluster_config[Instances][TerminationProtected]``

                For more information about the fields you can include in your cluster
                configuration, see
                https://docs.aws.amazon.com/emr/latest/APIReference/API_RunJobFlow.html.
                Note that if you want to use ``cluster_config``, then you have to set
                ``cluster_id`` as None.
            execution_role_arn(str): The ARN of the runtime role assumed by this `EMRStep`. The
                job submitted to your EMR cluster uses this role to access AWS resources. This
                value is passed as ExecutionRoleArn to the AddJobFlowSteps request (an EMR request)
                called on the cluster specified by ``cluster_id``, so you can only include this
                field if ``cluster_id`` is not None.
        """
        super(EMRStep, self).__init__(name, display_name, description, StepTypeEnum.EMR, depends_on)

        emr_step_args = {"StepConfig": step_config.to_request()}
        root_property = Properties(step_name=name, shape_name="Step", service_name="emr")

        if cluster_id is None and cluster_config is None:
            raise ValueError(ERR_STR_WITHOUT_CLUSTER_ID_AND_CLUSTER_CFG.format(step_name=name))

        if cluster_id is not None and cluster_config is not None:
            raise ValueError(ERR_STR_WITH_BOTH_CLUSTER_ID_AND_CLUSTER_CFG.format(step_name=name))

        if execution_role_arn is not None and cluster_id is None:
            raise ValueError(
                ERR_STR_WITH_EXEC_ROLE_ARN_AND_WITHOUT_CLUSTER_ID.format(step_name=name)
            )

        if cluster_id is not None:
            emr_step_args["ClusterId"] = cluster_id
            root_property.__dict__["ClusterId"] = cluster_id

            if execution_role_arn is not None:
                emr_step_args["ExecutionRoleArn"] = execution_role_arn
                root_property.__dict__["ExecutionRoleArn"] = execution_role_arn
        elif cluster_config is not None:
            self._validate_cluster_config(cluster_config, name)
            emr_step_args["ClusterConfig"] = cluster_config
            root_property.__dict__["ClusterConfig"] = cluster_config

        self.args = emr_step_args
        self.cache_config = cache_config
        self._properties = root_property

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `AddJobFlowSteps`.

        NOTE: The AddFlowJobSteps request is not quite the args list that workflow needs.
        The Name attribute in AddJobFlowSteps cannot be passed; it will be set during runtime.
        In addition to that, we will also need to include emr job inputs and output config.
        """
        return self.args

    @property
    def properties(self) -> RequestType:
        """A Properties object representing the EMR DescribeStepResponse model"""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)
        return request_dict
