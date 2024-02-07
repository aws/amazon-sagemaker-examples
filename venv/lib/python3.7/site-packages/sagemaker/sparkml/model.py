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
"""Placeholder docstring"""
from __future__ import absolute_import

from typing import Union, Optional

from sagemaker import Model, Predictor, Session, image_uris
from sagemaker.serializers import CSVSerializer
from sagemaker.utils import pop_out_unused_kwarg
from sagemaker.workflow.entities import PipelineVariable

framework_name = "sparkml-serving"


class SparkMLPredictor(Predictor):
    """Performs predictions against an MLeap serialized SparkML model.

    The implementation of
    :meth:`~sagemaker.predictor.Predictor.predict` in this
    `Predictor` requires a json as input. The input should follow the
    json format as documented.

    ``predict()`` returns a csv output, comma separated if the output is a
    list.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=CSVSerializer(),
        **kwargs,
    ):
        """Initializes a SparkMLPredictor which should be used with SparkMLModel.

        It is used to perform predictions against SparkML models serialized via MLeap.
        The response is returned in text/csv format which is the default response
        format for SparkML Serving container.

        Args:
            endpoint (str): The name of the endpoint to perform inference on.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to text/csv.
        """
        sagemaker_session = sagemaker_session or Session()
        super(SparkMLPredictor, self).__init__(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=serializer,
            **kwargs,
        )


class SparkMLModel(Model):
    """Model data and S3 location holder for MLeap serialized SparkML model.

    Calling :meth:`~sagemaker.model.Model.deploy` creates an Endpoint and return
    a Predictor to performs predictions against an MLeap serialized SparkML
    model .
    """

    def __init__(
        self,
        model_data: Union[str, PipelineVariable],
        role: Optional[str] = None,
        spark_version: str = "3.3",
        sagemaker_session: Optional[Session] = None,
        **kwargs,
    ):
        """Initialize a SparkMLModel.

        Args:
            model_data (str or PipelineVariable): The S3 location of a SageMaker model data
                ``.tar.gz`` file. For SparkML, this will be the output that has
                been produced by the Spark job after serializing the Model via
                MLeap.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            spark_version (str): Spark version you want to use for executing the
                inference (default: '3.3').
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain. For local mode,
                please do not pass this variable.
            **kwargs: Additional parameters passed to the
                :class:`~sagemaker.model.Model` constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.Model`.
        """
        # For local mode, sagemaker_session should be passed as None but we need a session to get
        # boto_region_name
        region_name = (sagemaker_session or Session()).boto_region_name
        image_uri = image_uris.retrieve(framework_name, region_name, version=spark_version)
        pop_out_unused_kwarg("predictor_cls", kwargs, SparkMLPredictor.__name__)
        pop_out_unused_kwarg("image_uri", kwargs, image_uri)
        super(SparkMLModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=SparkMLPredictor,
            sagemaker_session=sagemaker_session,
            **kwargs,
        )
