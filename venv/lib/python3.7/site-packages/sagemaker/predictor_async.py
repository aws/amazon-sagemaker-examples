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
import threading
import time
import uuid
from botocore.exceptions import WaiterError

from sagemaker import s3
from sagemaker.exceptions import PollingTimeoutError, AsyncInferenceModelError
from sagemaker.async_inference import WaiterConfig, AsyncInferenceResponse
from sagemaker.s3 import parse_s3_url
from sagemaker.session import Session
from sagemaker.utils import name_from_base, sagemaker_timestamp


class AsyncPredictor:
    """Make async prediction requests to an Amazon SageMaker endpoint."""

    def __init__(
        self,
        predictor,
        name=None,
    ):
        """Initialize an ``AsyncPredictor``.

        Args:
            predictor (sagemaker.predictor.Predictor): General ``Predictor``
                object has useful methods and variables. ``AsyncPredictor``
                stands on top of it with capability for async inference.
        """
        self.predictor = predictor
        self.endpoint_name = predictor.endpoint_name
        self.sagemaker_session = predictor.sagemaker_session or Session()
        if self.sagemaker_session.s3_client is None:
            self.s3_client = self.sagemaker_session.boto_session.client(
                "s3",
                region_name=self.sagemaker_session.boto_region_name,
            )
        else:
            self.s3_client = self.sagemaker_session.s3_client

        self.serializer = predictor.serializer
        self.deserializer = predictor.deserializer
        self.name = name
        self._endpoint_config_name = None
        self._model_names = None
        self._context = None
        self._input_path = None

    def predict(
        self,
        data=None,
        input_path=None,
        initial_args=None,
        inference_id=None,
        waiter_config=WaiterConfig(),
    ):
        """Wait and return the Async Inference result from the specified endpoint.

        Args:
            data (object): Input data for which you want the model to provide
                inference. If a serializer was specified in the encapsulated
                Predictor object, the result of the serializer is sent as input
                data. Otherwise the data must be sequence of bytes, and the
                predict method then sends the bytes in the request body as is.
            input_path (str): Amazon S3 URI contains input data for which you want
                the model to provide async inference. (Default: None)
            initial_args (dict[str,str]): Optional. Default arguments for boto3
                ``invoke_endpoint_async`` call. (Default: None).
            inference_id (str): If you provide a value, it is added to the captured data
                when you enable data capture on the endpoint (Default: None).
            waiter_config (sagemaker.async_inference.waiter_config.WaiterConfig): Configuration
                for the waiter. (Default: {"Delay": 15 seconds, "MaxAttempts": 60}
        Raises:
            ValueError: If both input data and input Amazon S3 path are not provided
        Returns:
            object: Inference for the given input. If a deserializer was specified when creating
                the Predictor, the result of the deserializer is
                returned. Otherwise the response returns the sequence of bytes
                as is.
        """
        if data is None and input_path is None:
            raise ValueError(
                "Please provide input data or input Amazon S3 location to use async prediction"
            )
        if data is not None:
            input_path = self._upload_data_to_s3(data, input_path)

        self._input_path = input_path
        response = self._submit_async_request(input_path, initial_args, inference_id)
        output_location = response["OutputLocation"]
        failure_location = response.get("FailureLocation")
        result = self._wait_for_output(
            output_path=output_location, failure_path=failure_location, waiter_config=waiter_config
        )

        return result

    def predict_async(
        self,
        data=None,
        input_path=None,
        initial_args=None,
        inference_id=None,
    ):
        """Return the Async Inference ouput Amazon S3 path from the specified endpoint.

        Args:
            data (object): Input data for which you want the model to provide
                inference. If a serializer was specified in the encapsulated
                Predictor object, the result of the serializer is sent as input
                data. Otherwise the data must be sequence of bytes, and the
                predict method then upload the data to the ``input_s3_path``. If
                ``input_s3_path`` is None, upload the data to
            input_path (str): Amazon S3 URI contains input data for which you want
                the model to provide async inference. (Default: None)
            initial_args (dict[str,str]): Optional. Default arguments for boto3
                ``invoke_endpoint_async`` call. (Default: None).
            inference_id (str): If you provide a value, it is added to the captured data
                when you enable data capture on the endpoint (Default: None).
        Raises:
            ValueError: If both input data and input Amazon S3 path are not provided
        Returns:
            AsyncInferenceResponse:
                Inference response for the given input. It provides method to check
                the result in the Amazon S3 output path.
        """
        if data is None and input_path is None:
            raise ValueError(
                "Please provide input data or input Amazon S3 location to use async prediction"
            )
        if data is not None:
            input_path = self._upload_data_to_s3(data, input_path)

        self._input_path = input_path
        response = self._submit_async_request(input_path, initial_args, inference_id)
        output_location = response["OutputLocation"]
        failure_location = response.get("FailureLocation")
        response_async = AsyncInferenceResponse(
            predictor_async=self,
            output_path=output_location,
            failure_path=failure_location,
        )

        return response_async

    def _upload_data_to_s3(
        self,
        data,
        input_path=None,
    ):
        """Upload request data to Amazon S3 for users"""
        if input_path:
            bucket, key = parse_s3_url(input_path)
        else:
            my_uuid = str(uuid.uuid4())
            timestamp = sagemaker_timestamp()
            bucket = self.sagemaker_session.default_bucket()
            key = s3.s3_path_join(
                self.sagemaker_session.default_bucket_prefix,
                "async-endpoint-inputs",
                name_from_base(self.name, short=True),
                "{}-{}".format(timestamp, my_uuid),
            )

        data = self.serializer.serialize(data)
        self.s3_client.put_object(
            Body=data, Bucket=bucket, Key=key, ContentType=self.serializer.CONTENT_TYPE
        )
        input_path = input_path or "s3://{}/{}".format(bucket, key)

        return input_path

    def _create_request_args(
        self,
        input_path,
        initial_args=None,
        inference_id=None,
    ):
        """Create request args for ``invoke_endpoint_async``"""
        args = dict(initial_args) if initial_args else {}
        args["InputLocation"] = input_path
        if "EndpointName" not in args:
            args["EndpointName"] = self.predictor.endpoint_name

        if "Accept" not in args:
            args["Accept"] = ", ".join(self.predictor.accept)

        if inference_id:
            args["InferenceId"] = inference_id

        return args

    def _submit_async_request(
        self,
        input_path,
        initial_args,
        inference_id,
    ):
        """Create request and invoke async endpoint with the request"""
        request_args = self._create_request_args(input_path, initial_args, inference_id)

        response = self.sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async(
            **request_args
        )

        return response

    def _wait_for_output(self, output_path, failure_path, waiter_config):
        """Retrieve output based on the presense of failure_path."""
        if failure_path is not None:
            return self._check_output_and_failure_paths(output_path, failure_path, waiter_config)

        return self._check_output_path(output_path, waiter_config)

    def _check_output_path(self, output_path, waiter_config):
        """Check the Amazon S3 output path for the output.

        Periodically check Amazon S3 output path for async inference result.
        Timeout automatically after max attempts reached
        """
        bucket, key = parse_s3_url(output_path)
        s3_waiter = self.s3_client.get_waiter("object_exists")
        try:
            s3_waiter.wait(Bucket=bucket, Key=key, WaiterConfig=waiter_config._to_request_dict())
        except WaiterError:
            raise PollingTimeoutError(
                message="Inference could still be running",
                output_path=output_path,
                seconds=waiter_config.delay * waiter_config.max_attempts,
            )
        s3_object = self.s3_client.get_object(Bucket=bucket, Key=key)
        result = self.predictor._handle_response(response=s3_object)
        return result

    def _check_output_and_failure_paths(self, output_path, failure_path, waiter_config):
        """Check the Amazon S3 output path for the output.

        This method waits for either the output file or the failure file to be found on the
        specified S3 output path. Whichever file is found first, its corresponding event is
        triggered, and the method executes the appropriate action based on the event.

        Args:
            output_path (str): The S3 path where the output file is expected to be found.
            failure_path (str): The S3 path where the failure file is expected to be found.
            waiter_config (boto3.waiter.WaiterConfig): The configuration for the S3 waiter.

        Returns:
            Any: The deserialized result from the output file, if the output file is found first.
            Otherwise, raises an exception.

        Raises:
            AsyncInferenceModelError: If the failure file is found before the output file.
            PollingTimeoutError: If both files are not found and the S3 waiter
             has thrown a WaiterError.
        """
        output_bucket, output_key = parse_s3_url(output_path)
        failure_bucket, failure_key = parse_s3_url(failure_path)

        output_file_found = threading.Event()
        failure_file_found = threading.Event()

        def check_output_file():
            try:
                output_file_waiter = self.s3_client.get_waiter("object_exists")
                output_file_waiter.wait(
                    Bucket=output_bucket,
                    Key=output_key,
                    WaiterConfig=waiter_config._to_request_dict(),
                )
                output_file_found.set()
            except WaiterError:
                pass

        def check_failure_file():
            try:
                failure_file_waiter = self.s3_client.get_waiter("object_exists")
                failure_file_waiter.wait(
                    Bucket=failure_bucket,
                    Key=failure_key,
                    WaiterConfig=waiter_config._to_request_dict(),
                )
                failure_file_found.set()
            except WaiterError:
                pass

        output_thread = threading.Thread(target=check_output_file)
        failure_thread = threading.Thread(target=check_failure_file)

        output_thread.start()
        failure_thread.start()

        while not output_file_found.is_set() and not failure_file_found.is_set():
            time.sleep(1)

        if output_file_found.is_set():
            s3_object = self.s3_client.get_object(Bucket=output_bucket, Key=output_key)
            result = self.predictor._handle_response(response=s3_object)
            return result

        failure_object = self.s3_client.get_object(Bucket=failure_bucket, Key=failure_key)
        failure_response = self.predictor._handle_response(response=failure_object)

        raise AsyncInferenceModelError(
            message=failure_response
        ) if failure_file_found.is_set() else PollingTimeoutError(
            message="Inference could still be running",
            output_path=output_path,
            seconds=waiter_config.delay * waiter_config.max_attempts,
        )

    def update_endpoint(
        self,
        initial_instance_count=None,
        instance_type=None,
        accelerator_type=None,
        model_name=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        wait=True,
    ):
        """Update the existing endpoint with the provided attributes.

        This creates a new EndpointConfig in the process. If ``initial_instance_count``,
        ``instance_type``, ``accelerator_type``, or ``model_name`` is specified, then a new
        ProductionVariant configuration is created; values from the existing configuration
        are not preserved if any of those parameters are specified.

        Args:
            initial_instance_count (int): The initial number of instances to run in the endpoint.
                This is required if ``instance_type``, ``accelerator_type``, or ``model_name`` is
                specified. Otherwise, the values from the existing endpoint configuration's
                ProductionVariants are used.
            instance_type (str): The EC2 instance type to deploy the endpoint to.
                This is required if ``initial_instance_count`` or ``accelerator_type`` is specified.
                Otherwise, the values from the existing endpoint configuration's
                ``ProductionVariants`` are used.
            accelerator_type (str): The type of Elastic Inference accelerator to attach to
                the endpoint, e.g. "ml.eia1.medium". If not specified, and
                ``initial_instance_count``, ``instance_type``, and ``model_name`` are also ``None``,
                the values from the existing endpoint configuration's ``ProductionVariants`` are
                used. Otherwise, no Elastic Inference accelerator is attached to the endpoint.
            model_name (str): The name of the model to be associated with the endpoint.
                This is required if ``initial_instance_count``, ``instance_type``, or
                ``accelerator_type`` is specified and if there is more than one model associated
                with the endpoint. Otherwise, the existing model for the endpoint is used.
            tags (list[dict[str, str]]): The list of tags to add to the endpoint
                config. If not specified, the tags of the existing endpoint configuration are used.
                If any of the existing tags are reserved AWS ones (i.e. begin with "aws"),
                they are not carried over to the new endpoint configuration.
            kms_key (str): The KMS key that is used to encrypt the data on the storage volume
                attached to the instance hosting the endpoint If not specified,
                the KMS key of the existing endpoint configuration is used.
            data_capture_config_dict (dict): The endpoint data capture configuration
                for use with Amazon SageMaker Model Monitoring. If not specified,
                the data capture configuration of the existing endpoint configuration is used.
            wait (bool): Wait for updating to finish
        """

        self.predictor.update_endpoint(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            model_name=model_name,
            tags=tags,
            kms_key=kms_key,
            data_capture_config_dict=data_capture_config_dict,
            wait=wait,
        )

    def delete_endpoint(self, delete_endpoint_config=True):
        """Delete the Amazon SageMaker endpoint backing this async predictor.

        This also delete the endpoint configuration attached to it if
        delete_endpoint_config is True.

        Args:
            delete_endpoint_config (bool, optional): Flag to indicate whether to
                delete endpoint configuration together with endpoint. Defaults
                to True. If True, both endpoint and endpoint configuration will
                be deleted. If False, only endpoint will be deleted.
        """
        self.predictor.delete_endpoint(delete_endpoint_config)

    def delete_model(self):
        """Deletes the Amazon SageMaker models backing this predictor."""
        self.predictor.delete_model()

    def enable_data_capture(self):
        """Enables data capture by updating DataCaptureConfig.

        This function updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker
        Endpoint to enable data capture. For a more customized experience, refer to
        update_data_capture_config, instead.
        """
        self.predictor.enable_data_capture()

    def disable_data_capture(self):
        """Disables data capture by updating DataCaptureConfig.

        This function updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker
        Endpoint to disable data capture. For a more customized experience, refer to
        update_data_capture_config, instead.
        """
        self.predictor.disable_data_capture()

    def update_data_capture_config(self, data_capture_config):
        """Updates the DataCaptureConfig for the Predictor's associated Amazon SageMaker Endpoint.

        Update is done using the provided DataCaptureConfig.

        Args:
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): The
                DataCaptureConfig to update the predictor's endpoint to use.
        """
        self.predictor.update_data_capture_config(data_capture_config)

    def list_monitors(self):
        """Generates ModelMonitor objects (or DefaultModelMonitors).

        Objects are generated based on the schedule(s) associated with the endpoint
        that this predictor refers to.

        Returns:
            [sagemaker.model_monitor.model_monitoring.ModelMonitor]: A list of
                ModelMonitor (or DefaultModelMonitor) objects.

        """
        return self.predictor.list_monitors()

    def endpoint_context(self):
        """Retrieves the lineage context object representing the endpoint.

        Examples:
            .. code-block:: python

                predictor = Predictor()
                context = predictor.endpoint_context()
                models = context.models()

        Returns:
            ContextEndpoint: The context for the endpoint.
        """

        return self.predictor.endpoint_context()
