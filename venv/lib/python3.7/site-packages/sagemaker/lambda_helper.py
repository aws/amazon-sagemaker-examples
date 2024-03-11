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
"""This module contains helper methods related to Lambda."""
from __future__ import print_function, absolute_import

from io import BytesIO
import zipfile
import time
from botocore.exceptions import ClientError

from sagemaker import s3
from sagemaker.session import Session


class Lambda:
    """Contains lambda boto3 wrappers to Create, Update, Delete and Invoke Lambda functions."""

    def __init__(
        self,
        function_arn: str = None,
        function_name: str = None,
        execution_role_arn: str = None,
        zipped_code_dir: str = None,
        s3_bucket: str = None,
        script: str = None,
        handler: str = None,
        session: Session = None,
        timeout: int = 120,
        memory_size: int = 128,
        runtime: str = "python3.8",
        vpc_config: dict = None,
        environment: dict = None,
        layers: list = None,
    ):
        """Constructs a Lambda instance.

        This instance represents a Lambda function and provides methods for updating,
        deleting and invoking the function.

        This class can be used either for creating a new Lambda function or using an existing one.
        When using an existing Lambda function, only the function_arn argument is required.
        When creating a new one the function_name, execution_role_arn and handler arguments
        are required, as well as either script or zipped_code_dir.

        Args:
            function_arn (str): The arn of the Lambda function.
            function_name (str): The name of the Lambda function.
                Function name must be provided to create a Lambda function.
            execution_role_arn (str): The role to be attached to Lambda function.
            zipped_code_dir (str): The path of the zipped code package of the Lambda function.
            s3_bucket (str): The bucket where zipped code is uploaded.
                If not provided, default session bucket is used to upload zipped_code_dir.
            script (str): The path of Lambda function script for direct zipped upload
            handler (str): The Lambda handler. The format for handler should be
                file_name.function_name. For ex: if the name of the Lambda script is
                hello_world.py and Lambda function definition in that script is
                lambda_handler(event, context), the handler should be hello_world.lambda_handler
            session (sagemaker.session.Session): Session object which manages interactions
                with Amazon SageMaker APIs and any other AWS services needed.
                If not specified, new session is created.
            timeout (int): Timeout of the Lambda function in seconds. Default is 120 seconds.
            memory_size (int): Memory of the Lambda function in megabytes. Default is 128 MB.
            runtime (str): Runtime of the Lambda function. Default is set to python3.8.
            vpc_config (dict): VPC to deploy the Lambda function to. Default is None.
            environment (dict): Environment Variables for the Lambda function. Default is None.
            layers (list): List of Lambda layers for the Lambda function. Default is None.
        """
        self.function_arn = function_arn
        self.function_name = function_name
        self.zipped_code_dir = zipped_code_dir
        self.s3_bucket = s3_bucket
        self.script = script
        self.handler = handler
        self.execution_role_arn = execution_role_arn
        self.session = session if session is not None else Session()
        self.timeout = timeout
        self.memory_size = memory_size
        self.runtime = runtime
        self.vpc_config = vpc_config or {}
        self.environment = environment or {}
        self.layers = layers or []

        if function_arn is None and function_name is None:
            raise ValueError("Either function_arn or function_name must be provided.")

        if function_name is not None:
            if execution_role_arn is None:
                raise ValueError("execution_role_arn must be provided.")
            if zipped_code_dir is None and script is None:
                raise ValueError("Either zipped_code_dir or script must be provided.")
            if zipped_code_dir and script:
                raise ValueError("Provide either script or zipped_code_dir, not both.")
            if handler is None:
                raise ValueError("Lambda handler must be provided.")

        if function_arn is not None:
            if zipped_code_dir and script:
                raise ValueError("Provide either script or zipped_code_dir, not both.")

    def create(self):
        """Method to create a lambda function.

        Returns: boto3 response from Lambda's create_function method.
        """
        lambda_client = _get_lambda_client(self.session)

        if self.function_name is None:
            raise ValueError("FunctionName must be provided to create a Lambda function.")

        if self.script is not None:
            code = {"ZipFile": _zip_lambda_code(self.script)}
        else:
            bucket, key_prefix = s3.determine_bucket_and_prefix(
                bucket=self.s3_bucket, key_prefix=None, sagemaker_session=self.session
            )
            key = _upload_to_s3(
                s3_client=_get_s3_client(self.session),
                function_name=self.function_name,
                zipped_code_dir=self.zipped_code_dir,
                s3_bucket=bucket,
                s3_key_prefix=key_prefix,
            )
            code = {"S3Bucket": bucket, "S3Key": key}

        try:
            response = lambda_client.create_function(
                FunctionName=self.function_name,
                Runtime=self.runtime,
                Handler=self.handler,
                Role=self.execution_role_arn,
                Code=code,
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                VpcConfig=self.vpc_config,
                Environment=self.environment,
                Layers=self.layers,
            )
            return response
        except ClientError as e:
            error = e.response["Error"]
            raise ValueError(error)

    def update(self):
        """Method to update a lambda function.

        Returns: boto3 response from Lambda's update_function method.
        """
        lambda_client = _get_lambda_client(self.session)
        retry_attempts = 7
        for i in range(retry_attempts):
            try:
                if self.script is not None:
                    response = lambda_client.update_function_code(
                        FunctionName=self.function_name or self.function_arn,
                        ZipFile=_zip_lambda_code(self.script),
                    )
                else:
                    bucket, key_prefix = s3.determine_bucket_and_prefix(
                        bucket=self.s3_bucket, key_prefix=None, sagemaker_session=self.session
                    )

                    # get function name to be used in S3 upload path
                    if self.function_arn:
                        versioned_function_name = self.function_arn.split("funtion:")[-1]
                        if ":" in versioned_function_name:
                            function_name_for_s3 = versioned_function_name.split(":")[0]
                        else:
                            function_name_for_s3 = versioned_function_name
                    else:
                        function_name_for_s3 = self.function_name

                    response = lambda_client.update_function_code(
                        FunctionName=(self.function_name or self.function_arn),
                        S3Bucket=bucket,
                        S3Key=_upload_to_s3(
                            s3_client=_get_s3_client(self.session),
                            function_name=function_name_for_s3,
                            zipped_code_dir=self.zipped_code_dir,
                            s3_bucket=bucket,
                            s3_key_prefix=key_prefix,
                        ),
                    )
                return response
            except ClientError as e:
                error = e.response["Error"]
                code = error["Code"]
                if code == "ResourceConflictException":
                    if i == retry_attempts - 1:
                        raise ValueError(error)
                    # max wait time = 2**0 + 2**1 + .. + 2**6 = 127 seconds
                    time.sleep(2**i)
                else:
                    raise ValueError(error)

    def upsert(self):
        """Method to create a lambda function or update it if it already exists

        Returns: boto3 response from Lambda's methods.
        """
        try:
            return self.create()
        except ValueError as error:
            if "ResourceConflictException" in str(error):
                return self.update()
            raise

    def invoke(self):
        """Method to invoke a lambda function.

        Returns: boto3 response from Lambda's invoke method.
        """
        lambda_client = _get_lambda_client(self.session)
        try:
            response = lambda_client.invoke(
                FunctionName=self.function_name or self.function_arn,
                InvocationType="RequestResponse",
            )
            return response
        except ClientError as e:
            error = e.response["Error"]
            raise ValueError(error)

    def delete(self):
        """Method to delete a lambda function.

        Returns: boto3 response from Lambda's delete_function method.
        """
        lambda_client = _get_lambda_client(self.session)
        try:
            response = lambda_client.delete_function(
                FunctionName=self.function_name or self.function_arn
            )
            return response
        except ClientError as e:
            error = e.response["Error"]
            raise ValueError(error)


def _get_s3_client(session):
    """Method to get a boto3 s3 client.

    Returns: a s3 client.
    """
    sagemaker_session = session or Session()
    if sagemaker_session.s3_client is None:
        s3_client = sagemaker_session.boto_session.client(
            "s3", region_name=sagemaker_session.boto_region_name
        )
    else:
        s3_client = sagemaker_session.s3_client
    return s3_client


def _get_lambda_client(session):
    """Method to get a boto3 lambda client.

    Returns: a lambda client.
    """
    sagemaker_session = session or Session()
    if sagemaker_session.lambda_client is None:
        lambda_client = sagemaker_session.boto_session.client(
            "lambda", region_name=sagemaker_session.boto_region_name
        )
    else:
        lambda_client = sagemaker_session.lambda_client
    return lambda_client


def _upload_to_s3(s3_client, function_name, zipped_code_dir, s3_bucket, s3_key_prefix=None):
    """Upload the zipped code to S3 bucket provided in the Lambda instance.

    Lambda instance must have a path to the zipped code folder and a S3 bucket to upload
    the code. The key will lambda/function_name/code and the S3 URI where the code is
    uploaded is in this format: s3://bucket_name/lambda/function_name/code.

    Returns: the S3 key where the code is uploaded.
    """

    key = s3.s3_path_join(
        s3_key_prefix,
        "lambda",
        function_name,
        "code",
    )
    s3_client.upload_file(zipped_code_dir, s3_bucket, key)
    return key


def _zip_lambda_code(script):
    """This method zips the lambda function script.

    Lambda function script is provided in the lambda instance and reads that zipped file.

    Returns: A buffer of zipped lambda function script.
    """
    buffer = BytesIO()
    code_dir = script.split("/")[-1]

    with zipfile.ZipFile(buffer, "w") as z:
        z.write(script, code_dir)
    buffer.seek(0)
    return buffer.read()
