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
"""This module contains/maintains the schema of the Config file."""
from __future__ import absolute_import, print_function

SECURITY_GROUP_IDS = "SecurityGroupIds"
SUBNETS = "Subnets"
ENABLE_NETWORK_ISOLATION = "EnableNetworkIsolation"
VOLUME_KMS_KEY_ID = "VolumeKmsKeyId"
KMS_KEY_ID = "KmsKeyId"
RESOURCE_KEY = "ResourceKey"
ROLE_ARN = "RoleArn"
TAGS = "Tags"
KEY = "Key"
VALUE = "Value"
EXECUTION_ROLE_ARN = "ExecutionRoleArn"
CLUSTER_ROLE_ARN = "ClusterRoleArn"
VPC_CONFIG = "VpcConfig"
OUTPUT_DATA_CONFIG = "OutputDataConfig"
AUTO_ML_JOB_CONFIG = "AutoMLJobConfig"
ASYNC_INFERENCE_CONFIG = "AsyncInferenceConfig"
OUTPUT_CONFIG = "OutputConfig"
PROCESSING_OUTPUT_CONFIG = "ProcessingOutputConfig"
CLUSTER_CONFIG = "ClusterConfig"
NETWORK_CONFIG = "NetworkConfig"
CORE_DUMP_CONFIG = "CoreDumpConfig"
DATA_CAPTURE_CONFIG = "DataCaptureConfig"
MONITORING_OUTPUT_CONFIG = "MonitoringOutputConfig"
RESOURCE_CONFIG = "ResourceConfig"
SCHEMA_VERSION = "SchemaVersion"
DATASET_DEFINITION = "DatasetDefinition"
ATHENA_DATASET_DEFINITION = "AthenaDatasetDefinition"
REDSHIFT_DATASET_DEFINITION = "RedshiftDatasetDefinition"
MONITORING_JOB_DEFINITION = "MonitoringJobDefinition"
SAGEMAKER = "SageMaker"
PYTHON_SDK = "PythonSDK"
MODULES = "Modules"
REMOTE_FUNCTION = "RemoteFunction"
DEPENDENCIES = "Dependencies"
PRE_EXECUTION_SCRIPT = "PreExecutionScript"
PRE_EXECUTION_COMMANDS = "PreExecutionCommands"
ENVIRONMENT_VARIABLES = "EnvironmentVariables"
IMAGE_URI = "ImageUri"
INCLUDE_LOCAL_WORKDIR = "IncludeLocalWorkDir"
INSTANCE_TYPE = "InstanceType"
S3_KMS_KEY_ID = "S3KmsKeyId"
S3_ROOT_URI = "S3RootUri"
JOB_CONDA_ENV = "JobCondaEnvironment"
OFFLINE_STORE_CONFIG = "OfflineStoreConfig"
ONLINE_STORE_CONFIG = "OnlineStoreConfig"
S3_STORAGE_CONFIG = "S3StorageConfig"
SECURITY_CONFIG = "SecurityConfig"
TRANSFORM_JOB_DEFINITION = "TransformJobDefinition"
MONITORING_SCHEDULE_CONFIG = "MonitoringScheduleConfig"
MONITORING_RESOURCES = "MonitoringResources"
PROCESSING_RESOURCES = "ProcessingResources"
PRODUCTION_VARIANTS = "ProductionVariants"
TRANSFORM_OUTPUT = "TransformOutput"
TRANSFORM_RESOURCES = "TransformResources"
VALIDATION_ROLE = "ValidationRole"
VALIDATION_SPECIFICATION = "ValidationSpecification"
VALIDATION_PROFILES = "ValidationProfiles"
PROCESSING_INPUTS = "ProcessingInputs"
FEATURE_GROUP = "FeatureGroup"
EDGE_PACKAGING_JOB = "EdgePackagingJob"
TRAINING_JOB = "TrainingJob"
PROCESSING_JOB = "ProcessingJob"
MODEL_PACKAGE = "ModelPackage"
MODEL = "Model"
MONITORING_SCHEDULE = "MonitoringSchedule"
ENDPOINT_CONFIG = "EndpointConfig"
ENDPOINT = "Endpoint"
AUTO_ML_JOB = "AutoMLJob"
COMPILATION_JOB = "CompilationJob"
CUSTOM_PARAMETERS = "CustomParameters"
PIPELINE = "Pipeline"
TRANSFORM_JOB = "TransformJob"
PROPERTIES = "properties"
PATTERN_PROPERTIES = "patternProperties"
TYPE = "type"
OBJECT = "object"
ADDITIONAL_PROPERTIES = "additionalProperties"
ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION = "EnableInterContainerTrafficEncryption"
SESSION = "Session"
DEFAULT_S3_BUCKET = "DefaultS3Bucket"
DEFAULT_S3_OBJECT_KEY_PREFIX = "DefaultS3ObjectKeyPrefix"
ENVIRONMENT = "Environment"
CONTAINERS = "Containers"
PRIMARY_CONTAINER = "PrimaryContainer"
INFERENCE_SPECIFICATION = "InferenceSpecification"
PROFILER_CONFIG = "ProfilerConfig"
DISABLE_PROFILER = "DisableProfiler"
ESTIMATOR = "Estimator"
DEBUG_HOOK_CONFIG = "DebugHookConfig"


def _simple_path(*args: str):
    """Appends an arbitrary number of strings to use as path constants"""
    return ".".join(args)


# Paths for reference elsewhere in the code.
COMPILATION_JOB_VPC_CONFIG_PATH = _simple_path(SAGEMAKER, COMPILATION_JOB, VPC_CONFIG)
COMPILATION_JOB_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, COMPILATION_JOB, OUTPUT_CONFIG, KMS_KEY_ID
)
COMPILATION_JOB_OUTPUT_CONFIG_PATH = _simple_path(SAGEMAKER, COMPILATION_JOB, OUTPUT_CONFIG)
COMPILATION_JOB_ROLE_ARN_PATH = _simple_path(SAGEMAKER, COMPILATION_JOB, ROLE_ARN)
TRAINING_JOB_ENVIRONMENT_PATH = _simple_path(SAGEMAKER, TRAINING_JOB, ENVIRONMENT)
TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH = _simple_path(
    SAGEMAKER, TRAINING_JOB, ENABLE_NETWORK_ISOLATION
)
TRAINING_JOB_KMS_KEY_ID_PATH = _simple_path(SAGEMAKER, TRAINING_JOB, OUTPUT_DATA_CONFIG, KMS_KEY_ID)
TRAINING_JOB_RESOURCE_CONFIG_PATH = _simple_path(SAGEMAKER, TRAINING_JOB, RESOURCE_CONFIG)
TRAINING_JOB_OUTPUT_DATA_CONFIG_PATH = _simple_path(SAGEMAKER, TRAINING_JOB, OUTPUT_DATA_CONFIG)
TRAINING_JOB_VOLUME_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, TRAINING_JOB, RESOURCE_CONFIG, VOLUME_KMS_KEY_ID
)
TRAINING_JOB_ROLE_ARN_PATH = _simple_path(SAGEMAKER, TRAINING_JOB, ROLE_ARN)
TRAINING_JOB_VPC_CONFIG_PATH = _simple_path(SAGEMAKER, TRAINING_JOB, VPC_CONFIG)
TRAINING_JOB_SECURITY_GROUP_IDS_PATH = _simple_path(
    TRAINING_JOB_VPC_CONFIG_PATH, SECURITY_GROUP_IDS
)
TRAINING_JOB_SUBNETS_PATH = _simple_path(TRAINING_JOB_VPC_CONFIG_PATH, SUBNETS)
TRAINING_JOB_PROFILE_CONFIG_PATH = _simple_path(SAGEMAKER, TRAINING_JOB, PROFILER_CONFIG)
TRAINING_JOB_DISABLE_PROFILER_PATH = _simple_path(
    TRAINING_JOB_PROFILE_CONFIG_PATH, DISABLE_PROFILER
)
EDGE_PACKAGING_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, EDGE_PACKAGING_JOB, OUTPUT_CONFIG, KMS_KEY_ID
)
EDGE_PACKAGING_OUTPUT_CONFIG_PATH = _simple_path(SAGEMAKER, EDGE_PACKAGING_JOB, OUTPUT_CONFIG)
EDGE_PACKAGING_ROLE_ARN_PATH = _simple_path(SAGEMAKER, EDGE_PACKAGING_JOB, ROLE_ARN)
EDGE_PACKAGING_RESOURCE_KEY_PATH = _simple_path(SAGEMAKER, EDGE_PACKAGING_JOB, RESOURCE_KEY)
ENDPOINT_CONFIG_DATA_CAPTURE_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, ENDPOINT_CONFIG, DATA_CAPTURE_CONFIG, KMS_KEY_ID
)
ENDPOINT_CONFIG_DATA_CAPTURE_PATH = _simple_path(SAGEMAKER, ENDPOINT_CONFIG, DATA_CAPTURE_CONFIG)
ENDPOINT_CONFIG_ASYNC_INFERENCE_PATH = _simple_path(
    SAGEMAKER, ENDPOINT_CONFIG, ASYNC_INFERENCE_CONFIG
)
ENDPOINT_CONFIG_PRODUCTION_VARIANTS_PATH = _simple_path(
    SAGEMAKER, ENDPOINT_CONFIG, PRODUCTION_VARIANTS
)
ENDPOINT_CONFIG_ASYNC_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, ENDPOINT_CONFIG, ASYNC_INFERENCE_CONFIG, OUTPUT_CONFIG, KMS_KEY_ID
)
ENDPOINT_CONFIG_KMS_KEY_ID_PATH = _simple_path(SAGEMAKER, ENDPOINT_CONFIG, KMS_KEY_ID)
ENDPOINT_TAGS_PATH = _simple_path(SAGEMAKER, ENDPOINT, TAGS)
FEATURE_GROUP_ONLINE_STORE_CONFIG_PATH = _simple_path(SAGEMAKER, FEATURE_GROUP, ONLINE_STORE_CONFIG)
FEATURE_GROUP_OFFLINE_STORE_CONFIG_PATH = _simple_path(
    SAGEMAKER, FEATURE_GROUP, OFFLINE_STORE_CONFIG
)
FEATURE_GROUP_ROLE_ARN_PATH = _simple_path(SAGEMAKER, FEATURE_GROUP, ROLE_ARN)
FEATURE_GROUP_OFFLINE_STORE_KMS_KEY_ID_PATH = _simple_path(
    FEATURE_GROUP_OFFLINE_STORE_CONFIG_PATH, S3_STORAGE_CONFIG, KMS_KEY_ID
)
FEATURE_GROUP_ONLINE_STORE_KMS_KEY_ID_PATH = _simple_path(
    FEATURE_GROUP_ONLINE_STORE_CONFIG_PATH, SECURITY_CONFIG, KMS_KEY_ID
)
AUTO_ML_OUTPUT_CONFIG_PATH = _simple_path(SAGEMAKER, AUTO_ML_JOB, OUTPUT_DATA_CONFIG)
AUTO_ML_KMS_KEY_ID_PATH = _simple_path(SAGEMAKER, AUTO_ML_JOB, OUTPUT_DATA_CONFIG, KMS_KEY_ID)
AUTO_ML_VOLUME_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, AUTO_ML_JOB, AUTO_ML_JOB_CONFIG, SECURITY_CONFIG, VOLUME_KMS_KEY_ID
)
AUTO_ML_ROLE_ARN_PATH = _simple_path(SAGEMAKER, AUTO_ML_JOB, ROLE_ARN)
AUTO_ML_VPC_CONFIG_PATH = _simple_path(
    SAGEMAKER, AUTO_ML_JOB, AUTO_ML_JOB_CONFIG, SECURITY_CONFIG, VPC_CONFIG
)
AUTO_ML_JOB_CONFIG_PATH = _simple_path(SAGEMAKER, AUTO_ML_JOB, AUTO_ML_JOB_CONFIG)
MONITORING_JOB_DEFINITION_PREFIX = _simple_path(
    SAGEMAKER,
    MONITORING_SCHEDULE,
    MONITORING_SCHEDULE_CONFIG,
    MONITORING_JOB_DEFINITION,
)
MONITORING_JOB_ENVIRONMENT_PATH = _simple_path(MONITORING_JOB_DEFINITION_PREFIX, ENVIRONMENT)
MONITORING_JOB_OUTPUT_KMS_KEY_ID_PATH = _simple_path(
    MONITORING_JOB_DEFINITION_PREFIX, MONITORING_OUTPUT_CONFIG, KMS_KEY_ID
)
MONITORING_JOB_VOLUME_KMS_KEY_ID_PATH = _simple_path(
    MONITORING_JOB_DEFINITION_PREFIX,
    MONITORING_RESOURCES,
    CLUSTER_CONFIG,
    VOLUME_KMS_KEY_ID,
)
MONITORING_JOB_NETWORK_CONFIG_PATH = _simple_path(MONITORING_JOB_DEFINITION_PREFIX, NETWORK_CONFIG)
MONITORING_JOB_ENABLE_NETWORK_ISOLATION_PATH = _simple_path(
    MONITORING_JOB_DEFINITION_PREFIX, NETWORK_CONFIG, ENABLE_NETWORK_ISOLATION
)
MONITORING_JOB_VPC_CONFIG_PATH = _simple_path(
    MONITORING_JOB_DEFINITION_PREFIX, NETWORK_CONFIG, VPC_CONFIG
)
MONITORING_JOB_SECURITY_GROUP_IDS_PATH = _simple_path(
    MONITORING_JOB_VPC_CONFIG_PATH, SECURITY_GROUP_IDS
)
MONITORING_JOB_SUBNETS_PATH = _simple_path(MONITORING_JOB_VPC_CONFIG_PATH, SUBNETS)
MONITORING_JOB_ROLE_ARN_PATH = _simple_path(MONITORING_JOB_DEFINITION_PREFIX, ROLE_ARN)
PIPELINE_ROLE_ARN_PATH = _simple_path(SAGEMAKER, PIPELINE, ROLE_ARN)
PIPELINE_TAGS_PATH = _simple_path(SAGEMAKER, PIPELINE, TAGS)
TRANSFORM_JOB_ENVIRONMENT_PATH = _simple_path(SAGEMAKER, TRANSFORM_JOB, ENVIRONMENT)
TRANSFORM_OUTPUT_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, TRANSFORM_JOB, TRANSFORM_OUTPUT, KMS_KEY_ID
)
TRANSFORM_RESOURCES_VOLUME_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, TRANSFORM_JOB, TRANSFORM_RESOURCES, VOLUME_KMS_KEY_ID
)
TRANSFORM_JOB_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, TRANSFORM_JOB, DATA_CAPTURE_CONFIG, KMS_KEY_ID
)
TRANSFORM_JOB_VOLUME_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, TRANSFORM_JOB, TRANSFORM_RESOURCES, VOLUME_KMS_KEY_ID
)
MODEL_CONTAINERS_PATH = _simple_path(SAGEMAKER, MODEL, CONTAINERS)
MODEL_VPC_CONFIG_PATH = _simple_path(SAGEMAKER, MODEL, VPC_CONFIG)
MODEL_ENABLE_NETWORK_ISOLATION_PATH = _simple_path(SAGEMAKER, MODEL, ENABLE_NETWORK_ISOLATION)
MODEL_EXECUTION_ROLE_ARN_PATH = _simple_path(SAGEMAKER, MODEL, EXECUTION_ROLE_ARN)
MODEL_PRIMARY_CONTAINER_PATH = _simple_path(SAGEMAKER, MODEL, PRIMARY_CONTAINER)
MODEL_PRIMARY_CONTAINER_ENVIRONMENT_PATH = _simple_path(MODEL_PRIMARY_CONTAINER_PATH, ENVIRONMENT)
PROCESSING_JOB_ENABLE_NETWORK_ISOLATION_PATH = _simple_path(
    SAGEMAKER, PROCESSING_JOB, NETWORK_CONFIG, ENABLE_NETWORK_ISOLATION
)
PROCESSING_JOB_INPUTS_PATH = _simple_path(SAGEMAKER, PROCESSING_JOB, PROCESSING_INPUTS)
REDSHIFT_DATASET_DEFINITION_KMS_KEY_ID_PATH = _simple_path(
    DATASET_DEFINITION, REDSHIFT_DATASET_DEFINITION, KMS_KEY_ID
)
ATHENA_DATASET_DEFINITION_KMS_KEY_ID_PATH = _simple_path(
    DATASET_DEFINITION, ATHENA_DATASET_DEFINITION, KMS_KEY_ID
)
REDSHIFT_DATASET_DEFINITION_CLUSTER_ROLE_ARN_PATH = _simple_path(
    DATASET_DEFINITION, REDSHIFT_DATASET_DEFINITION, CLUSTER_ROLE_ARN
)
PROCESSING_JOB_NETWORK_CONFIG_PATH = _simple_path(SAGEMAKER, PROCESSING_JOB, NETWORK_CONFIG)
PROCESSING_JOB_VPC_CONFIG_PATH = _simple_path(SAGEMAKER, PROCESSING_JOB, NETWORK_CONFIG, VPC_CONFIG)
PROCESSING_JOB_SUBNETS_PATH = _simple_path(PROCESSING_JOB_VPC_CONFIG_PATH, SUBNETS)
PROCESSING_JOB_SECURITY_GROUP_IDS_PATH = _simple_path(
    PROCESSING_JOB_VPC_CONFIG_PATH, SECURITY_GROUP_IDS
)
PROCESSING_OUTPUT_CONFIG_PATH = _simple_path(SAGEMAKER, PROCESSING_JOB, PROCESSING_OUTPUT_CONFIG)
PROCESSING_JOB_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, PROCESSING_JOB, PROCESSING_OUTPUT_CONFIG, KMS_KEY_ID
)
PROCESSING_JOB_PROCESSING_RESOURCES_PATH = _simple_path(
    SAGEMAKER, PROCESSING_JOB, PROCESSING_RESOURCES
)
PROCESSING_JOB_VOLUME_KMS_KEY_ID_PATH = _simple_path(
    SAGEMAKER, PROCESSING_JOB, PROCESSING_RESOURCES, CLUSTER_CONFIG, VOLUME_KMS_KEY_ID
)
PROCESSING_JOB_ROLE_ARN_PATH = _simple_path(SAGEMAKER, PROCESSING_JOB, ROLE_ARN)
MODEL_PACKAGE_INFERENCE_SPECIFICATION_CONTAINERS_PATH = _simple_path(
    SAGEMAKER, MODEL_PACKAGE, INFERENCE_SPECIFICATION, CONTAINERS
)
MODEL_PACKAGE_VALIDATION_ROLE_PATH = _simple_path(
    SAGEMAKER, MODEL_PACKAGE, VALIDATION_SPECIFICATION, VALIDATION_ROLE
)
MODEL_PACKAGE_VALIDATION_PROFILES_PATH = _simple_path(
    SAGEMAKER, MODEL_PACKAGE, VALIDATION_SPECIFICATION, VALIDATION_PROFILES
)
REMOTE_FUNCTION_DEPENDENCIES = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, DEPENDENCIES
)
REMOTE_FUNCTION_PRE_EXECUTION_COMMANDS = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, PRE_EXECUTION_COMMANDS
)
REMOTE_FUNCTION_PRE_EXECUTION_SCRIPT = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, PRE_EXECUTION_SCRIPT
)
REMOTE_FUNCTION_ENVIRONMENT_VARIABLES = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, ENVIRONMENT_VARIABLES
)
REMOTE_FUNCTION_IMAGE_URI = _simple_path(SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, IMAGE_URI)
REMOTE_FUNCTION_INCLUDE_LOCAL_WORKDIR = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, INCLUDE_LOCAL_WORKDIR
)
REMOTE_FUNCTION_INSTANCE_TYPE = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, INSTANCE_TYPE
)
REMOTE_FUNCTION_JOB_CONDA_ENV = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, JOB_CONDA_ENV
)
REMOTE_FUNCTION_ROLE_ARN = _simple_path(SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, ROLE_ARN)
REMOTE_FUNCTION_S3_KMS_KEY_ID = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, S3_KMS_KEY_ID
)
REMOTE_FUNCTION_S3_ROOT_URI = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, S3_ROOT_URI
)
REMOTE_FUNCTION_TAGS = _simple_path(SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, TAGS)
REMOTE_FUNCTION_VOLUME_KMS_KEY_ID = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, VOLUME_KMS_KEY_ID
)
REMOTE_FUNCTION_VPC_CONFIG_SUBNETS = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, VPC_CONFIG, SUBNETS
)
REMOTE_FUNCTION_VPC_CONFIG_SECURITY_GROUP_IDS = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, REMOTE_FUNCTION, VPC_CONFIG, SECURITY_GROUP_IDS
)
REMOTE_FUNCTION_ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION = _simple_path(
    SAGEMAKER,
    PYTHON_SDK,
    MODULES,
    REMOTE_FUNCTION,
    ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION,
)
MONITORING_SCHEDULE_INTER_CONTAINER_ENCRYPTION_PATH = _simple_path(
    SAGEMAKER,
    MONITORING_SCHEDULE,
    MONITORING_SCHEDULE_CONFIG,
    MONITORING_JOB_DEFINITION,
    NETWORK_CONFIG,
    ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION,
)
AUTO_ML_INTER_CONTAINER_ENCRYPTION_PATH = _simple_path(
    SAGEMAKER,
    AUTO_ML_JOB,
    AUTO_ML_JOB_CONFIG,
    SECURITY_CONFIG,
    ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION,
)
PROCESSING_JOB_ENVIRONMENT_PATH = _simple_path(SAGEMAKER, PROCESSING_JOB, ENVIRONMENT)
PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH = _simple_path(
    SAGEMAKER, PROCESSING_JOB, NETWORK_CONFIG, ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION
)
TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH = _simple_path(
    SAGEMAKER, TRAINING_JOB, ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION
)
SESSION_DEFAULT_S3_BUCKET_PATH = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, SESSION, DEFAULT_S3_BUCKET
)
SESSION_DEFAULT_S3_OBJECT_KEY_PREFIX_PATH = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, SESSION, DEFAULT_S3_OBJECT_KEY_PREFIX
)
ESTIMATOR_DEBUG_HOOK_CONFIG_PATH = _simple_path(
    SAGEMAKER, PYTHON_SDK, MODULES, ESTIMATOR, DEBUG_HOOK_CONFIG
)


SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    TYPE: OBJECT,
    "required": [SCHEMA_VERSION],
    ADDITIONAL_PROPERTIES: False,
    "definitions": {
        "roleArn": {
            # Schema for IAM Role. This includes a Regex validator.
            TYPE: "string",
            "pattern": r"^arn:aws[a-z\-]*:iam::\d{12}:role/?[a-zA-Z_0-9+=,.@\-_/]+$",
            "minLength": 20,
            "maxLength": 2048,
        },
        "kmsKeyId": {
            TYPE: "string",
            "maxLength": 2048,
        },
        "securityGroupId": {
            TYPE: "string",
            "pattern": r"[-0-9a-zA-Z]+",
            "maxLength": 32,
        },
        "subnet": {
            TYPE: "string",
            "pattern": r"[-0-9a-zA-Z]+",
            "maxLength": 32,
        },
        "vpcConfig": {
            # Schema for VPC Configs.
            # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference
            # /API_VpcConfig.html
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PROPERTIES: {
                SECURITY_GROUP_IDS: {
                    TYPE: "array",
                    "items": {"$ref": "#/definitions/securityGroupId"},
                    "minItems": 1,
                    "maxItems": 5,
                },
                SUBNETS: {
                    TYPE: "array",
                    "items": {"$ref": "#/definitions/subnet"},
                    "minItems": 1,
                    "maxItems": 16,
                },
            },
        },
        "productionVariant": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PROPERTIES: {
                CORE_DUMP_CONFIG: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                }
            },
        },
        "validationProfile": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PROPERTIES: {
                TRANSFORM_JOB_DEFINITION: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        ENVIRONMENT: {
                            "$ref": "#/definitions/environment-Length10240-Properties16",
                        },
                        TRANSFORM_OUTPUT: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        TRANSFORM_RESOURCES: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {VOLUME_KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                    },
                }
            },
        },
        "processingInput": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PROPERTIES: {
                DATASET_DEFINITION: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        ATHENA_DATASET_DEFINITION: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        REDSHIFT_DATASET_DEFINITION: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"},
                                CLUSTER_ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                            },
                        },
                    },
                }
            },
        },
        # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_Tag.html
        "tags": {
            TYPE: "array",
            "items": {
                TYPE: OBJECT,
                ADDITIONAL_PROPERTIES: False,
                PROPERTIES: {
                    KEY: {
                        TYPE: "string",
                        "pattern": r"^[\w\s\d_.:/=+\-@]*$",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    VALUE: {
                        TYPE: "string",
                        "pattern": r"^[\w\s\d_.:/=+\-@]*$",
                        "minLength": 0,
                        "maxLength": 256,
                    },
                },
                "required": [KEY, VALUE],
            },
            "minItems": 0,
            "maxItems": 50,
        },
        # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html#sagemaker-CreateTrainingJob-request-Environment
        "environmentVariables": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PATTERN_PROPERTIES: {
                r"([a-zA-Z_][a-zA-Z0-9_]*){1,512}": {
                    TYPE: "string",
                    "pattern": r"[\S\s]*",
                    "maxLength": 512,
                }
            },
            "maxProperties": 48,
        },
        # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_S3DataSource.html#sagemaker-Type-S3DataSource-S3Uri
        "s3Uri": {
            TYPE: "string",
            "pattern": "^(https|s3)://([^/]+)/?(.*)$",
            "maxLength": 1024,
        },
        # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AlgorithmSpecification.html#sagemaker-Type-AlgorithmSpecification-ContainerEntrypoint
        "preExecutionCommand": {TYPE: "string", "pattern": r".*"},
        # Regex based on https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_PipelineDefinitionS3Location.html
        # except with an additional ^ and $ for the beginning and the end to closer align to
        # https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html
        "s3Bucket": {
            TYPE: "string",
            "pattern": r"^[a-z0-9][\.\-a-z0-9]{1,61}[a-z0-9]$",
            "minLength": 3,
            "maxLength": 63,
        },
        # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_MonitoringJobDefinition.html#sagemaker-Type-MonitoringJobDefinition-Environment
        "environment-Length256-Properties50": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PATTERN_PROPERTIES: {
                r"([a-zA-Z_][a-zA-Z0-9_]*){1,256}": {
                    TYPE: "string",
                    "pattern": r"[\S\s]*",
                    "maxLength": 256,
                }
            },
            "maxProperties": 50,
        },
        # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTransformJob.html#sagemaker-CreateTransformJob-request-Environment
        "environment-Length10240-Properties16": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PATTERN_PROPERTIES: {
                r"[a-zA-Z_][a-zA-Z0-9_]{0,1023}": {
                    TYPE: "string",
                    "pattern": r"[\S\s]*",
                    "maxLength": 10240,
                }
            },
            "maxProperties": 16,
        },
        # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ContainerDefinition.html#sagemaker-Type-ContainerDefinition-Environment
        "environment-Length1024-Properties16": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PATTERN_PROPERTIES: {
                r"([a-zA-Z_][a-zA-Z0-9_]*){1,1024}": {
                    TYPE: "string",
                    "pattern": r"[\S\s]*",
                    "maxLength": 1024,
                }
            },
            "maxProperties": 16,
        },
        # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html#sagemaker-CreateProcessingJob-request-Environment
        "environment-Length256-Properties100": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PATTERN_PROPERTIES: {
                r"([a-zA-Z_][a-zA-Z0-9_]*){1,256}": {
                    TYPE: "string",
                    "pattern": r"[\S\s]*",
                    "maxLength": 256,
                }
            },
            "maxProperties": 100,
        },
        # Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html#sagemaker-CreateTrainingJob-request-Environment
        "environment-Length512-Properties48": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PATTERN_PROPERTIES: {
                r"([a-zA-Z_][a-zA-Z0-9_]*){1,512}": {
                    TYPE: "string",
                    "pattern": r"[\S\s]*",
                    "maxLength": 512,
                }
            },
            "maxProperties": 48,
        },
        "container": {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PROPERTIES: {
                ENVIRONMENT: {"$ref": "#/definitions/environment-Length1024-Properties16"},
            },
        },
        "containers": {
            TYPE: "array",
            "items": {"$ref": "#/definitions/container"},
            # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ContainerDefinition.html
            # According to the API docs, the array can have maximum of 15 items.
            "minItems": 0,
            "maxItems": 15,
        },
        "modelPackageContainers": {
            TYPE: "array",
            "items": {"$ref": "#/definitions/container"},
            # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ModelPackageContainerDefinition.html
            # According to the API docs, the array can have minimum of 1 and maximum of 15 items.
            "minItems": 1,
            "maxItems": 15,
        },
    },
    PROPERTIES: {
        SCHEMA_VERSION: {
            TYPE: "string",
            # Currently we support only one schema version (1.0).
            # In the future this might change if we introduce any breaking changes.
            # So added an enum as a validator.
            "enum": ["1.0"],
            "description": "The schema version of the document.",
        },
        CUSTOM_PARAMETERS: {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PATTERN_PROPERTIES: {
                r"^[\w\s\d_.:/=+\-@]+$": {TYPE: "string"},
            },
        },
        SAGEMAKER: {
            TYPE: OBJECT,
            ADDITIONAL_PROPERTIES: False,
            PROPERTIES: {
                PYTHON_SDK: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        MODULES: {
                            # Any SageMaker Python SDK specific configuration will be added here.
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                SESSION: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {
                                        DEFAULT_S3_BUCKET: {
                                            "description": "sets `default_bucket` of Session",
                                            "$ref": "#/definitions/s3Bucket",
                                        },
                                        DEFAULT_S3_OBJECT_KEY_PREFIX: {
                                            "description": (
                                                "sets `default_bucket_prefix` of Session"
                                            ),
                                            TYPE: "string",
                                            # S3 guidelines:
                                            # https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html
                                            # Note that the PythonSDK at the time of writing
                                            # tends to collapse multiple "/" in a row to one "/"
                                            # (even though S3 allows multiple "/" in a row)
                                            "minLength": 1,
                                            "maxLength": 1024,
                                        },
                                    },
                                },
                                ESTIMATOR: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {
                                        DEBUG_HOOK_CONFIG: {
                                            TYPE: "boolean",
                                            "description": (
                                                "Sets a boolean for `debugger_hook_config` of"
                                                "Estimator which will be then used for training job"
                                                "API call. Today, the config_schema doesn't support"
                                                "a dictionary as a valid value to be provided."
                                                "In the future to add support for DebugHookConfig"
                                                "as a dictionary, schema should be added under"
                                                "the config path `SageMaker.TrainingJob` instead of"
                                                "here, since the TrainingJob API supports"
                                                "DebugHookConfig as a dictionary, we can add"
                                                "a schema for it at API level."
                                            ),
                                        },
                                    },
                                },
                                REMOTE_FUNCTION: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {
                                        DEPENDENCIES: {TYPE: "string"},
                                        PRE_EXECUTION_COMMANDS: {
                                            TYPE: "array",
                                            "items": {"$ref": "#/definitions/preExecutionCommand"},
                                        },
                                        PRE_EXECUTION_SCRIPT: {TYPE: "string"},
                                        ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: {
                                            TYPE: "boolean"
                                        },
                                        ENVIRONMENT_VARIABLES: {
                                            "$ref": "#/definitions/environmentVariables"
                                        },
                                        IMAGE_URI: {TYPE: "string"},
                                        INCLUDE_LOCAL_WORKDIR: {TYPE: "boolean"},
                                        INSTANCE_TYPE: {TYPE: "string"},
                                        JOB_CONDA_ENV: {TYPE: "string"},
                                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                                        S3_KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"},
                                        S3_ROOT_URI: {"$ref": "#/definitions/s3Uri"},
                                        TAGS: {"$ref": "#/definitions/tags"},
                                        VOLUME_KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"},
                                        VPC_CONFIG: {"$ref": "#/definitions/vpcConfig"},
                                    },
                                },
                            },
                        },
                    },
                },
                # Feature Group
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateFeatureGroup
                # .html
                FEATURE_GROUP: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        OFFLINE_STORE_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                S3_STORAGE_CONFIG: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                                }
                            },
                        },
                        ONLINE_STORE_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                SECURITY_CONFIG: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                                }
                            },
                        },
                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Monitoring Schedule
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateMonitoringSchedule.html
                MONITORING_SCHEDULE: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        MONITORING_SCHEDULE_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                MONITORING_JOB_DEFINITION: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {
                                        ENVIRONMENT: {
                                            "$ref": (
                                                "#/definitions/environment-Length256-Properties50"
                                            ),
                                        },
                                        MONITORING_OUTPUT_CONFIG: {
                                            TYPE: OBJECT,
                                            ADDITIONAL_PROPERTIES: False,
                                            PROPERTIES: {
                                                KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}
                                            },
                                        },
                                        MONITORING_RESOURCES: {
                                            TYPE: OBJECT,
                                            ADDITIONAL_PROPERTIES: False,
                                            PROPERTIES: {
                                                CLUSTER_CONFIG: {
                                                    TYPE: OBJECT,
                                                    ADDITIONAL_PROPERTIES: False,
                                                    PROPERTIES: {
                                                        VOLUME_KMS_KEY_ID: {
                                                            "$ref": "#/definitions/kmsKeyId"
                                                        }
                                                    },
                                                }
                                            },
                                        },
                                        NETWORK_CONFIG: {
                                            TYPE: OBJECT,
                                            ADDITIONAL_PROPERTIES: False,
                                            PROPERTIES: {
                                                ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: {
                                                    TYPE: "boolean"
                                                },
                                                ENABLE_NETWORK_ISOLATION: {TYPE: "boolean"},
                                                VPC_CONFIG: {"$ref": "#/definitions/vpcConfig"},
                                            },
                                        },
                                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                                    },
                                }
                            },
                        },
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Endpoint
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpoint.html
                ENDPOINT: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {TAGS: {"$ref": "#/definitions/tags"}},
                },
                # Endpoint Config
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpointConfig.html
                # Note: there is a separate API for creating Endpoints.
                ENDPOINT_CONFIG: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        ASYNC_INFERENCE_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                OUTPUT_CONFIG: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                                }
                            },
                        },
                        DATA_CAPTURE_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"},
                        PRODUCTION_VARIANTS: {
                            TYPE: "array",
                            "items": {"$ref": "#/definitions/productionVariant"},
                        },
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Auto ML
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateAutoMLJob.html
                AUTO_ML_JOB: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        AUTO_ML_JOB_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                SECURITY_CONFIG: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {
                                        ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: {
                                            TYPE: "boolean"
                                        },
                                        VOLUME_KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"},
                                        VPC_CONFIG: {"$ref": "#/definitions/vpcConfig"},
                                    },
                                }
                            },
                        },
                        OUTPUT_DATA_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Transform Job
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTransformJob.html
                TRANSFORM_JOB: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        DATA_CAPTURE_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        ENVIRONMENT: {"$ref": "#/definitions/environment-Length10240-Properties16"},
                        TRANSFORM_OUTPUT: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        TRANSFORM_RESOURCES: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {VOLUME_KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Compilation Job
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateCompilationJob
                # .html
                COMPILATION_JOB: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        OUTPUT_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                        VPC_CONFIG: {"$ref": "#/definitions/vpcConfig"},
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Pipeline
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreatePipeline.html
                PIPELINE: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Model
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateModel.html
                # Note: It is recommended that customers should use either of
                # [Containers] or [PrimaryContainer] but not both
                # Since SDK will throw a ValueError as both can't be passed in.
                MODEL: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        CONTAINERS: {"$ref": "#/definitions/containers"},
                        ENABLE_NETWORK_ISOLATION: {TYPE: "boolean"},
                        EXECUTION_ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                        PRIMARY_CONTAINER: {"$ref": "#/definitions/container"},
                        VPC_CONFIG: {"$ref": "#/definitions/vpcConfig"},
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Model Package
                # https://docs.aws.amazon.com/sagemaker/latest
                # /APIReference/API_CreateModelPackage.html
                # At present, we are not supporting [Environment] defaults for
                # [AdditionalInfereceSpecifcations] Since this parameter is not supported
                # in the SDK, we are not adding it to the config schema as well.
                MODEL_PACKAGE: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        INFERENCE_SPECIFICATION: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                CONTAINERS: {"$ref": "#/definitions/modelPackageContainers"}
                            },
                        },
                        VALIDATION_SPECIFICATION: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                VALIDATION_PROFILES: {
                                    TYPE: "array",
                                    "items": {"$ref": "#/definitions/validationProfile"},
                                    # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ModelPackageValidationSpecification.html
                                    # According to the API docs, This array should have exactly 1
                                    # item.
                                    "minItems": 1,
                                    "maxItems": 1,
                                },
                                VALIDATION_ROLE: {"$ref": "#/definitions/roleArn"},
                            },
                        },
                    },
                },
                # Processing Job
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateProcessingJob.html
                PROCESSING_JOB: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        ENVIRONMENT: {"$ref": "#/definitions/environment-Length256-Properties100"},
                        NETWORK_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: {TYPE: "boolean"},
                                ENABLE_NETWORK_ISOLATION: {TYPE: "boolean"},
                                VPC_CONFIG: {"$ref": "#/definitions/vpcConfig"},
                            },
                        },
                        PROCESSING_INPUTS: {
                            TYPE: "array",
                            "items": {"$ref": "#/definitions/processingInput"},
                            "minItems": 0,
                            "maxItems": 10,
                        },
                        PROCESSING_OUTPUT_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        PROCESSING_RESOURCES: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {
                                CLUSTER_CONFIG: {
                                    TYPE: OBJECT,
                                    ADDITIONAL_PROPERTIES: False,
                                    PROPERTIES: {
                                        VOLUME_KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}
                                    },
                                }
                            },
                        },
                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Training Job
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html
                # Please note that we currently support 'DebugHookConfig' as a boolean value
                # which can be provided under [SageMaker.PythonSDK.Modules.Estimator] config path.
                # As of today, config_schema does not support the dict as a valid value to be
                # provided. In case, we decide to support it in the future, we can add a new schema
                # for it under [SageMaker.TrainingJob] config path.
                TRAINING_JOB: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION: {TYPE: "boolean"},
                        ENABLE_NETWORK_ISOLATION: {TYPE: "boolean"},
                        ENVIRONMENT: {"$ref": "#/definitions/environment-Length512-Properties48"},
                        OUTPUT_DATA_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        RESOURCE_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {VOLUME_KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        PROFILER_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {DISABLE_PROFILER: {TYPE: "boolean"}},
                        },
                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                        VPC_CONFIG: {"$ref": "#/definitions/vpcConfig"},
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
                # Edge Packaging Job
                # https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEdgePackagingJob.html
                EDGE_PACKAGING_JOB: {
                    TYPE: OBJECT,
                    ADDITIONAL_PROPERTIES: False,
                    PROPERTIES: {
                        OUTPUT_CONFIG: {
                            TYPE: OBJECT,
                            ADDITIONAL_PROPERTIES: False,
                            PROPERTIES: {KMS_KEY_ID: {"$ref": "#/definitions/kmsKeyId"}},
                        },
                        RESOURCE_KEY: {"$ref": "#/definitions/kmsKeyId"},
                        ROLE_ARN: {"$ref": "#/definitions/roleArn"},
                        TAGS: {"$ref": "#/definitions/tags"},
                    },
                },
            },
        },
    },
}
